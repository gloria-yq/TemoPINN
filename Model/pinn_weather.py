
import math
import os
import itertools
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from utils.util import AverageMeter, get_logger
from Model.itransformer_block import ITransformerRegressor

try:
    from contextlib import nullcontext as _nullcontext
except ImportError:
    @contextmanager
    def _nullcontext():
        yield


def build_mlp(input_dim, hidden_dim, output_dim, num_layers, dropout=0.1):
    layers = []
    depth = max(2, num_layers)
    for i in range(depth):
        in_dim = input_dim if i == 0 else hidden_dim
        out_dim = output_dim if i == depth - 1 else hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        if i < depth - 1:
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FlexibleRegressor(nn.Module):

    def __init__(self, model_type, seq_length, input_features, hidden_dim,
                 num_layers, output_dim, dropout=0.1, time_conditioned=True,
                 transformer_heads=4):
        super().__init__()
        self.model_type = model_type
        self.seq_length = seq_length
        self.input_features = input_features
        self.time_conditioned = time_conditioned
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        extra_t = 1 if time_conditioned else 0
        seq_in_dim = input_features + extra_t

        if model_type == 'mlp':
            flat_dim = seq_length * input_features + extra_t
            self.encoder = build_mlp(flat_dim, hidden_dim, output_dim, num_layers, dropout)
            self.head = None
            return

        if model_type == 'cnn':
            channels = []
            in_ch = seq_in_dim
            depth = max(2, num_layers)
            for _ in range(depth):
                channels.append(nn.Conv1d(in_ch, hidden_dim, kernel_size=3, padding=1))
                channels.append(nn.ReLU())
                channels.append(nn.Dropout(dropout))
                in_ch = hidden_dim
            self.encoder = nn.Sequential(*channels)

        elif model_type in ['gru', 'lstm']:
            rnn_cls = nn.GRU if model_type == 'gru' else nn.LSTM
            self.encoder = rnn_cls(
                input_size=seq_in_dim,
                hidden_size=hidden_dim,
                num_layers=max(1, num_layers),
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self._is_rnn = True

        elif model_type == 'transformer':
            heads = max(1, min(transformer_heads, hidden_dim))
            while hidden_dim % heads != 0 and heads > 1:
                heads -= 1
            self.input_proj = nn.Linear(seq_in_dim, hidden_dim)
            self.position = SinusoidalPositionalEncoding(hidden_dim, seq_length)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu',
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, num_layers))
        else:
            raise ValueError(f'Unsupported model_type: {model_type}')

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def _append_time(self, x, t):
        if not self.time_conditioned:
            return x
        t_rep = t.unsqueeze(1).expand(-1, x.size(1), -1)
        return torch.cat([x, t_rep], dim=-1)

    def forward(self, x, t=None):
        if self.model_type == 'mlp':
            flat = x.reshape(x.size(0), -1)
            if self.time_conditioned:
                if t is None:
                    raise ValueError('time-conditioned MLP requires t')
                flat = torch.cat([flat, t], dim=1)
            return self.encoder(flat)

        x = self._append_time(x, t)

        if self.model_type == 'cnn':
            hidden = self.encoder(x.transpose(1, 2)).mean(dim=-1)
        elif self.model_type in ['gru', 'lstm']:
            out, _ = self.encoder(x)
            hidden = out[:, -1, :]
        elif self.model_type == 'transformer':
            hidden = self.input_proj(x)
            hidden = self.position(hidden)
            hidden = self.encoder(hidden)
            hidden = hidden[:, -1, :]
        else:
            raise ValueError(f'Unsupported model_type: {self.model_type}')

        return self.head(hidden)


class Solution_u(nn.Module):
    def __init__(self, model_type, seq_length, num_features, hidden_dim,
                 num_layers, num_targets, dropout=0.1, transformer_heads=4):
        super().__init__()
        self.model_type = model_type

        if model_type in ('itransformer_time', 'itransformer_var'):
            ffn_type = 'time' if model_type == 'itransformer_time' else 'variate'
            self.regressor = ITransformerRegressor(
                seq_length=seq_length,
                input_features=num_features,
                d_model=hidden_dim,
                n_heads=transformer_heads,
                num_layers=num_layers,
                output_dim=num_targets,
                ffn_type=ffn_type,
                dropout=dropout,
                time_conditioned=True,
            )
        else:
            self.regressor = FlexibleRegressor(
                model_type=model_type,
                seq_length=seq_length,
                input_features=num_features,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=num_targets,
                dropout=dropout,
                time_conditioned=True,
                transformer_heads=transformer_heads,
            )

    def forward(self, x, t):
        return self.regressor(x, t)


class DynamicalF(nn.Module):
    def __init__(self, model_type, seq_length, num_features, num_targets,
                 hidden_dim, num_layers, dropout=0.1, transformer_heads=4):
        super().__init__()
        self.model_type = model_type
        self.seq_length = seq_length
        self.num_features = num_features
        self.num_targets = num_targets

        if model_type == 'mlp':
            input_dim = 2 * seq_length * num_features + 1 + 2 * num_targets
            self.regressor = build_mlp(
                input_dim, hidden_dim, num_targets, num_layers, dropout
            )
        elif model_type in ('itransformer_time', 'itransformer_var'):
            seq_features = 2 * num_features + 1 + 2 * num_targets
            ffn_type = 'time' if model_type == 'itransformer_time' else 'variate'
            self.regressor = ITransformerRegressor(
                seq_length=seq_length,
                input_features=seq_features,
                d_model=hidden_dim,
                n_heads=transformer_heads,
                num_layers=num_layers,
                output_dim=num_targets,
                ffn_type=ffn_type,
                dropout=dropout,
                time_conditioned=False,
            )
        else:
            seq_features = 2 * num_features + 1 + 2 * num_targets
            self.regressor = FlexibleRegressor(
                model_type=model_type,
                seq_length=seq_length,
                input_features=seq_features,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=num_targets,
                dropout=dropout,
                time_conditioned=False,
                transformer_heads=transformer_heads,
            )

    def forward(self, x, t, u, du_dx, du_dt):
        if self.model_type == 'mlp':
            flat = torch.cat([
                x.reshape(x.size(0), -1),
                t,
                u,
                du_dx.reshape(du_dx.size(0), -1),
                du_dt,
            ], dim=1)
            return self.regressor(flat)

        global_cond = torch.cat([t, u, du_dt], dim=1)
        global_cond = global_cond.unsqueeze(1).expand(-1, x.size(1), -1)
        seq_in = torch.cat([x, du_dx, global_cond], dim=2)
        return self.regressor(seq_in)


class WeatherPINN(nn.Module):
    def __init__(self, args, num_features, num_targets, target_names):
        super().__init__()
        self.args = args
        self.mode = args.mode
        self.device = args.device
        self.num_features = num_features
        self.num_targets = num_targets
        self.target_names = target_names
        self.alpha = args.alpha
        self.beta = args.beta
        self.f_model = args.f_model
        self.g_model = args.g_model

        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(args.save_folder, f'train_{ts}.txt')
        self.logger = get_logger(log_path)

    
        f_hidden = args.f_hidden_dim or args.hidden_dim
        if args.f_model == 'mlp':
            f_hidden = f_hidden * 2   

        self.solution_u = Solution_u(
            model_type=args.f_model,
            seq_length=args.seq_length,
            num_features=num_features,
            hidden_dim=f_hidden,
            num_layers=args.num_layers,
            num_targets=num_targets,
            dropout=args.dropout,
            transformer_heads=args.transformer_heads,
        ).to(self.device)

        self.dynamical_F = None
        if self.mode in ['B', 'C']:
    
            g_hidden = args.g_hidden_dim or max(256, args.hidden_dim // 2)
            if args.g_model == 'mlp':
                g_hidden = g_hidden * 2
            g_layers = max(2, min(3, args.num_layers))


            if args.g_model in ('gru', 'lstm') or args.f_model in ('gru', 'lstm'):
                self.logger.info(
                    "[WARNING]"
                )

            self.dynamical_F = DynamicalF(
                model_type=args.g_model,
                seq_length=args.seq_length,
                num_features=num_features,
                num_targets=num_targets,
                hidden_dim=g_hidden,
                num_layers=g_layers,
                dropout=args.dropout,
                transformer_heads=args.transformer_heads,
            ).to(self.device)

        self.optimizer_F = torch.optim.Adam(self.solution_u.parameters(), lr=args.lr)
        self.optimizer_G = None
        if self.dynamical_F is not None:
            self.optimizer_G = torch.optim.Adam(self.dynamical_F.parameters(), lr=args.lr * 0.1)

        _ATTN_MODELS = {'transformer', 'itransformer_time', 'itransformer_var'}
        is_transformer = (self.f_model in _ATTN_MODELS or self.g_model in _ATTN_MODELS)
        warmup_epochs = max(5, args.epochs // 10)  

        if is_transformer:
          
            warmup_F = torch.optim.lr_scheduler.LinearLR(
                self.optimizer_F, start_factor=0.01, total_iters=warmup_epochs)
            decay_F = torch.optim.lr_scheduler.StepLR(
                self.optimizer_F, step_size=30, gamma=0.5)
            self.scheduler_F = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer_F, schedulers=[warmup_F, decay_F],
                milestones=[warmup_epochs])
        else:
            self.scheduler_F = torch.optim.lr_scheduler.StepLR(
                self.optimizer_F, step_size=30, gamma=0.5)

        self.scheduler_G = None
        if self.optimizer_G is not None:
            if is_transformer:
                warmup_G = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer_G, start_factor=0.01, total_iters=warmup_epochs)
                decay_G = torch.optim.lr_scheduler.StepLR(
                    self.optimizer_G, step_size=30, gamma=0.5)
                self.scheduler_G = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer_G, schedulers=[warmup_G, decay_G],
                    milestones=[warmup_epochs])
            else:
                self.scheduler_G = torch.optim.lr_scheduler.StepLR(
                    self.optimizer_G, step_size=30, gamma=0.5)

        self.loss_func = nn.MSELoss()
        self.best_model = None

        self.logger.info(
            f"Mode={self.mode} | F={self.f_model} | G={self.g_model} | "
            f"Features={num_features} | Targets={num_targets}"
        )
        if self.mode in ['B', 'C']:
            self.logger.info(f"alpha={self.alpha} | g_input=1")
        if self.mode in ['C', 'D']:
            self.logger.info(f"beta={self.beta}")

    def _pinn_autograd_ctx(self):
        need_rnn = self.f_model in ('gru', 'lstm') or self.g_model in ('gru', 'lstm')
        _ATTN_MODELS = {'transformer', 'itransformer_time', 'itransformer_var'}
        need_attn = self.f_model in _ATTN_MODELS or self.g_model in _ATTN_MODELS

        @contextmanager
        def _combined():
            rnn_ctx = torch.backends.cudnn.flags(enabled=False) if need_rnn else _nullcontext(
            if need_attn:
                try:
                    from torch.nn.attention import sdpa_kernel, SDPBackend
                    attn_ctx = sdpa_kernel(SDPBackend.MATH)
                except (ImportError, AttributeError):

                        attn_ctx = torch.backends.cuda.sdp_kernel(
                            enable_flash=False, enable_math=True, enable_mem_efficient=False)
                    except AttributeError:
                        attn_ctx = _nullcontext()
            else:
                attn_ctx = _nullcontext()

            with rnn_ctx:
                with attn_ctx:
                    yield

        return _combined()

    def forward_pinn(self, x, t_raw):
        x_req = x.detach().clone().requires_grad_(True)
        t = t_raw.detach().clone().requires_grad_(True)

        with self._pinn_autograd_ctx():
            u = self.solution_u(x_req, t)

            du_dt_list, du_dx_list = [], []
            for i in range(self.num_targets):
                scalar = u[:, i].sum()
                du_dt_list.append(torch.autograd.grad(
                    scalar, t, create_graph=True, retain_graph=True)[0])
                du_dx_list.append(torch.autograd.grad(
                    scalar, x_req, create_graph=True, retain_graph=True)[0])

            du_dt = torch.cat(du_dt_list, dim=1)
            du_dx = du_dx_list[0] if self.num_targets == 1 \
                else torch.stack(du_dx_list).mean(0)

            g_out = self.dynamical_F(x_req, t, u, du_dx, du_dt)
            f = du_dt - g_out
        return u, f

    def train_one_epoch(self, epoch, loader, period_loader=None):
        self.train()
        m_data = AverageMeter()
        m_pde = AverageMeter()
        m_period = AverageMeter()

        period_iter = itertools.cycle(period_loader) if period_loader is not None else None

        for step, (x, t, y) in enumerate(loader):
            x = x.to(self.device)
            t = t.to(self.device)
            y = y.to(self.device)

            if self.mode in ['A', 'D']:
                u = self.solution_u(x, t)
                loss_data = self.loss_func(u, y)
                loss_pde = torch.tensor(0.0, device=self.device)
                loss = loss_data

                loss_period = torch.tensor(0.0, device=self.device)
                if self.mode == 'D' and period_iter is not None:
                    px1, pt1, px2, pt2 = next(period_iter)
                    px1, pt1 = px1.to(self.device), pt1.to(self.device)
                    px2, pt2 = px2.to(self.device), pt2.to(self.device)
                    pu1 = self.solution_u(px1, pt1)
                    pu2 = self.solution_u(px2, pt2)
                    loss_period = self.loss_func(pu1, pu2)
                    loss = loss + self.beta * loss_period
            else:
                compute_pde = (step % self.args.pde_freq == 0)

                if compute_pde:
                    u, f = self.forward_pinn(x, t)
                    loss_data = self.loss_func(u, y)
                    loss_pde = self.loss_func(f, torch.zeros_like(f))
                    loss = loss_data + self.alpha * loss_pde
                else:
                    u = self.solution_u(x, t)
                    loss_data = self.loss_func(u, y)
                    loss_pde = torch.tensor(0.0, device=self.device)
                    loss = loss_data

                loss_period = torch.tensor(0.0, device=self.device)
                if self.mode == 'C' and period_iter is not None and compute_pde:
                    px1, pt1, px2, pt2 = next(period_iter)
                    px1, pt1 = px1.to(self.device), pt1.to(self.device)
                    px2, pt2 = px2.to(self.device), pt2.to(self.device)
                    pu1 = self.solution_u(px1, pt1)
                    pu2 = self.solution_u(px2, pt2)
                    loss_period = self.loss_func(pu1, pu2)
                    loss = loss + self.beta * loss_period

            self.optimizer_F.zero_grad()
            if self.optimizer_G is not None:
                self.optimizer_G.zero_grad()
            loss.backward()
            self.optimizer_F.step()
            if self.optimizer_G is not None:
                self.optimizer_G.step()

            m_data.update(loss_data.item())
            m_pde.update(loss_pde.item())
            m_period.update(loss_period.item())

            if (step + 1) % 50 == 0:
                parts = [f"[ep:{epoch} it:{step + 1}] data:{loss_data.item():.6f}"]
                if self.mode in ['B', 'C']:
                    parts.append(f"alpha*pde:{self.alpha * loss_pde.item():.6f}")
                if self.mode in ['C', 'D']:
                    parts.append(f"beta*period:{self.beta * loss_period.item():.6f}")
                self.logger.info(' '.join(parts))

        return m_data.avg, m_pde.avg, m_period.avg

    @torch.no_grad()
    def evaluate(self, loader):
        self.eval()
        trues, preds = [], []
        for x, t, y in loader:
            x = x.to(self.device)
            t = t.to(self.device)
            u = self.solution_u(x, t)
            trues.append(y.numpy())
            preds.append(u.cpu().numpy())
        return np.concatenate(trues), np.concatenate(preds)

    def compute_metrics(self, true, pred, scaler_info):
        means = scaler_info['target_means']
        stds = scaler_info['target_stds']
        unit_map = {
            't2m': 'K',
            't_850': 'K',
            'z': 'm^2/s^2',
            'z_500': 'm^2/s^2',
            'u10': 'm/s',
            'v10': 'm/s',
            'tp': 'm',
            'tcc': '[0,1]',
            'tisr': 'J/m^2',
        }

        metrics = {}
        for i, name in enumerate(self.target_names):
            t_phys = true[:, i] * stds[name] + means[name]
            p_phys = pred[:, i] * stds[name] + means[name]
            rmse = np.sqrt(np.mean((t_phys - p_phys) ** 2))
            metrics[name] = {'RMSE': rmse, 'unit': unit_map.get(name, '')}
        return metrics

    def run_training(self, loaders, period_loader, scaler_info):
        min_val_mse = float('inf')
        patience = 0
        history = {
            'epoch': [], 'data': [], 'pde': [], 'period': [],
            'valid_mse': [], 'test_rmse': []
        }

        for epoch in range(1, self.args.epochs + 1):
            ld, lp, lper = self.train_one_epoch(epoch, loaders['train'], period_loader)
            self.scheduler_F.step()
            if self.scheduler_G is not None:
                self.scheduler_G.step()

            lr = self.optimizer_F.param_groups[0]['lr']
            parts = [f"[Train] ep:{epoch} lr:{lr:.6f} data:{ld:.6f}"]
            if self.mode in ['B', 'C']:
                parts.append(f"pde:{lp:.2e}")
            if self.mode in ['C', 'D']:
                parts.append(f"period:{lper:.2e}")
            self.logger.info(' '.join(parts))

            true_v, pred_v = self.evaluate(loaders['valid'])
            val_mse = np.mean((true_v - pred_v) ** 2)
            self.logger.info(f"[Valid] ep:{epoch} MSE:{val_mse:.8f}")

            history['epoch'].append(epoch)
            history['data'].append(ld)
            history['pde'].append(lp)
            history['period'].append(lper)
            history['valid_mse'].append(val_mse)

            test_rmse = None
            if val_mse < min_val_mse:
                min_val_mse = val_mse
                patience = 0
                self.best_model = {
                    'solution_u': {k: v.cpu().clone() for k, v in self.solution_u.state_dict().items()}
                }
                if self.dynamical_F is not None:
                    self.best_model['dynamical_F'] = {
                        k: v.cpu().clone() for k, v in self.dynamical_F.state_dict().items()
                    }

                true_t, pred_t = self.evaluate(loaders['test_72h'])
                metrics = self.compute_metrics(true_t, pred_t, scaler_info)
                rmses = []
                for name, item in metrics.items():
                    rmse = item['RMSE']
                    rmses.append(rmse)
                    fmt = f"{rmse:.4e}" if rmse < 0.01 else f"{rmse:.4f}"
                    self.logger.info(f"[Test 72h] {name} RMSE: {fmt} {item['unit']}")
                if rmses:
                    test_rmse = float(np.mean(rmses))
            else:
                patience += 1

            history['test_rmse'].append(test_rmse)

            if patience >= self.args.early_stop:
                self.logger.info(f"Early stop at epoch {epoch}")
                break

        if self.best_model and self.args.save_folder:
            torch.save(self.best_model, os.path.join(self.args.save_folder, 'model.pth'))

        self._plot_curves(history)

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

    def _plot_curves(self, history):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.info("no matplotlib")
            return

        epochs = history['epoch']
        if not epochs:
            return

        has_pde = any(v > 0 for v in history['pde'])
        has_period = any(v > 0 for v in history['period'])
        n_plots = 2 + int(has_pde) + int(has_period)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]

        title = f"Mode {self.mode} | F={self.f_model} | G={self.g_model} | seed={self.args.seed}"
        if self.mode in ['B', 'C']:
            title += f" | alpha={self.alpha}"
        if self.mode in ['C', 'D']:
            title += f" | beta={self.beta}"
        valid_rmses = [r for r in history['test_rmse'] if r is not None]
        if valid_rmses:
            title += f" | Best RMSE={min(valid_rmses):.4f}"
        fig.suptitle(title, fontsize=12, fontweight='bold')

        idx = 0
        axes[idx].plot(epochs, history['data'], '#1f77b4', linewidth=1.5)
        axes[idx].set_ylabel('Data Loss')
        axes[idx].grid(True, alpha=0.3)
        idx += 1

        if has_pde:
            axes[idx].plot(epochs, history['pde'], '#d62728', linewidth=1.5)
            axes[idx].set_ylabel('PDE Loss')
            axes[idx].set_yscale('log')
            axes[idx].grid(True, alpha=0.3)
            idx += 1

        if has_period:
            axes[idx].plot(epochs, history['period'], '#2ca02c', linewidth=1.5)
            axes[idx].set_ylabel('Period Loss')
            axes[idx].grid(True, alpha=0.3)
            idx += 1

        axes[idx].plot(epochs, history['valid_mse'], '#9467bd', linewidth=1.5, marker='.', markersize=3)
        axes[idx].set_ylabel('Valid MSE')
        axes[idx].set_xlabel('Epoch')
        axes[idx].grid(True, alpha=0.3)

        best_idx = int(np.argmin(history['valid_mse']))
        axes[idx].axvline(epochs[best_idx], color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(self.args.save_folder, f'loss_curves_{ts}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Loss curves saved: {out_path}")
