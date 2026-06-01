"""
天气数据加载器 
输入: CSV 
输出: DataLoader → (X, T, Y)
  X: [B, seq_len, num_features]  
  T: [B, 1]                      
  Y: [B, num_targets]            
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

SURFACE_VARS = ['z', 'u10', 'v10', 't2m', 'tp', 'tcc', 'tisr']
PRESSURE_PREFIXES = ['z_', 't_', 'u_', 'v_', 'q_', 'r_', 'vo_', 'pv_']
STATIC_VARS = ['orography', 'lsm', 'slt', 'lat2d', 'lon2d']
YEAR_HOURS = 8760


def classify_columns(columns):
    surface, pressure, static = [], [], []
    for c in columns:
        if c in STATIC_VARS:
            static.append(c)
        elif c in SURFACE_VARS:
            surface.append(c)
        elif any(c.startswith(p) for p in PRESSURE_PREFIXES):
            pressure.append(c)
    return surface, pressure, static


def get_feature_cols(columns, target_cols):
    surface, pressure, static = classify_columns(columns)
    feats = surface + pressure + static
    for tc in target_cols:
        if tc not in feats and tc in columns:
            feats.insert(0, tc)
    return feats, static


def extract_time_features(df, time_col):
    dt = pd.to_datetime(df[time_col])
    h, doy, m = dt.dt.hour, dt.dt.dayofyear, dt.dt.month
    return np.column_stack([
        np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12),
    ]).astype(np.float32)


def generate_demo_data(num_days=730):
    np.random.seed(42)
    n = num_days * 24
    t = np.arange(n)
    annual = 15 * np.sin(2 * np.pi * t / (365 * 24))

    data = {
        'time': pd.date_range('2020-01-01', periods=n, freq='h'),
        't2m': 288 + annual + 5 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 1.5, n),
        'z_500': 51000 + 500 * np.sin(2 * np.pi * t / (365 * 24)) + np.random.normal(0, 100, n),
        't_850': 275 + annual * 0.8 + np.random.normal(0, 2, n),
        'u10': np.random.normal(3, 2, n),
        'v10': np.random.normal(0, 1.5, n),
        'z': 50000 + np.random.normal(0, 100, n),
        'orography': np.full(n, 150.0),
        'lsm': np.full(n, 1.0),
    }
    df = pd.DataFrame(data)
    feats, static = get_feature_cols(df.columns, ['t2m', 'z_500', 't_850'])
    return df, feats, static


def create_dataloader(df, feature_cols, target_cols, args):
    static_list = [c for c in feature_cols if c in STATIC_VARS]
    tv_cols = [c for c in feature_cols if c not in static_list]

    # 填补空值
    df[tv_cols] = (df[tv_cols].interpolate('linear', limit_direction='both')
                   .ffill().bfill())

    # 时间特征
    time_col = next((c for c in ['time', 'datetime', 'Time'] if c in df.columns), None)
    if time_col:
        time_feats = extract_time_features(df, time_col)
    else:
        time_feats = None

    # 标准化特征
    scaler = StandardScaler()
    tv_data = scaler.fit_transform(df[tv_cols].values.astype(np.float32))

    static_data = None
    if static_list:
        static_data = StandardScaler().fit_transform(
            df[static_list].values.astype(np.float32))

    # 目标标准化参数
    target_means, target_stds = {}, {}
    for tc in target_cols:
        idx = tv_cols.index(tc)
        target_means[tc] = float(scaler.mean_[idx])
        target_stds[tc] = float(scaler.scale_[idx])

    targets = np.column_stack([
        (df[tc].values - target_means[tc]) / target_stds[tc]
        for tc in target_cols
    ]).astype(np.float32)

    # 拼接所有特征
    parts = [tv_data]
    if time_feats is not None:
        parts.append(time_feats)
    if static_data is not None:
        parts.append(static_data)
    features = np.concatenate(parts, axis=1)
    num_feat = features.shape[1]

    sl = args.seq_length
    fh = args.forecast_hours
    lead_times = np.array(args.lead_times_list, dtype=np.int32)
    max_lead = int(lead_times.max())
    n_lt = len(lead_times)

    n_windows = len(features) - sl - max_lead
    n_total = n_windows * n_lt

    print(f"  多超前时间采样: {n_windows} 窗口 × {n_lt} 个 lead_times = {n_total} 样本")
    print(f"  Lead times: [{lead_times[0]}...{lead_times[-1]}]h | "
          f"T 归一化: [{lead_times[0]/fh:.3f}, {lead_times[-1]/fh:.3f}]")

    n_test_win = int(n_windows * 0.2)
    n_remain_win = n_windows - n_test_win
    n_valid_win = int(n_remain_win * 0.2)
    n_train_win = n_remain_win - n_valid_win
    train_ratio = float(getattr(args, 'train_ratio', 1.0))
    if not (0 < train_ratio <= 1.0):
        raise ValueError(f'train_ratio must be in (0, 1], got {train_ratio}')
    n_train_used_win = max(1, int(n_train_win * train_ratio))

    class MultiLeadDataset(torch.utils.data.Dataset):
        def __init__(self, win_start, win_end):
            self.win_start = win_start
            self.win_end = win_end
            self.n_wins = win_end - win_start

        def __len__(self):
            return self.n_wins * n_lt

        def __getitem__(self, idx):
            win_idx = self.win_start + idx // n_lt
            lt_idx = idx % n_lt
            lt = int(lead_times[lt_idx])

            x = torch.from_numpy(features[win_idx:win_idx + sl])  # [sl, feat]
            t = torch.tensor([lt / fh], dtype=torch.float32)      # [1]
            y = torch.from_numpy(targets[win_idx + sl - 1 + lt])  # [num_targets]
            return x, t, y

    class SingleLeadDataset(torch.utils.data.Dataset):
        def __init__(self, win_start, win_end, lead_hour):
            self.win_start = win_start
            self.win_end = win_end
            self.lead_hour = lead_hour

        def __len__(self):
            return self.win_end - self.win_start

        def __getitem__(self, idx):
            win_idx = self.win_start + idx
            x = torch.from_numpy(features[win_idx:win_idx + sl])
            t = torch.tensor([self.lead_hour / fh], dtype=torch.float32)
            y = torch.from_numpy(targets[win_idx + sl - 1 + self.lead_hour])
            return x, t, y

    loaders = {
        'train': DataLoader(MultiLeadDataset(0, n_train_used_win),
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=0),
        'valid': DataLoader(MultiLeadDataset(n_train_win, n_train_win + n_valid_win),
                            batch_size=args.batch_size, shuffle=False),
        'test':  DataLoader(MultiLeadDataset(n_train_win + n_valid_win, n_windows),
                            batch_size=args.batch_size, shuffle=False),
        'test_72h': DataLoader(
            SingleLeadDataset(n_train_win + n_valid_win, n_windows, fh),
            batch_size=args.batch_size, shuffle=False),
    }


    for lead_hour in sorted(set([int(x) for x in lead_times] + [int(fh)])):
        loaders[f'test_{lead_hour}h'] = DataLoader(
            SingleLeadDataset(n_train_win + n_valid_win, n_windows, lead_hour),
            batch_size=args.batch_size, shuffle=False)

    period_loader = None
    if args.mode in ['C', 'D'] and n_train_used_win > YEAR_HOURS:
        idx1, idx2 = [], []
        for i in range(0, n_train_used_win - YEAR_HOURS, 6):
            idx1.append(i)
            idx2.append(i + YEAR_HOURS)
        # 构建 TensorDataset (仅 t=72h 的样本对)
        t_72 = torch.tensor([[fh / fh]], dtype=torch.float32).expand(len(idx1), -1)
        pX1 = torch.from_numpy(np.array([features[i:i+sl] for i in idx1]))
        pX2 = torch.from_numpy(np.array([features[i:i+sl] for i in idx2]))
        pds = TensorDataset(pX1, t_72, pX2, t_72.clone())
        period_loader = DataLoader(pds, batch_size=args.batch_size, shuffle=True)
        print(f"  周期性样本对: {len(idx1)}")

    scaler_info = {
        'target_means': target_means,
        'target_stds': target_stds,
        'target_cols': target_cols,
    }

    if train_ratio < 1.0:
        print(f"  Train ratio: {train_ratio:.2f} "
              f"({n_train_used_win}/{n_train_win} train windows)")
    print(f"  数据集: 训练 {n_train_used_win * n_lt} | "
          f"验证 {n_valid_win * n_lt} | 测试 {n_test_win * n_lt}")
    print(f"  特征: {num_feat} 维 | 窗口: {sl}h | 最大预测: {fh}h")

    return loaders, period_loader, scaler_info, num_feat
