
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer('pe', pe.unsqueeze(0))  

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x + self.pe[:, :x.size(1)]



class TimewiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1,
                 activation: str = 'gelu'):
        super().__init__()
        act = F.gelu if activation == 'gelu' else F.relu
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        return self.net(x)


class VariatewiseFFN(nn.Module):
    def __init__(self, seq_length: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        d_ff_v = max(seq_length * 2, seq_length)  # 扩展比例可调
        self.fc1 = nn.Conv1d(d_model, d_ff_v, kernel_size=1)
        self.fc2 = nn.Conv1d(d_ff_v, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        y = x.transpose(1, 2)              # [B, d_model, T]
        y = self.drop(self.act(self.fc1(y)))  # [B, d_ff_v, T]
        y = self.drop(self.fc2(y))            # [B, d_model, T]
        return y.transpose(1, 2)           # [B, T, d_model]



class ITransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int, seq_length: int,
                 dropout: float = 0.1,
                 ffn_type: str = 'time',
                 activation: str = 'gelu'):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model({d_model}) 必须能被 n_heads({n_heads}) 整除"

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN
        self.ffn_type = ffn_type
        if ffn_type == 'time':
            self.ffn = TimewiseFFN(d_model, d_ff, dropout, activation)
        elif ffn_type == 'variate':
            self.ffn = VariatewiseFFN(seq_length, d_model, dropout)
        else:
            raise ValueError(f"ffn_type 必须是 'time' 或 'variate'，得到 '{ffn_type}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
    
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + self.drop(attn_out)

        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x



class ITransformerRegressor(nn.Module):

    def __init__(
        self,
        seq_length: int,
        input_features: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        output_dim: int,
        d_ff: int = None,
        ffn_type: str = 'time',
        dropout: float = 0.1,
        activation: str = 'gelu',
        time_conditioned: bool = True,
    ):
        super().__init__()
        self.ffn_type = ffn_type
        self.time_conditioned = time_conditioned


        in_dim = input_features + (1 if time_conditioned else 0)
        d_ff = d_ff or d_model * 4

        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1

        self.input_proj = nn.Linear(in_dim, d_model)

        self.pos_enc = SinusoidalPE(d_model, max_len=seq_length + 10)

    
        self.encoder = nn.ModuleList([
            ITransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                seq_length=seq_length,
                dropout=dropout,
                ffn_type=ffn_type,
                activation=activation,
            )
            for _ in range(max(1, num_layers))
        ])

        self.norm = nn.LayerNorm(d_model)


        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
  
        if self.time_conditioned and t is not None:
            t_rep = t.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, T, 1]
            x = torch.cat([x, t_rep], dim=-1)                 # [B, T, F+1]

        x = self.input_proj(x)

        x = self.pos_enc(x)

        # Encoder
        for layer in self.encoder:
            x = layer(x)
        x = self.norm(x)

        x = x[:, -1, :]  # [B, d_model]

        # 回归头
        return self.head(x)  # [B, output_dim]
