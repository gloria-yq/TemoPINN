
import argparse
import os

import torch


MODEL_CHOICES = ['mlp', 'cnn', 'gru', 'lstm', 'transformer',
                 'itransformer_time', 'itransformer_var']


def get_args():
    p = argparse.ArgumentParser(description='PINN Weather Prediction')

    # 数据
    p.add_argument('--data_path', type=str, default=None)
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--train_ratio', type=float, default=1.0,
                   help='Use only this ratio of chronological training windows, while keeping validation/test unchanged.')
    p.add_argument('--target_cols', type=str, default='t2m',
                   help='预测目标列名，逗号分隔')
    p.add_argument('--seq_length', type=int, default=72)
    p.add_argument('--forecast_hours', type=int, default=72)
    p.add_argument('--lead_times', type=str, default='1,6,12,24,48,72',
                   help='超前小时数，如 "1-72" 或 "1,6,12,24,48,72"')
    p.add_argument('--batch_size', type=int, default=64)

    # 模型
    p.add_argument('--mode', type=str, default='A', choices=['A', 'B', 'C', 'D'],
                   help='A: L_data | B: +L_PDE | C: +L_PDE+L_period | D: +L_period')
    p.add_argument('--model', type=str, default=None, choices=MODEL_CHOICES,
                   help='若指定，则同时设置 F/G 两个网络的主干类型')
    p.add_argument('--f_model', type=str, default='mlp', choices=MODEL_CHOICES,
                   help='F 网络主干类型')
    p.add_argument('--g_model', type=str, default='mlp', choices=MODEL_CHOICES,
                   help='G 网络主干类型')
    p.add_argument('--g_input', type=int, default=1, choices=[1],
                   help='G=[x,t,u,u_x,u_t]')
    p.add_argument('--hidden_dim', type=int, default=512,
                   help='隐藏层维度')
    p.add_argument('--f_hidden_dim', type=int, default=None,
                   help='F 网络隐藏层维度，覆盖 hidden_dim')
    p.add_argument('--g_hidden_dim', type=int, default=None,
                   help='G 网络隐藏层维度，覆盖 hidden_dim')
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--transformer_heads', type=int, default=4,
                   help='Transformer 多头注意力头数')

    # 训练
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--early_stop', type=int, default=15)
    p.add_argument('--alpha', type=float, default=1.0, help='L_PDE 权重')
    p.add_argument('--beta', type=float, default=0.1, help='L_period 权重')
    p.add_argument('--pde_freq', type=int, default=1,
                   help='每隔多少个 batch 计算一次 PDE/period loss')
    p.add_argument('--eval_all_leads', action='store_true',
                   help='Evaluate and save RMSE for every configured lead time.')

    # 其他
    p.add_argument('--save_folder', type=str, default='results')
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--seed', type=int, default=42, help='随机种子')
    p.add_argument('--demo', action='store_true')

    args = p.parse_args()
    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model is not None:
        args.f_model = args.model
        args.g_model = args.model

    lt_str = args.lead_times
    if '-' in lt_str and ',' not in lt_str:
        a, b = lt_str.split('-')
        args.lead_times_list = list(range(int(a), int(b) + 1))
    else:
        args.lead_times_list = [int(x) for x in lt_str.split(',')]

    folder = f'mode_{args.mode}_F_{args.f_model}_G_{args.g_model}'
    if args.mode in ['B', 'C']:
        folder += f'_g{args.g_input}'
    args.save_folder = os.path.join(args.save_folder, folder)
    os.makedirs(args.save_folder, exist_ok=True)
    return args
