
import json
import os
import random

import numpy as np
import pandas as pd
import torch

from configs.config import get_args
from dataloader.weather_dataloader import (
    classify_columns,
    create_dataloader,
    generate_demo_data,
    get_feature_cols,
)
from Model.pinn_weather import WeatherPINN


MODE_DESC = {
    'A': 'L_data',
    'B': 'L_data + alpha * L_PDE',
    'C': 'L_data + alpha * L_PDE + beta * L_period',
    'D': 'L_data + beta * L_period',
}


def evaluate_all_leads(model, loaders, scaler_info, lead_times, save_folder):
    rows = []
    lead_metrics = {}
    for lead in sorted(set(int(x) for x in lead_times)):
        key = f'test_{lead}h'
        if key not in loaders:
            continue
        true, pred = model.evaluate(loaders[key])
        metrics = model.compute_metrics(true, pred, scaler_info)
        lead_metrics[str(lead)] = {}
        for name, item in metrics.items():
            lead_metrics[str(lead)][name] = float(item['RMSE'])
            rows.append({
                'lead_hour': lead,
                'target': name,
                'rmse': float(item['RMSE']),
                'unit': item['unit'],
            })

    if not rows:
        return lead_metrics

    df = pd.DataFrame(rows)
    os.makedirs(save_folder, exist_ok=True)
    csv_path = os.path.join(save_folder, 'lead_time_metrics.csv')
    json_path = os.path.join(save_folder, 'lead_time_metrics.json')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(lead_metrics, f, ensure_ascii=False, indent=2)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4.2))
        for target, group in df.groupby('target'):
            group = group.sort_values('lead_hour')
            plt.plot(group['lead_hour'], group['rmse'], marker='o',
                     linewidth=1.8, label=target)
        plt.xlabel('Lead time (hours)')
        plt.ylabel('RMSE')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, 'lead_time_rmse.png'), dpi=200)
        plt.close()
    except Exception as exc:
        print(f"  Lead-time plot skipped: {exc}")

    print("\n  Lead-time RMSE")
    print("  " + "-" * 48)
    for row in rows:
        print(f"  {row['lead_hour']:>3}h | {row['target']:<12} RMSE: "
              f"{row['rmse']:.4f} {row['unit']}")
    print(f"  Saved lead-time metrics: {csv_path}")
    print("LEAD_RMSE_JSON: " + json.dumps(
        lead_metrics, ensure_ascii=False, sort_keys=True))
    return lead_metrics


def main():
    args = get_args()
    target_cols = [c.strip() for c in args.target_cols.split(',')]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 70)
    print(f"  PINN 天气预测 | Mode {args.mode}: {MODE_DESC[args.mode]}")
    print("=" * 70)
    print(f"  设备: {args.device} | Epochs: {args.epochs} | Seed: {args.seed}")
    print(f"  F 模型: {args.f_model} | G 模型: {args.g_model} | g_input={args.g_input}")
    print(f"  目标: {target_cols} | 历史窗口: {args.seq_length}h | 预测跨度: {args.forecast_hours}h")
    if args.mode in ['B', 'C']:
        print(f"  alpha={args.alpha}")
    if args.mode in ['C', 'D']:
        print(f"  beta={args.beta}")
    print(f"  Lead times: {args.lead_times_list}")

    if args.demo:
        df, feature_cols, _ = generate_demo_data()
    else:
        df = pd.read_csv(args.data_path)
        if args.max_samples and len(df) > args.max_samples:
            print(f"  截取最近 {args.max_samples} 条样本（原始 {len(df)}）")
            df = df.iloc[-args.max_samples:].reset_index(drop=True)
        feature_cols, _ = get_feature_cols(df.columns, target_cols)
        surface, pressure, static = classify_columns(df.columns)
        print(f"  列统计: 地表 {len(surface)} | 气压层 {len(pressure)} | 静态 {len(static)}")

    loaders, period_loader, scaler_info, num_feat = create_dataloader(
        df, feature_cols, target_cols, args)

    model = WeatherPINN(args, num_feat, len(target_cols), target_cols)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数量: {n_params:,}")
    print("=" * 70)

    model.run_training(loaders, period_loader, scaler_info)

    if model.best_model:
        model.solution_u.load_state_dict(model.best_model['solution_u'])
    true, pred = model.evaluate(loaders['test_72h'])
    metrics = model.compute_metrics(true, pred, scaler_info)

    print("\n" + "=" * 70)
    print(f"  最终结果 ({args.forecast_hours}h 预测)")
    print("-" * 70)
    for name, metric in metrics.items():
        rmse = metric['RMSE']
        fmt = f"{rmse:.4e}" if rmse < 0.01 else f"{rmse:.4f}"
        print(f"    {name:<12} RMSE: {fmt:>14}  {metric['unit']}")
    print("=" * 70)

    np.save(os.path.join(args.save_folder, 'true.npy'), true)
    np.save(os.path.join(args.save_folder, 'pred.npy'), pred)

    if args.eval_all_leads:
        evaluate_all_leads(
            model=model,
            loaders=loaders,
            scaler_info=scaler_info,
            lead_times=args.lead_times_list,
            save_folder=args.save_folder,
        )


if __name__ == '__main__':
    main()
