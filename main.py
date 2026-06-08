"""
llm_main.py — MoE-LLM ensemble pipeline entry point.

Loads expert models, pre-computes predictions on GPU, then runs
per-station LLM-guided training (reserved set) + evaluation (test set)
with concurrent ThreadPoolExecutor processing.
"""
import os
import sys
import json
import argparse
import random
import concurrent.futures
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch

diffstg_path = '/root/MoELLM/diffusion'
if diffstg_path not in sys.path:
    sys.path.insert(0, diffstg_path)

# Ensure local modules take priority over DiffSTG's utils/
_local_path = os.path.dirname(os.path.abspath(__file__))
if _local_path not in sys.path:
    sys.path.insert(0, _local_path)

from script import dataloader

from expert_loader import ExpertModelLoader
from llm_agent import LLMAgent
from ensemble_predictor import EnsemblePredictor
import importlib.util as _ilu, os as _os
_spec = _ilu.spec_from_file_location("utils", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "utils.py"))
_utils_mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_utils_mod)
calculate_metrics = _utils_mod.calculate_metrics


def set_env(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='LLM Agent for Expert Model Ensemble')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API Key')
    parser.add_argument('--dataset', type=str, default='shanghai')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--enable_cuda', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_workers', type=int, default=20,
                        help='Number of concurrent stations to process (default: 20)')
    parser.add_argument('--station_ids', type=str, default='0',
                        help='Comma-separated base station IDs to test, e.g., "0,10,20,30" (0-4504 for shanghai)')
    parser.add_argument('--enable_llm', type=str, default='true',
                        help='Enable LLM for weight decision (true/false). If false, use rule-based weights.')
    parser.add_argument('--use_proxy', type=str, default='false',
                        help='Use proxy for API connection (true/false)')
    parser.add_argument('--proxy_url', type=str, default='http://127.0.0.1:7890',
                        help='Proxy URL if use_proxy is true')
    parser.add_argument('--timeout', type=float, default=15.0,
                        help='API timeout in seconds (default: 15s)')
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini',
                        help='LLM model name (default: gpt-4o-mini)')
    args = parser.parse_args()

    enable_llm = args.enable_llm.lower() == 'true'
    use_proxy = args.use_proxy.lower() == 'true'

    set_env(args.seed)

    if args.enable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")


    # ── Load data ──
    print("\n" + "="*50)
    print("Loading data...")
    print("="*50)

    dataset_path = f'/root/MoELLM/data/{args.dataset}'
    traffic_df = pd.read_csv(os.path.join(dataset_path, 'traffic.csv'))
    data_col = traffic_df.shape[0]
    n_vertex = traffic_df.shape[1]

    traffic_trend_df    = pd.read_csv(os.path.join(dataset_path, 'traffic_trend.csv'))
    traffic_seasonal_df = pd.read_csv(os.path.join(dataset_path, 'traffic_seasonal.csv'))
    traffic_residual_df = pd.read_csv(os.path.join(dataset_path, 'traffic_residual.csv'))
    # Parse station_ids
    if args.station_ids.lower() == 'all':
        station_ids = list(range(n_vertex))
        print(f"\n*** Testing with ALL {len(station_ids)} stations (0-{n_vertex-1}) ***")
    elif args.station_ids.lower().startswith('random:'):
        try:
            n_random = int(args.station_ids.split(':')[1])
            n_random = min(n_random, n_vertex)
            np.random.seed(args.seed)
            station_ids = sorted(np.random.choice(n_vertex, n_random, replace=False).tolist())
            print(f"\n*** Testing with {len(station_ids)} RANDOM stations (seed={args.seed}) ***")
            print(f"    First 20 stations: {station_ids[:20]}{'...' if len(station_ids) > 20 else ''}")
        except (ValueError, IndexError) as e:
            print(f"Error parsing random station count: {e}")
            station_ids = [0]
    else:
        station_ids = [int(sid.strip()) for sid in args.station_ids.split(',')]
        print(f"\n*** Testing with {len(station_ids)} station(s): {station_ids[:10]}{'...' if len(station_ids) > 10 else ''} ***")

    traffic_with_ts = pd.read_csv(os.path.join(dataset_path, 'traffic_timestamp.csv'))
    timestamps = pd.to_datetime(traffic_with_ts.iloc[:, 0])

    n_his = 12
    n_pred = 1
    window_size = n_his + n_pred  # 13

    len_test     = 8  * window_size + n_his   # 116
    len_val      = 4  * window_size + n_his   # 64
    len_reserved = 16 * window_size + n_his   # 220
    len_train    = data_col - len_reserved - len_val - len_test
    len_train_full = len_train + len_reserved

    train, val, test, reserved = dataloader.load_data(args.dataset, len_train_full, len_train, len_val)

    trend_data    = traffic_trend_df.values
    seasonal_data = traffic_seasonal_df.values
    residual_data = traffic_residual_df.values

    trend_reserved    = trend_data[len_train:len_train_full]
    trend_val         = trend_data[len_train_full:len_train_full+len_val]
    trend_test        = trend_data[len_train_full+len_val:]

    seasonal_reserved = seasonal_data[len_train:len_train_full]
    seasonal_val      = seasonal_data[len_train_full:len_train_full+len_val]
    seasonal_test     = seasonal_data[len_train_full+len_val:]

    residual_reserved = residual_data[len_train:len_train_full]
    residual_val      = residual_data[len_train_full:len_train_full+len_val]
    residual_test     = residual_data[len_train_full+len_val:]

    reserved_start_idx = len_train + n_his
    all_timestamps = timestamps[reserved_start_idx:].reset_index(drop=True)

    # Using FULL multi-station data for prediction

    reserved_np = reserved.values if hasattr(reserved, 'values') else reserved
    val_np      = val.values      if hasattr(val,      'values') else val
    test_np     = test.values     if hasattr(test,     'values') else test

    # Sliding-window transform
    x_reserved, y_reserved = dataloader.data_transform(reserved_np,    n_his, n_pred, device)
    x_val,      y_val      = dataloader.data_transform(val_np,         n_his, n_pred, device)
    x_test,     y_test     = dataloader.data_transform(test_np,        n_his, n_pred, device)

    x_trend_reserved, _ = dataloader.data_transform(trend_reserved,    n_his, n_pred, device)
    x_trend_val,      _ = dataloader.data_transform(trend_val,         n_his, n_pred, device)
    x_trend_test,     _ = dataloader.data_transform(trend_test,        n_his, n_pred, device)

    x_seasonal_reserved, _ = dataloader.data_transform(seasonal_reserved, n_his, n_pred, device)
    x_seasonal_val,      _ = dataloader.data_transform(seasonal_val,      n_his, n_pred, device)
    x_seasonal_test,     _ = dataloader.data_transform(seasonal_test,     n_his, n_pred, device)

    x_residual_reserved, _ = dataloader.data_transform(residual_reserved, n_his, n_pred, device)
    x_residual_val,      _ = dataloader.data_transform(residual_val,      n_his, n_pred, device)
    x_residual_test,     _ = dataloader.data_transform(residual_test,     n_his, n_pred, device)

    window_step = n_his + n_pred

    sampling_indices = {
        'reserved': torch.arange(0, x_reserved.shape[0]).cpu().numpy(),
        'val':      torch.arange(0, x_val.shape[0]).cpu().numpy(),
        'test':     torch.arange(0, x_test.shape[0]).cpu().numpy()
    }

    # ── Load expert models ──
    loader = ExpertModelLoader(device, args.dataset)
    expert_models = loader.load_all_models()

    ensemble_predictor = EnsemblePredictor(expert_models, device, args.dataset)

    # Load POI data
    try:
        poi_file = f'/root/MoELLM/data/{args.dataset}/{args.dataset}_poi.json'
        with open(poi_file, 'r', encoding='utf-8') as f:
            poi_data_full = json.load(f)
    except Exception:
        poi_data_full = {}

    poi_idx_map = {v['idx_final']: v for v in poi_data_full.values() if 'idx_final' in v}

    def get_station_poi_info(station_id):
        value = poi_idx_map.get(station_id)
        if value:
            poi_stats = value.get('poi_category_stats', {})
            total_poi = sum(poi_stats.values())
            if total_poi > 0:
                top = sorted(poi_stats.items(), key=lambda x: x[1], reverse=True)[:5]
                cat_str = ", ".join([
                    f"{c} {n/total_poi*100:.0f}%"
                    for c, n in top
                ])
                return f"Nearby POIs (total {total_poi}): {cat_str}"
        return f"Station {station_id}: Mixed urban area"

    def get_station_prior(station_id):
        value = poi_idx_map.get(station_id)
        if not value:
            return 0.9, 0.1
        poi_stats = value.get('poi_category_stats', {})
        total = sum(poi_stats.values())
        if total == 0:
            return 0.9, 0.1
        office_ratio   = poi_stats.get('Offices & Enterprises', 0) / total
        shopping_ratio = poi_stats.get('Shopping & Retail', 0) / total
        transit_ratio  = poi_stats.get('Transportation', 0) / total
        resi_ratio     = poi_stats.get('Residential & Business', 0) / total
        if transit_ratio > 0.15:
            return 0.8, 0.2
        elif office_ratio + shopping_ratio > 0.35:
            return 0.85, 0.15
        elif resi_ratio > 0.2:
            return 0.9, 0.1
        else:
            return 0.9, 0.1

    reserved_timestamps = all_timestamps.iloc[sampling_indices['reserved']].reset_index(drop=True)
    val_timestamps      = all_timestamps.iloc[sampling_indices['val']].reset_index(drop=True)
    test_timestamps     = all_timestamps.iloc[sampling_indices['test']].reset_index(drop=True)

    # Pre-computing expert predictions for all samples 

    expert_names_list = ['spatial_geo', 'spatial_poi', 'spatial_similarity',
                         'linear_trend', 'lstm_seasonal', 'fourier_seasonal', 'residual']

    def precompute_split(x_data, x_trend_data, x_seasonal_data, x_residual_data,
                         y_data, timestamps_series, split_name):
        result = {}
        n = x_data.shape[0]
        for i in range(n):
            x_input = x_data[i:i+1].squeeze(1)
            all_preds_i = ensemble_predictor.get_all_predictions(
                x_input,
                x_trend=x_trend_data[i:i+1].squeeze(1),
                x_seasonal=x_seasonal_data[i:i+1].squeeze(1),
                x_residual=x_residual_data[i:i+1].squeeze(1),
                station_id=None
            )
            result[i] = {
                'expert_preds': {
                    name: all_preds_i[name].cpu().numpy()
                    for name in expert_names_list
                },
                'y_true':    y_data[i].cpu().numpy(),
                'timestamp': str(timestamps_series.iloc[i]),
                'hist_seq':  x_input[0].cpu().numpy()
            }
        return result

    precomp = {}
    precomp['reserved'] = precompute_split(
        x_reserved, x_trend_reserved, x_seasonal_reserved, x_residual_reserved,
        y_reserved, reserved_timestamps, 'reserved')
    precomp['val'] = precompute_split(
        x_val, x_trend_val, x_seasonal_val, x_residual_val,
        y_val, val_timestamps, 'val')
    precomp['test'] = precompute_split(
        x_test, x_trend_test, x_seasonal_test, x_residual_test,
        y_test, test_timestamps, 'test')

    # Initializing reliance scores (POI-based priors) 

    scenes = ["morning_rush", "evening_rush", "late_night", "regular"]
    init_reliance_map = {}
    for sid in station_ids:
        sw, dw = get_station_prior(sid)
        rs = {}
        for spatial_n in ['spatial_geo', 'spatial_poi', 'spatial_similarity']:
            rs[spatial_n] = {s: [sw, sw, sw] for s in scenes}
        rs['component_group'] = {s: [dw, dw, dw] for s in scenes}
        rs['lstm_seasonal']   = {s: [0.5, 0.5, 0.5] for s in scenes}
        rs['fourier_seasonal'] = {s: [0.5, 0.5, 0.5] for s in scenes}
        init_reliance_map[sid] = rs

    def process_station(station_id):
        """Train on reserved set, evaluate on test set for one station."""
        agent = LLMAgent(
            api_key=args.api_key,
            model_name=args.model_name,
            enable_llm=enable_llm,
            use_proxy=use_proxy,
            proxy_url=args.proxy_url if use_proxy else None,
            timeout=args.timeout
        )

        station_poi_info = get_station_poi_info(station_id)

        init_rs = init_reliance_map[station_id]
        agent.reliance_scores = {k: {sk: list(sv) for sk, sv in v.items()} for k, v in init_rs.items()}

        max_epochs = 5
        patience   = 3
        best_val_mse          = float('inf')
        best_reliance_scores  = {k: {sk: list(sv) for sk, sv in v.items()} for k, v in agent.reliance_scores.items()}
        best_expert_val_mae   = {name: 1.0 for name in agent.expert_names}
        best_experience_summary = ""
        patience_counter = 0
        num_reserved = x_reserved.shape[0]
        num_val      = x_val.shape[0]

        for epoch in range(max_epochs):
            reserved_records = []
            for i in range(num_reserved):
                d = precomp['reserved'][i]
                expert_pred_values = {
                    name: (d['expert_preds'][name][0, station_id]
                           if d['expert_preds'][name].shape[-1] > 1
                           else d['expert_preds'][name].flatten()[0])
                    for name in agent.expert_names
                }
                sample = {
                    'timestamp':           d['timestamp'],
                    'poi_info':            station_poi_info,
                    'historical_sequence': d['hist_seq'][:, station_id].tolist(),
                    'expert_predictions':  expert_pred_values
                }
                agent.predict_batch_with_llm(station_id, [sample])
                true_val = float(d['y_true'][station_id])
                scene = agent.extract_scene_features(d['timestamp'], station_poi_info,
                                                     sample['historical_sequence'])
                agent.update_reliance_scores_with_llm(
                    station_id=station_id, scene=scene,
                    expert_predictions=expert_pred_values, true_value=true_val,
                    timestamp=d['timestamp'], poi_info=station_poi_info
                )
                errors = {name: abs(expert_pred_values[name] - true_val) for name in agent.expert_names}
                best_expert = min(errors, key=errors.get)
                reserved_records.append({
                    'timestamp':          d['timestamp'],
                    'scene_key':          scene['scene_key'],
                    'expert_predictions': expert_pred_values,
                    'true_value':         true_val,
                    'best_expert':        best_expert,
                    'errors':             errors
                })

            agent.experience_summary = agent.build_experience_summary(reserved_records)

            val_preds, val_trues = [], []
            expert_val_preds = {name: [] for name in agent.expert_names}
            for j in range(num_val):
                dv = precomp['val'][j]
                expert_pred_val = {
                    name: (dv['expert_preds'][name][0, station_id]
                           if dv['expert_preds'][name].shape[-1] > 1
                           else dv['expert_preds'][name].flatten()[0])
                    for name in agent.expert_names
                }
                val_sample = {
                    'timestamp':           dv['timestamp'],
                    'poi_info':            station_poi_info,
                    'historical_sequence': dv['hist_seq'][:, station_id].tolist(),
                    'expert_predictions':  expert_pred_val
                }
                pred = agent.predict_batch_with_llm(station_id, [val_sample])[0]
                val_preds.append(pred)
                val_trues.append(float(dv['y_true'][station_id]))
                for name in agent.expert_names:
                    expert_val_preds[name].append(expert_pred_val[name])

            val_mse = float(np.mean((np.array(val_trues) - np.array(val_preds)) ** 2))
            val_trues_arr = np.array(val_trues)
            expert_val_mae = {
                name: float(np.mean(np.abs(np.array(expert_val_preds[name]) - val_trues_arr)))
                for name in agent.expert_names
            }

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_reliance_scores = {k: {sk: list(sv) for sk, sv in v.items()}
                                        for k, v in agent.reliance_scores.items()}
                best_expert_val_mae     = expert_val_mae.copy()
                best_experience_summary = agent.experience_summary
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        agent.reliance_scores    = best_reliance_scores
        agent.experience_summary = best_experience_summary

        agent.total_prompt_tokens = 0
        agent.total_completion_tokens = 0
        num_test = x_test.shape[0]
        test_preds, test_trues = [], []
        for i in range(num_test):
            d = precomp['test'][i]
            expert_pred_values = {
                name: (d['expert_preds'][name][0, station_id]
                       if d['expert_preds'][name].shape[-1] > 1
                       else d['expert_preds'][name].flatten()[0])
                for name in agent.expert_names
            }
            test_sample = {
                'timestamp':           d['timestamp'],
                'poi_info':            station_poi_info,
                'historical_sequence': d['hist_seq'][:, station_id].tolist(),
                'expert_predictions':  expert_pred_values,
                'expert_val_mae':      best_expert_val_mae
            }
            pred = agent.predict_batch_with_llm(station_id, [test_sample])[0]
            test_preds.append(pred)
            test_trues.append(float(d['y_true'][station_id]))

        station_preds_np = np.array(test_preds)
        station_trues_np = np.array(test_trues)
        station_mae  = float(np.mean(np.abs(station_trues_np - station_preds_np)))
        station_rmse = float(np.sqrt(np.mean((station_trues_np - station_preds_np) ** 2)))
        ss_res_s = float(np.sum((station_trues_np - station_preds_np) ** 2))
        ss_tot_s = float(np.sum((station_trues_np - np.mean(station_trues_np)) ** 2))
        station_r2 = float(1 - ss_res_s / ss_tot_s) if ss_tot_s > 0 else 0.0
        print(f"  Station {station_id}: MAE={station_mae:.4f}, RMSE={station_rmse:.4f}, R2={station_r2:.4f}")

        return station_preds_np, station_trues_np

    # ── Concurrent per-station processing ──
    results = [None] * len(station_ids)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_idx = {executor.submit(process_station, sid): idx
                         for idx, sid in enumerate(station_ids)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"  Station {station_ids[idx]} failed: {e}")
                results[idx] = None

    all_preds = [r[0] for r in results if r is not None]
    all_trues = [r[1] for r in results if r is not None]

    all_preds_arr = np.concatenate(all_preds, axis=0)
    all_trues_arr = np.concatenate(all_trues, axis=0)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    metrics = calculate_metrics(all_trues_arr, all_preds_arr)
    print(f"MAE:   {metrics['mae']:.4f}")
    print(f"RMSE:  {metrics['rmse']:.4f}")
    print(f"R2:    {metrics['r2']:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()
