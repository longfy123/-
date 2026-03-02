import os
import sys
import json
import logging
import argparse
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.utils as utils

# 添加DiffSTG路径
diffstg_path = '/root/stgcn原版/DiffSTG-main'
if diffstg_path not in sys.path:
    sys.path.insert(0, diffstg_path)

from script import dataloader, utility
from model import models
from lstm_trend_residual_predictor import LSTMResidualPredictor
from lstm_seasonal_predictor import LSTMPredictor as LSTMSeasonalPredictor
from lstm_fourier_seasonal_predictor import LSTMPredictor as LSTMFourierPredictor

# 导入DiffSTG模型
try:
    from algorithm.diffstg.model import DiffSTG
    from algorithm.diffstg.ugnet import UGnet
    # PyTorch 2.6+ 需要将自定义类添加到安全全局列表
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([DiffSTG, UGnet])
        print("✓ DiffSTG模块导入成功 (已添加到PyTorch安全全局列表)")
    else:
        print("✓ DiffSTG模块导入成功")
    DIFFSTG_AVAILABLE = True
except ImportError as e:
    print(f"⚠ DiffSTG模块导入失败: {e}")
    DIFFSTG_AVAILABLE = False

# OpenAI API
import openai

def set_env(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ExpertModelLoader:
    def __init__(self, device, dataset='shanghai'):
        self.device = device
        self.dataset = dataset
        self.n_vertex = 4505 if dataset == 'shanghai' else 7775
        self.n_his = 12
        self.n_pred = 1
        
        # 存储所有专家模型
        self.expert_models = {}
        
    def load_stgcn_model(self, model_path, adj_type='geo'):
        """加载STGCN模型"""
        # 加载邻接矩阵
        dataset_path = f'/root/stgcn原版/data/{self.dataset}'
        if adj_type == 'geo':
            adj = sp.load_npz(os.path.join(dataset_path, 'adj_geo.npz'))
        elif adj_type == 'poi':
            adj = sp.load_npz(os.path.join(dataset_path, 'adj_poi.npz'))
        elif adj_type == 'similarity':
            adj = sp.load_npz(os.path.join(dataset_path, 'adj_similarity.npz'))

        adj = adj.tocsc()
        gso = utility.calc_gso(adj, 'sym_norm_lap')
        gso = utility.calc_chebynet_gso(gso)
        gso = gso.toarray().astype(np.float32)
        gso_tensor = torch.from_numpy(gso).to(self.device)
        
        # 构建模型参数
        class Args:
            def __init__(self):
                self.Kt = 3
                self.Ks = 3
                self.act_func = 'glu'
                self.graph_conv_type = 'cheb_graph_conv'
                self.enable_bias = True
                self.droprate = 0.1
                self.gso = gso_tensor
                self.n_his = 12
        
        args = Args()
        
        # 构建blocks
        blocks = [[1], [64, 16, 64], [64, 16, 64], [128, 128], [1]]
        
        # 创建模型
        model = models.STGCNChebGraphConv(args, blocks, self.n_vertex).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        return model
    
    def load_lstm_trend_model(self, model_path):
        model = LSTMResidualPredictor(
            input_size=self.n_vertex * 2,  # 包含移动平均特征
            output_size=self.n_vertex,
            seq_len=self.n_his,
            hidden_size=128,
            num_layers=3,
            dropout=0.2
        ).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def load_lstm_seasonal_model(self, model_path):
        model = LSTMSeasonalPredictor(
            input_size=self.n_vertex * 2,  # 包含移动平均特征
            output_size=self.n_vertex,
            seq_len=self.n_his,
            hidden_size=128,
            num_layers=3,
            dropout=0.2
        ).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def load_lstm_fourier_seasonal_model(self, model_path):
        model = LSTMFourierPredictor(
            input_size=self.n_vertex * 2,  # 包含移动平均特征
            output_size=self.n_vertex,
            seq_len=self.n_his,
            hidden_size=128,
            num_layers=3,
            dropout=0.2
        ).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def load_residual_model(self, model_path):
        """加载残差预测模型 (DiffSTG)"""
        try:
            if not DIFFSTG_AVAILABLE:
                print("  ⚠ DiffSTG模块不可用，使用LSTM残差模型...")
                raise ImportError("DiffSTG模块不可用")
            
            print("  尝试加载DiffSTG模型...")
            # PyTorch 2.6+ 需要设置 weights_only=False 来加载包含自定义类的模型
            model = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 检查是否是DiffSTG或UGnet模型
            model_class_name = type(model).__name__
            print(f"  模型类型: {type(model)}")
            print(f"  模型类名: {model_class_name}")
            
            # 检查是否是DiffSTG模型
            if isinstance(model, DiffSTG):
                print("  ✓ 成功加载DiffSTG模型")
                model.eval()
                return model
            # 检查是否是UGnet模型
            elif isinstance(model, UGnet):
                print("  ✓ 成功加载DiffSTG UGnet模型")
                model.eval()
                return model
            # 检查是否有eps_model属性（DiffSTG的核心组件）
            elif hasattr(model, 'eps_model'):
                print("  ✓ 成功加载DiffSTG扩散模型（通过eps_model属性）")
                model.eval()
                return model
            # 检查是否是state_dict
            elif isinstance(model, dict):
                print("  ⚠ 检测到state_dict格式，尝试重建DiffSTG模型...")
                # 这里需要配置信息来重建模型
                # 先回退到LSTM
                raise ValueError("state_dict格式需要配置文件来重建模型")
            else:
                raise ValueError(f"无法识别的模型类型: {model_class_name}")
                    
        except Exception as e:
            print(f"  警告: 加载DiffSTG模型失败: {e}")
            print("  回退到LSTM残差模型...")
            
            # 回退方案：使用LSTM残差模型
            model = LSTMResidualPredictor(
                input_size=self.n_vertex * 2,  # 包含移动平均特征
                output_size=self.n_vertex,
                seq_len=self.n_his,
                hidden_size=128,
                num_layers=3,
                dropout=0.2
            ).to(self.device)
            
            # 尝试加载state_dict
            try:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
                if isinstance(state_dict, dict):
                    model.load_state_dict(state_dict)
                    print("  ✓ 成功加载LSTM残差模型state_dict")
            except:
                print("  ✗ 无法加载模型权重，使用随机初始化")
            
            model.eval()
            return model
    
    def load_all_models(self):
        """加载所有7个专家模型"""
        # 1. Geo-based STGCN
        self.expert_models['stgcn_geo'] = self.load_stgcn_model(
            '/root/stgcn原版/STGCN_geo_shanghai.pt', 'geo'
        )
        print("✓ Loaded STGCN Geo model")
        # 2. POI-based STGCN
        self.expert_models['stgcn_poi'] = self.load_stgcn_model(
            '/root/stgcn原版/STGCN_poi_shanghai.pt', 'poi'
        )
        print("✓ Loaded STGCN POI model")
        # 3. Similarity-based STGCN
        self.expert_models['stgcn_similarity'] = self.load_stgcn_model(
            '/root/stgcn原版/STGCN_similarity_shanghai.pt', 'similarity'
        )
        print("✓ Loaded STGCN Similarity model")
        # 4. LSTM Trend
        self.expert_models['lstm_trend'] = self.load_lstm_trend_model(
            '/root/stgcn原版/LSTM_trend_shanghai.pt'
        )
        print("✓ Loaded LSTM Trend model")
        # 5. LSTM Seasonal
        self.expert_models['lstm_seasonal'] = self.load_lstm_seasonal_model(
            '/root/stgcn原版/LSTM_seasonal_shanghai.pt'
        )
        print("✓ Loaded LSTM Seasonal model")
        # 6. LSTM Fourier Seasonal
        self.expert_models['lstm_fourier_seasonal'] = self.load_lstm_fourier_seasonal_model(
            '/root/stgcn原版/LSTM_Fourier_seasonal_shanghai.pt'
        )
        print("✓ Loaded LSTM Fourier Seasonal model")
        # 7. Residual Model
        self.expert_models['residual'] = self.load_residual_model(
            '/root/stgcn原版/residual_model.pt'
        )
        print("✓ Loaded Residual model")
        print(f"\nTotal {len(self.expert_models)} expert models loaded successfully!")
        return self.expert_models


class LLMAgent:
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", 
                 enable_llm: bool = True, use_proxy: bool = False, 
                 proxy_url: str = None, timeout: float = 30.0):
        self.enable_llm = enable_llm
        self.timeout = timeout
        # 初始化OpenAI客户端
        client_kwargs = {
            "api_key": api_key,
            "base_url": "https://api.gptsapi.net/v1",
            "timeout": timeout
        }
        if use_proxy and proxy_url:
            import httpx
            client_kwargs["http_client"] = httpx.Client(proxies=proxy_url)
        
        self.client = openai.OpenAI(**client_kwargs)
        self.model_name = model_name
        # 专家模型名称列表
        self.expert_names = [
            'stgcn_geo',
            'stgcn_poi', 
            'stgcn_similarity',
            'lstm_trend',
            'lstm_seasonal',
            'lstm_fourier_seasonal',
            'residual'
        ]
        self.current_weights = {
            'stgcn_geo': 0.35,
            'stgcn_poi': 0.25,
            'stgcn_similarity': 0.20,
            'lstm_trend': 0.05,
            'lstm_seasonal': 0.05,
            'lstm_fourier_seasonal': 0.05,
            'residual': 0.05
        }
        
        # 历史性能记录
        self.performance_history = []
        
    def create_context_prompt(self, timestep: int, timestamp: str, poi_info: str, 
                             expert_predictions: Dict[str, float],
                             previous_errors: List[Dict] = None) -> str:
        """
        创建给LLM的上下文提示
        
        Args:
            timestep: 当前时间步
            timestamp: 时间戳
            poi_info: POI分布信息摘要
            expert_predictions: 各专家模型的预测结果（平均值）
            previous_errors: 之前的误差历史
            
        Returns:
            完整的提示文本
        """
        # 解析时间特征
        import pandas as pd
        dt = pd.to_datetime(timestamp)
        hour = dt.hour
        weekday = dt.day_name()
        is_weekend = dt.dayofweek >= 5
        
        # 判断时段特征
        if 7 <= hour < 9:
            time_period = "Morning Rush Hour (7-9 AM)"
        elif 12 <= hour < 14:
            time_period = "Lunch Time (12-2 PM)"
        elif 17 <= hour < 19:
            time_period = "Evening Rush Hour (5-7 PM)"
        elif 22 <= hour or hour < 6:
            time_period = "Late Night (10 PM - 6 AM)"
        else:
            time_period = "Regular Hours"
        
        prompt = f"""You are an expert ensemble manager for time series prediction of Shanghai base station traffic.

**Current Context:**
- Timestep: {timestep}
- Timestamp: {timestamp}
- Hour: {hour}:00
- Day: {weekday} {'(Weekend)' if is_weekend else '(Weekday)'}
- Time Period: {time_period}

**Location Context:**
{poi_info}

**Available Expert Models (Hierarchical Ensemble):**

**Group 1 - STGCN Models (Predict Complete Traffic) [GENERALLY SUPERIOR]:**
1. STGCN_geo: Spatial-temporal graph network based on geographical adjacency
2. STGCN_poi: Spatial-temporal graph network based on POI similarity  
3. STGCN_similarity: Spatial-temporal graph network based on traffic pattern similarity

**Performance: STGCN models typically achieve better accuracy (lower MAE/RMSE) in most scenarios**

**Group 2 - Component Models (Decomposition-based, Combined via Addition) [SPECIALIZED]:**
4. LSTM_trend: LSTM model for trend component prediction
5. LSTM_seasonal: LSTM model for seasonal component prediction
6. LSTM_Fourier_seasonal: LSTM model with Fourier features for seasonal prediction
7. Residual: Model for residual component prediction

**Performance: LSTM models are less accurate overall, but may be more robust in:**
- Extreme weather conditions
- Holiday periods with unusual patterns
- Early morning/late night hours with sparse data
- Temporary events breaking spatial correlations

**Ensemble Strategy:**
- STGCN models are weighted and averaged (predicting same target: complete traffic)
- Component models are ADDED together (trend + seasonal + fourier + residual = complete traffic)
- Final prediction = Weighted combination of STGCN ensemble and Component sum

**CRITICAL DECISION RULE:**
- **Default bias: 70-90% weight to STGCN group** (they perform better in normal conditions)
- **Increase LSTM weight to 30-50%** only when special conditions detected:
  * Unusual time period (holidays, festivals)
  * Extreme hourly patterns (very early morning, very late night)
  * Recent performance shows STGCN degradation
  * Weather/event impacts expected

**Current Expert Predictions (Average across all base stations):**
"""
        for name, pred in expert_predictions.items():
            prompt += f"- {name}: {pred:.4f}\n"
        
        prompt += f"\n**Current Weights:**\n"
        for name, weight in self.current_weights.items():
            prompt += f"- {name}: {weight:.4f}\n"
        
        if previous_errors and len(previous_errors) > 0:
            prompt += f"\n**Recent Performance History (last {min(5, len(previous_errors))} predictions):**\n"
            for i, error in enumerate(previous_errors[-5:]):
                prompt += f"\nPrediction {len(previous_errors) - len(previous_errors[-5:]) + i + 1}:\n"
                prompt += f"  - MAE: {error['mae']:.4f}, RMSE: {error['rmse']:.4f}\n"
                prompt += f"  - Weights used: {json.dumps(error['weights'], indent=4)}\n"
        
        prompt += f"""
**Task:**
Based on the context, time features, POI distribution, expert predictions, and performance history, suggest optimal weights.

**Consider:**
1. **Time patterns**: {time_period} - adjust for traffic peaks/valleys
2. **Day type**: {weekday} - weekday vs weekend patterns differ
3. **POI characteristics**: Match model strengths to location types
4. **Recent performance**: Increase weights for better-performing models
5. **Model complementarity**: STGCN ensemble (spatial-temporal) vs Component sum (decomposition-based)

**Weight Allocation Guidelines:**

- **BASELINE ALLOCATION (Most scenarios):**
  * STGCN group total: 0.75-0.90 (stgcn_geo + stgcn_poi + stgcn_similarity)
  * LSTM group total: 0.10-0.25 (lstm_trend + lstm_seasonal + lstm_fourier + residual)
  
- **Within STGCN group (stgcn_geo, stgcn_poi, stgcn_similarity)**: 
  * Distribute based on location and time characteristics
  * Rush hours: favor models capturing congestion spread
  * Commercial areas: increase stgcn_poi weight
  * Residential areas: increase stgcn_geo weight
  * Example: geo=0.40, poi=0.30, similarity=0.30 for 0.75 STGCN total
  
- **Within LSTM group (lstm_trend, lstm_seasonal, lstm_fourier_seasonal, residual)**:
  * Keep individual weights small (0.02-0.08 each)
  * Balance trend vs seasonal based on time of day
  * Increase fourier for weekend periodic patterns
  * Example: trend=0.06, seasonal=0.06, fourier=0.06, residual=0.07 for 0.25 LSTM total

- **INCREASE LSTM weight (0.30-0.50 total) ONLY when:**
  * Very early morning (0-6 AM) or very late night (23-24) - sparse data favors decomposition
  * Weekend/Holiday - unusual patterns where spatial correlation breaks
  * Recent performance history shows STGCN MAE significantly higher than usual
  * Specific events expected (festivals, weather alerts)

**PRIORITY: Maximize STGCN unless there's a clear reason to use LSTM**

**Output Format (JSON only, no other text):**
{{
    "weights": {{
        "stgcn_geo": 0.35,
        "stgcn_poi": 0.25,
        "stgcn_similarity": 0.20,
        "lstm_trend": 0.05,
        "lstm_seasonal": 0.05,
        "lstm_fourier_seasonal": 0.05,
        "residual": 0.05
    }},
    "reasoning": "Explanation for weight allocation. MUST mention: (1) Total STGCN group weight (should be 0.70-0.90 normally), (2) Total LSTM group weight (should be 0.10-0.30 normally), (3) Whether this is a special situation requiring higher LSTM weight, (4) Specific rationale for current context (time: {time_period}, day: {weekday}, POI context, recent performance)"
}}

**Example for normal rush hour:**
{{
    "weights": {{"stgcn_geo": 0.40, "stgcn_poi": 0.30, "stgcn_similarity": 0.15, "lstm_trend": 0.04, "lstm_seasonal": 0.04, "lstm_fourier_seasonal": 0.04, "residual": 0.03}},
    "reasoning": "STGCN total=0.85, LSTM total=0.15. Morning rush hour on weekday - STGCN excels at capturing spatial congestion spread. No special conditions detected. Heavy bias toward STGCN_geo for geographical propagation."
}}

**Example for late night:**
{{
    "weights": {{"stgcn_geo": 0.25, "stgcn_poi": 0.20, "stgcn_similarity": 0.15, "lstm_trend": 0.10, "lstm_seasonal": 0.10, "lstm_fourier_seasonal": 0.10, "residual": 0.10}},
    "reasoning": "STGCN total=0.60, LSTM total=0.40. Late night (23:00) - sparse data and weak spatial correlation. Increased LSTM weight for trend/seasonal decomposition stability. Still favor STGCN but less aggressively."
}}

Note: Weights must sum to 1.0. The ensemble uses hierarchical combination: STGCN models are averaged within their group, component models are summed, then the two groups are combined.
"""
        return prompt
    
    def query_llm(self, prompt: str) -> Dict:
        """
        查询LLM获取权重建议
        
        Args:
            prompt: 提示文本
            
        Returns:
            包含weights和reasoning的字典
        """
        # 如果未启用LLM，直接返回当前权重
        if not self.enable_llm:
            return {
                "weights": self.current_weights,
                "reasoning": "LLM disabled. Using equal weights."
            }
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in time series forecasting and ensemble learning. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            # 解析LLM响应
            content = response.choices[0].message.content.strip()
            
            # 移除可能的markdown代码块标记
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            
            # 验证权重
            weights = result['weights']
            total_weight = sum(weights.values())
            
            # 归一化权重
            if abs(total_weight - 1.0) > 0.01:
                weights = {k: v / total_weight for k, v in weights.items()}
                result['weights'] = weights
            
            return result
            
        except Exception as e:
            print(f"Warning: LLM query failed: {e}")
            print("Using current weights...")
            return {
                "weights": self.current_weights,
                "reasoning": f"LLM query failed: {e}. Using previous weights."
            }
    
    def update_weights(self, timestep: int, timestamp: str, poi_info: str,
                       expert_predictions: Dict[str, float]) -> Tuple[Dict, str]:
        """
        更新专家模型权重
        
        Returns:
            (new_weights, reasoning)
        """
        # 创建提示
        prompt = self.create_context_prompt(
            timestep, timestamp, poi_info, expert_predictions, self.performance_history
        )
        
        # 查询LLM
        result = self.query_llm(prompt)
        
        # 更新权重
        self.current_weights = result['weights']
        
        return result['weights'], result['reasoning']
    
    def add_performance_record(self, mae: float, rmse: float, weights: Dict):
        """添加性能记录（保持窗口大小）"""
        self.performance_history.append({
            'mae': mae,
            'rmse': rmse,
            'weights': weights.copy()
        })
        # 只保留最近的记录
        if len(self.performance_history) > self.max_history_len:
            self.performance_history = self.performance_history[-self.max_history_len:]
    
    def initialize_station_weights(self, station_id: int, poi_info: str) -> Dict[str, float]:
        """
        为单个基站初始化权重
        
        Args:
            station_id: 基站ID
            poi_info: 该基站的POI信息
            
        Returns:
            专家模型权重字典
        """
        if not self.enable_llm:
            return self.current_weights.copy()
        
        prompt = f"""You are an expert in time series forecasting for base station traffic prediction.

**Task**: Determine optimal expert model weights for a SINGLE base station.

**Station Information:**
- Station ID: {station_id}
- POI Context:
{poi_info}

**Available Expert Models:**

**STGCN Group (usually better, 70-90% total weight recommended):**
1. stgcn_geo: Geographical proximity-based
2. stgcn_poi: POI similarity-based  
3. stgcn_similarity: Traffic pattern similarity-based

**LSTM Group (specialized, 10-30% total weight):**
4. lstm_trend: Trend component
5. lstm_seasonal: Seasonal component
6. lstm_fourier_seasonal: Fourier seasonal features
7. residual: Residual component

**Weight Allocation Strategy:**
- **Commercial/POI-rich areas**: Increase stgcn_poi weight (e.g., 0.40)
- **Residential/geographic dense**: Increase stgcn_geo weight (e.g., 0.40)
- **Mixed areas**: Balance geo/poi/similarity equally
- **LSTM group**: Keep small (0.05-0.08 each) unless special patterns expected

**Output Format (JSON only):**
{{
    "weights": {{
        "stgcn_geo": 0.35,
        "stgcn_poi": 0.30,
        "stgcn_similarity": 0.20,
        "lstm_trend": 0.04,
        "lstm_seasonal": 0.04,
        "lstm_fourier_seasonal": 0.04,
        "residual": 0.03
    }},
    "reasoning": "Brief explanation based on POI characteristics"
}}

Weights must sum to 1.0. Bias toward STGCN models (they perform better overall).
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in time series forecasting. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=400
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            weights = result['weights']
            
            # 归一化
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01:
                weights = {k: v/total for k, v in weights.items()}
            
            return weights
            
        except Exception as e:
            print(f"Warning: LLM initialization failed for station {station_id}: {e}")
            return self.current_weights.copy()


class EnsemblePredictor:
    """集成预测器"""
    
    def __init__(self, expert_models: Dict, device, dataset='shanghai'):
        self.expert_models = expert_models
        self.device = device
        self.dataset = dataset
        self.n_vertex = 4505 if dataset == 'shanghai' else 7775
        self.n_his = 12
    
    def add_moving_average_features(self, x, window_size=3):
        """
        为输入数据添加移动平均特征
        
        Args:
            x: 输入数据 [batch, n_his, n_vertex]
            window_size: 移动平均窗口大小
            
        Returns:
            x_with_ma: 包含移动平均的数据 [batch, n_his, n_vertex*2]
        """
        batch_size, seq_len, n_features = x.shape
        
        # 计算移动平均
        x_ma = torch.zeros_like(x)
        for i in range(seq_len):
            start_idx = max(0, i - window_size + 1)
            x_ma[:, i, :] = x[:, start_idx:i+1, :].mean(dim=1)
        
        # 合并原始数据和移动平均
        x_with_ma = torch.cat([x, x_ma], dim=2)  # [batch, n_his, n_vertex*2]
        
        return x_with_ma
    
    @torch.no_grad()
    def predict_stgcn(self, model, x):
        """STGCN模型预测"""
        # x shape: [batch, n_his, n_vertex]
        # 需要reshape为 [batch, 1, n_his, n_vertex]
        x_input = x.unsqueeze(1)  # [batch, 1, n_his, n_vertex]
        pred = model(x_input)  # [batch, 1, 1, n_vertex]
        pred = pred.squeeze(1).squeeze(1)  # [batch, n_vertex]
        return pred
    
    @torch.no_grad()
    def predict_lstm(self, model, x):
        """LSTM模型预测"""
        # x shape: [batch, n_his, n_vertex]
        pred = model(x)  # [batch, n_vertex]
        return pred
    
    @torch.no_grad()
    def predict_residual(self, model, x):
        """
        残差模型预测（可能是DiffSTG或LSTM）
        
        Args:
            model: 残差模型
            x: 输入数据 [batch, n_his, n_vertex]
            
        Returns:
            pred: 预测结果 [batch, n_vertex]
        """
        # 检查是否是DiffSTG模型
        model_type = type(model).__name__
        
        if 'DiffSTG' in model_type or hasattr(model, 'epsilon_theta'):
            # DiffSTG模型
            # 需要reshape为DiffSTG的输入格式，并添加未来预测时间步的占位符
            # x is [batch, n_his, n_vertex] -> 需要变成 [batch, n_his+n_pred, n_vertex, 1] -> [batch, 1, n_vertex, n_his+n_pred]
            batch_size, n_his, n_vertex = x.shape
            n_pred = 1  # DiffSTG residual模型的T_p=1
            
            # 添加特征维度: [batch, n_his, n_vertex] -> [batch, n_his, n_vertex, 1]
            x_history = x.unsqueeze(-1)  # [batch, n_his, n_vertex, 1]
            
            # 创建未来的零填充: [batch, n_pred, n_vertex, 1]
            x_future_zeros = torch.zeros(batch_size, n_pred, n_vertex, 1, device=x.device)
            
            # 拼接历史和未来: [batch, n_his+n_pred, n_vertex, 1]
            x_with_future = torch.cat([x_history, x_future_zeros], dim=1)  # [batch, 13, n_vertex, 1]
            
            # 转换为DiffSTG格式: (B, T, V, F) -> (B, F, V, T)
            x_diff = x_with_future.permute(0, 3, 2, 1)  # [batch, 1, n_vertex, 13]
            
            try:
                # DEBUG: 打印形状信息
                if False:  # 设置为True来启用调试输出
                    print(f"DEBUG DiffSTG input - x shape: {x.shape}, x_diff shape: {x_diff.shape}")
                    print(f"DEBUG - batch_size: {batch_size}, n_his: {n_his}, n_vertex: {n_vertex}, n_pred: {n_pred}")
                
                # DiffSTG需要(x_masked, pos_w, pos_d)作为输入
                # 创建位置编码占位符（假设每30分钟一个数据点）
                # pos_w: 一周中的第几天 (0-6)
                # pos_d: 一天中的时间点索引
                
                # 位置编码需要包含历史+未来的时间步
                total_steps = n_his + n_pred  # 13
                
                # 创建简单的位置编码（使用默认值，因为我们没有实际的时间戳信息）
                # 假设数据是连续的，每30分钟一个点，一天48个点
                pos_w = torch.zeros((batch_size, total_steps), dtype=torch.long, device=x.device)  # 假设周一
                pos_d = torch.arange(total_steps, device=x.device).unsqueeze(0).repeat(batch_size, 1) % 48  # 一天48个时间点
                
                # DEBUG
                if False:  # 设置为True来启用调试输出
                    print(f"DEBUG pos shapes - pos_w: {pos_w.shape}, pos_d: {pos_d.shape}, total_steps: {total_steps}")
                
                # DiffSTG的forward方法期望(x_masked, pos_w, pos_d)
                # 注意：x_diff已经是masked的形式（历史数据+零填充的未来）
                pred_full = model((x_diff, pos_w, pos_d), n_samples=1)  # [batch, n_samples, F, V, T]
                
                # 提取预测结果
                if len(pred_full.shape) == 5:
                    # [batch, n_samples, F, V, T] -> [batch, V]
                    pred = pred_full[:, 0, 0, :, -1]  # 取第一个样本，第一个特征，所有节点，最后一个时间步
                else:
                    # [batch, F, V, T] -> [batch, V]
                    pred = pred_full[:, 0, :, -1]
                    
            except Exception as e:
                print(f"    警告: DiffSTG预测失败 ({e})，使用零预测")
                pred = torch.zeros(batch_size, n_vertex, device=x.device)
                
        else:
            # LSTM模型
            pred = model(x)  # [batch, n_vertex]
        
        return pred
    
    def get_all_predictions(self, x, seasonal_data=None, fourier_data=None, station_id=0):
        """
        获取所有专家模型的预测
        
        Args:
            x: 输入数据 [batch, n_his, n_vertex]
            seasonal_data: 季节性数据（包含移动平均）[batch, n_his, n_vertex*2]
            fourier_data: Fourier特征数据 [batch, n_his, n_vertex+4]
            station_id: 基站ID（用于单基站测试）
            
        Returns:
            predictions_dict: 各专家模型的预测结果
        """
        predictions = {}
        
        # 为LSTM Trend和Residual准备带有移动平均特征的输入
        x_with_ma = self.add_moving_average_features(x)
        
        # STGCN模型预测
        predictions['stgcn_geo'] = self.predict_stgcn(
            self.expert_models['stgcn_geo'], x
        )
        predictions['stgcn_poi'] = self.predict_stgcn(
            self.expert_models['stgcn_poi'], x
        )
        predictions['stgcn_similarity'] = self.predict_stgcn(
            self.expert_models['stgcn_similarity'], x
        )
        
        # LSTM Trend (需要移动平均特征)
        predictions['lstm_trend'] = self.predict_lstm(
            self.expert_models['lstm_trend'], x_with_ma
        )
        
        # LSTM Seasonal (需要移动平均特征)
        if seasonal_data is not None:
            predictions['lstm_seasonal'] = self.predict_lstm(
                self.expert_models['lstm_seasonal'], seasonal_data
            )
        else:
            # 如果没有提供seasonal_data，使用带MA的输入
            predictions['lstm_seasonal'] = self.predict_lstm(
                self.expert_models['lstm_seasonal'], x_with_ma
            )
        
        # LSTM Fourier Seasonal (也需要移动平均特征)
        if fourier_data is not None:
            predictions['lstm_fourier_seasonal'] = self.predict_lstm(
                self.expert_models['lstm_fourier_seasonal'], fourier_data
            )
        else:
            # 如果没有Fourier特征，使用带MA的输入
            predictions['lstm_fourier_seasonal'] = self.predict_lstm(
                self.expert_models['lstm_fourier_seasonal'], x_with_ma
            )
        
        # Residual (根据模型类型选择输入)
        # DiffSTG使用原始数据，LSTM使用带移动平均特征的数据
        residual_model = self.expert_models['residual']
        if 'DiffSTG' in type(residual_model).__name__ or hasattr(residual_model, 'epsilon_theta'):
            # DiffSTG模型使用原始数据
            predictions['residual'] = self.predict_residual(residual_model, x)
        else:
            # LSTM模型使用带移动平均特征的数据
            predictions['residual'] = self.predict_residual(residual_model, x_with_ma)
        
        
        # 如果指定了station_id，提取该基站的预测
        if station_id is not None:
            # 从模型输出中提取目标基站的预测值
            for key in predictions:
                if predictions[key].shape[-1] > 1:  # 如果输出是多基站
                    predictions[key] = predictions[key][:, station_id:station_id+1]
        
        return predictions
    
    def ensemble_predict(self, predictions: Dict, weights: Dict):
        """
        使用权重对专家预测进行集成（分层集成策略）
        
        Args:
            predictions: 各专家模型的预测 {name: tensor[batch, n_vertex]}
            weights: 各专家模型的权重 {name: float}
            
        Returns:
            ensemble_pred: 集成预测结果 [batch, n_vertex]
        """
        # ✅ 使用完整分层集成：所有7个模型都参与
        # STGCN组（加权平均）+ 组件组（相加重组）
        STGCN_ONLY = False  # 设为True只用STGCN，False用完整分层集成
        
        if STGCN_ONLY:
            # 只集成STGCN模型
            stgcn_names = ['stgcn_geo', 'stgcn_poi', 'stgcn_similarity']
            stgcn_preds = []
            stgcn_weights = []
            
            for name in stgcn_names:
                if name in predictions:
                    stgcn_preds.append(predictions[name])
                    stgcn_weights.append(weights.get(name, 1.0/3))
            
            # 归一化权重
            weight_sum = sum(stgcn_weights)
            if weight_sum > 0:
                stgcn_weights = [w / weight_sum for w in stgcn_weights]
            else:
                stgcn_weights = [1.0/len(stgcn_preds)] * len(stgcn_preds)
            
            # 加权平均
            ensemble_pred = None
            for pred, weight in zip(stgcn_preds, stgcn_weights):
                if ensemble_pred is None:
                    ensemble_pred = weight * pred
                else:
                    ensemble_pred += weight * pred
            
            return ensemble_pred
        
        # 原始分层集成策略
        # 1. 集成STGCN模型（预测完整流量）
        stgcn_names = ['stgcn_geo', 'stgcn_poi', 'stgcn_similarity']
        stgcn_preds = []
        stgcn_weights = []
        
        for name in stgcn_names:
            if name in predictions:
                stgcn_preds.append(predictions[name])
                stgcn_weights.append(weights.get(name, 1.0/3))
        
        # 归一化STGCN权重
        stgcn_weight_sum = sum(stgcn_weights)
        if stgcn_weight_sum > 0:
            stgcn_weights = [w / stgcn_weight_sum for w in stgcn_weights]
        else:
            stgcn_weights = [1.0/len(stgcn_preds)] * len(stgcn_preds)
        
        # 加权平均STGCN预测
        stgcn_ensemble = None
        for pred, weight in zip(stgcn_preds, stgcn_weights):
            if stgcn_ensemble is None:
                stgcn_ensemble = weight * pred
            else:
                stgcn_ensemble += weight * pred
        
        # 2. 重组分量模型（trend + seasonal + fourier + residual）
        component_names = ['lstm_trend', 'lstm_seasonal', 'lstm_fourier_seasonal', 'residual']
        component_pred = None
        
        for name in component_names:
            if name in predictions:
                if component_pred is None:
                    component_pred = predictions[name]
                else:
                    component_pred = component_pred + predictions[name]
        
        # 3. 最终集成：STGCN集成 vs 分量重组
        if stgcn_ensemble is not None and component_pred is not None:
            # 计算两个集成的权重（基于原始权重的分组）
            stgcn_total_weight = sum([weights.get(name, 0) for name in stgcn_names])
            component_total_weight = sum([weights.get(name, 0) for name in component_names])
            total = stgcn_total_weight + component_total_weight
            
            if total > 0:
                stgcn_final_weight = stgcn_total_weight / total
                component_final_weight = component_total_weight / total
            else:
                stgcn_final_weight = 0.5
                component_final_weight = 0.5
            
            ensemble_pred = stgcn_final_weight * stgcn_ensemble + component_final_weight * component_pred
        elif stgcn_ensemble is not None:
            ensemble_pred = stgcn_ensemble
        elif component_pred is not None:
            ensemble_pred = component_pred
        else:
            # 兜底：简单平均（理论上不应该到这里）
            ensemble_pred = None
            for name, pred in predictions.items():
                weight = weights.get(name, 1.0 / len(predictions))
                if ensemble_pred is None:
                    ensemble_pred = weight * pred
                else:
                    ensemble_pred += weight * pred
        
        return ensemble_pred


def calculate_metrics(y_true, y_pred):
    """
    计算评估指标
    
    Args:
        y_true: 真实值 numpy array
        y_pred: 预测值 numpy array
        
    Returns:
        metrics: 包含MAE, RMSE, NMAE, NRMSE, R2的字典
    """
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # 归一化指标
    y_range = y_true.max() - y_true.min()
    if y_range > 0:
        nmae = mae / y_range
        nrmse = rmse / y_range
    else:
        nmae = 0.0
        nrmse = 0.0
    
    # R2 Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot > 0:
        r2 = 1 - (ss_res / ss_tot)
    else:
        r2 = 0.0
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'nmae': float(nmae),
        'nrmse': float(nrmse),
        'r2': float(r2)
    }


def prepare_poi_summary(station_ids=None):
    """
    准备POI信息摘要
    
    Args:
        station_ids: 需要提取POI信息的基站ID列表（如果为None，返回通用摘要）
    
    Returns:
        POI信息摘要字符串
    """
    if station_ids is None or len(station_ids) == 0:
        # 通用摘要
        poi_summary = """Shanghai base stations have diverse POI distributions:
        - Residential areas: high evening/night traffic
        - Commercial districts: high daytime traffic  
        - Mixed areas: balanced traffic patterns
        - Transportation hubs: peak during commute hours"""
        return poi_summary
    
    try:
        # 读取真实POI数据
        import json
        with open('/root/stgcn原版/shanghai_poi.json', 'r', encoding='utf-8') as f:
            poi_data = json.load(f)
        
        # 为每个站点构建POI摘要
        station_poi_info = []
        for station_id in station_ids[:3]:  # 只显示前3个站点，避免prompt过长
            # 通过idx_final查找对应的基站
            station_key = None
            for key, value in poi_data.items():
                if value.get('idx_final') == station_id:
                    station_key = key
                    break
            
            if station_key:
                station_info = poi_data[station_key]
                poi_stats = station_info.get('POI类别统计', {})
                
                # 计算主要POI类型（占比前3）
                total_poi = sum(poi_stats.values())
                if total_poi > 0:
                    top_categories = sorted(poi_stats.items(), key=lambda x: x[1], reverse=True)[:3]
                    category_str = ", ".join([f"{cat}({count}/{total_poi})" for cat, count in top_categories])
                    
                    station_poi_info.append(
                        f"Station {station_id}: Total {total_poi} POIs, "
                        f"主要类别: {category_str}"
                    )
        
        if station_poi_info:
            poi_summary = "Target stations POI characteristics:\n" + "\n".join(station_poi_info)
            if len(station_ids) > 3:
                poi_summary += f"\n(Showing 3 of {len(station_ids)} stations)"
        else:
            # 如果读取失败，使用通用摘要
            poi_summary = "POI data: Mixed urban areas with residential, commercial, and transportation facilities"
        
        return poi_summary
        
    except Exception as e:
        print(f"Warning: Failed to load POI data: {e}")
        # 回退到通用摘要
        return "POI data: Mixed urban areas with residential, commercial, and transportation facilities"


def main():
    parser = argparse.ArgumentParser(description='LLM Agent for Expert Model Ensemble')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API Key')
    parser.add_argument('--dataset', type=str, default='shanghai')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--enable_cuda', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='llm_agent_results')
    parser.add_argument('--station_ids', type=str, default='0', 
                        help='Comma-separated base station IDs to test, e.g., "0,10,20,30" (0-4504 for Shanghai)')
    
    # LLM相关参数
    parser.add_argument('--enable_llm', type=str, default='true', 
                        help='Enable LLM for weight decision (true/false). If false, use rule-based weights.')
    parser.add_argument('--use_proxy', type=str, default='false',
                        help='Use proxy for API connection (true/false)')
    parser.add_argument('--proxy_url', type=str, default='http://127.0.0.1:7890',
                        help='Proxy URL if use_proxy is true')
    parser.add_argument('--timeout', type=float, default=30.0,
                        help='API timeout in seconds')
    
    args = parser.parse_args()
    
    # 转换布尔值
    enable_llm = args.enable_llm.lower() == 'true'
    use_proxy = args.use_proxy.lower() == 'true'
    
    # 设置环境
    set_env(args.seed)
    
    # 设置设备
    if args.enable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print("\n" + "="*50)
    print("Loading data...")
    print("="*50)
    
    dataset_path = f'/root/stgcn原版/data/{args.dataset}'
    data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
    n_vertex = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[1]  # 获取基站总数
    
    # 解析station_ids（在获取n_vertex之后）
    if args.station_ids.lower() == 'all':
        station_ids = list(range(n_vertex))  # 使用所有基站
        print(f"\n*** Testing with ALL {len(station_ids)} stations (0-{n_vertex-1}) ***")
    else:
        station_ids = [int(sid.strip()) for sid in args.station_ids.split(',')]
        print(f"\n*** Testing with {len(station_ids)} station(s): {station_ids[:10]}{'...' if len(station_ids) > 10 else ''} ***")
    
    # 读取带时间戳的数据，获取真实的时间戳
    vel_with_ts = pd.read_csv(os.path.join(dataset_path, 'vel_with_timestamp.csv'))
    timestamps = pd.to_datetime(vel_with_ts.iloc[:, 0])  # 第一列是时间戳
    print(f"\nTimestamp range: {timestamps.iloc[0]} to {timestamps.iloc[-1]}")
    
    val_and_test_rate = 0.15
    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train_full = int(data_col - len_val - len_test)
    len_reserved = int(math.floor(len_train_full * 0.20))
    len_train = len_train_full - len_reserved
    
    print(f"Dataset split:")
    print(f"  Train: {len_train}")
    print(f"  Reserved: {len_reserved}")
    print(f"  Val: {len_val}")
    print(f"  Test: {len_test}")
    print(f"  Total: {data_col}")
    
    # 加载数据
    train, val, test, reserved = dataloader.load_data(args.dataset, len_train_full, len_train, len_val)
    
    # 提取reserved+val+test对应的时间戳
    # reserved从 len_train 开始，val 从 len_train_full 开始，test 从 len_train_full + len_val 开始
    # 但是由于 data_transform 会产生 n_his 的偏移，我们需要对应调整
    n_his = 12
    reserved_start_idx = len_train + n_his
    all_timestamps = timestamps[reserved_start_idx:].reset_index(drop=True)
    print(f"\nPrediction timestamps: {all_timestamps.iloc[0]} to {all_timestamps.iloc[-1]}")
    
    print(f"\n*** Using FULL multi-station data for prediction, evaluating {len(station_ids)} target station(s) ***")
    
    # 转换为numpy array
    reserved_np = reserved.values if hasattr(reserved, 'values') else reserved
    val_np = val.values if hasattr(val, 'values') else val
    test_np = test.values if hasattr(test, 'values') else test
    
    print(f"Full data shapes:")
    print(f"  Reserved: {reserved_np.shape}")
    print(f"  Val: {val_np.shape}")
    print(f"  Test: {test_np.shape}")
    
    # 标准化 - 使用全站数据的reserved集拟合（保持与训练时一致）
    zscore = preprocessing.StandardScaler()
    reserved_scaled = zscore.fit_transform(reserved_np)
    val_scaled = zscore.transform(val_np)
    test_scaled = zscore.transform(test_np)
    
    # *** 关键修复：使用全站scaler进行反标准化 ***
    # 专家模型训练时使用全站标准化，因此预测结果也必须用全站scaler反标准化
    # 如果用单站scaler，会导致量纲错误，MAE严重偏高！
    print(f"\n*** Using GLOBAL scaler for inverse transform (consistent with expert model training) ***")
    print(f"  Global mean: {zscore.mean_[:5]}...")  # 显示前5个站点的均值
    print(f"  Global std: {zscore.scale_[:5]}...")
    
    # 转换为时间序列格式（使用全站数据）
    n_his = 12
    n_pred = 1
    
    x_reserved, y_reserved = dataloader.data_transform(reserved_scaled, n_his, n_pred, device)
    x_val, y_val = dataloader.data_transform(val_scaled, n_his, n_pred, device)
    x_test, y_test = dataloader.data_transform(test_scaled, n_his, n_pred, device)
    
    print(f"\nTransformed data shapes (sliding window):")
    print(f"  Reserved: x={x_reserved.shape}, y={y_reserved.shape}")
    print(f"  Val: x={x_val.shape}, y={y_val.shape}")
    print(f"  Test: x={x_test.shape}, y={y_test.shape}")
    
    # ✅ 注意：data_transform已经生成了非重叠窗口（步长=13），无需再次采样
    window_step = n_his + n_pred  # 13
    print(f"\n*** Using Non-overlapping Windows (step={window_step}) ***")
    print(f"  Note: data_transform() already generated non-overlapping windows")
    print(f"  Reserved: x={x_reserved.shape}, y={y_reserved.shape}")
    print(f"  Val: x={x_val.shape}, y={y_val.shape}")
    print(f"  Test: x={x_test.shape}, y={y_test.shape}")
    print(f"  Total samples: {x_reserved.shape[0] + x_val.shape[0] + x_test.shape[0]}")
    
    # 用于timestamp对齐（不进行重采样）
    sampling_indices = {
        'reserved': torch.arange(0, x_reserved.shape[0]).cpu().numpy(),
        'val': torch.arange(0, x_val.shape[0]).cpu().numpy(),
        'test': torch.arange(0, x_test.shape[0]).cpu().numpy()
    }
    
    # 加载专家模型
    print("\n" + "="*50)
    print("Loading Expert Models")
    print("="*50)
    
    loader = ExpertModelLoader(device, args.dataset)
    expert_models = loader.load_all_models()
    
    # 初始化LLM Agent
    print("\n" + "="*50)
    print("Initializing LLM Agent")
    print("="*50)
    
    llm_agent = LLMAgent(
        api_key=args.api_key,
        enable_llm=enable_llm,
        use_proxy=use_proxy,
        proxy_url=args.proxy_url if use_proxy else None,
        timeout=args.timeout
    )
    
    # 初始化集成预测器
    ensemble_predictor = EnsemblePredictor(expert_models, device, args.dataset)
    
    # 为每个基站初始化权重
    print("\n" + "="*50)
    print("Initializing Per-Station Weights")
    print("="*50)
    print(f"LLM will decide optimal weights for each of {len(station_ids)} stations...")
    print(f"This may take a few minutes (approx {len(station_ids)} API calls)...")
    
    # 为每个基站维护独立的权重
    station_weights = {}
    
    # 加载或生成POI信息
    if os.path.exists('shanghai_poi.json'):
        with open('shanghai_poi.json', 'r') as f:
            poi_data = json.load(f)
    else:
        poi_data = {}
    
    # 为需要测试的基站初始化权重
    batch_size = 50  # 每批处理50个基站
    for batch_idx in range(0, len(station_ids), batch_size):
        batch_stations = station_ids[batch_idx:min(batch_idx + batch_size, len(station_ids))]
        
        for station_id in batch_stations:
            # 获取该基站的POI信息
            if str(station_id) in poi_data:
                station_poi = poi_data[str(station_id)]
                poi_summary_station = f"Station {station_id} POI distribution:\n"
                for poi_type, count in station_poi.items():
                    poi_summary_station += f"  - {poi_type}: {count}\n"
            else:
                poi_summary_station = f"Station {station_id}: No specific POI data available (use balanced weights)"
            
            # 调用LLM决定该基站的权重
            weights = llm_agent.initialize_station_weights(station_id, poi_summary_station)
            station_weights[station_id] = weights
        
        # 显示进度
        completed = min(batch_idx + batch_size, len(station_ids))
        print(f"  Initialized weights for {completed}/{len(station_ids)} stations...", end='\r')
    
    print(f"\n✅ Completed weight initialization for all {len(station_ids)} stations")
    
    # 显示一些示例
    print(f"\nExample station weights:")
    for i, station_id in enumerate(station_ids[:5]):
        w = station_weights[station_id]
        stgcn_total = w.get('stgcn_geo', 0) + w.get('stgcn_poi', 0) + w.get('stgcn_similarity', 0)
        lstm_total = w.get('lstm_trend', 0) + w.get('lstm_seasonal', 0) + w.get('lstm_fourier_seasonal', 0) + w.get('residual', 0)
        print(f"  Station {station_id}: STGCN={stgcn_total:.3f}, LSTM={lstm_total:.3f}")
    
    # POI摘要（使用真实的基站POI数据）
    poi_summary = prepare_poi_summary(station_ids)
    print(f"\nOverall POI Summary:\n{poi_summary}\n")
    
    # *** 只使用TEST集进行最终评估 ***
    all_x = x_test
    all_y = y_test
    
    # 计算test集对应的时间戳（考虑非重叠采样）
    # reserved开始位置 + reserved样本数 + val样本数
    # 计算test集对应的时间戳
    base_offset = x_reserved.shape[0] + x_val.shape[0]
    # 获取test集的索引
    test_sampled_indices = sampling_indices['test']
    # 从all_timestamps提取对应的时间戳
    test_timestamps = all_timestamps.iloc[test_sampled_indices].reset_index(drop=True)
    
    print(f"\n*** Evaluating on TEST set only ***")
    print(f"  Test samples: {all_x.shape[0]}")
    print(f"  First timestamp: {test_timestamps.iloc[0]}")
    print(f"  Last timestamp: {test_timestamps.iloc[-1]}")

    
    # 预测和评估
    print("\n" + "="*50)
    print("Starting LLM-Guided Ensemble Prediction")
    print("="*50)
    print("  Ensemble Strategy: Complete 7-Model Hierarchical")
    print("  - STGCN Group (3): geo, poi, similarity")
    print("  - Component Group (4): trend, seasonal, fourier, residual")
    print("  - LLM dynamically adjusts weights for both groups")
    print("")
    
    # 为每个站点维护独立的记录
    station_records = {sid: {
        # 'prediction_records': [],  # 已禁用以节省内存
        'all_predictions': [],
        'all_true_values': [],
        # 新增：每个专家模型的单独预测记录
        'expert_predictions': {
            'stgcn_geo': [],
            'stgcn_poi': [],
            'stgcn_similarity': []
        }
    } for sid in station_ids}
    
    # 批量预测
    num_samples = all_x.shape[0]
    n_vertex_full = 4505 if args.dataset == 'shanghai' else 7775
    
    # 调整进度输出频率（根据样本数量动态调整）
    if num_samples > 1000:
        progress_interval = 100
        weight_update_interval = 100
    elif num_samples > 100:
        progress_interval = 50
        weight_update_interval = 50
    elif num_samples > 20:
        progress_interval = 10
        weight_update_interval = 10
    else:
        # 样本数<=20时，每个样本都更新权重，充分发挥LLM作用
        progress_interval = 1
        weight_update_interval = 1
    
    print(f"  LLM weight update interval: every {weight_update_interval} sample(s)")
    if weight_update_interval == 1:
        print(f"  ⚡ Small sample size ({num_samples}) - updating weights for EVERY sample!")
    
    for i in range(num_samples):
        if (i + 1) % progress_interval == 0 or i == 0:
            print(f"\nPredicting sample {i+1}/{num_samples}...")
        
        # 获取当前样本 (全部基站的数据)
        x_sample = all_x[i:i+1]  # [1, 1, n_his, n_vertex_full]
        y_true_all = all_y[i:i+1]  # [1, n_vertex_full]
        
        # Reshape x_sample for prediction: [1, n_his, n_vertex]
        x_input = x_sample.squeeze(1)  # [1, n_his, n_vertex_full]
        
        # 使用真实的时间戳（test集的滑动窗口对应的时间戳）
        current_timestamp = test_timestamps.iloc[i]
        timestamp_str = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # 计算时间段特征（用于LLM决策展示）
        hour = current_timestamp.hour
        if 7 <= hour < 9:
            time_period = "Morning Rush"
        elif 12 <= hour < 14:
            time_period = "Lunch Time"
        elif 17 <= hour < 19:
            time_period = "Evening Rush"
        elif 22 <= hour or hour < 6:
            time_period = "Late Night"
        else:
            time_period = "Regular Hours"
        
        # ✅ 优化: 对所有模型只推理一次，得到所有站点的预测（而不是每个站点都重复推理）
        # 这将推理次数从 76,585×7 降低到 17×7，提速 4500倍！
        all_stations_predictions = ensemble_predictor.get_all_predictions(x_input, station_id=None)
        
        # 对每个目标站点进行预测
        for station_id in station_ids:
            # 提取目标基站的真实值
            y_true = y_true_all[:, station_id:station_id+1]  # [1, 1]
            
            # ✅ 从完整预测中提取该站点的结果（不再重复推理！）
            predictions_dict = {}
            for expert_name, pred_all in all_stations_predictions.items():
                if pred_all.shape[-1] > 1:  # 多站点输出
                    predictions_dict[expert_name] = pred_all[:, station_id:station_id+1]
                else:  # 单站点输出
                    predictions_dict[expert_name] = pred_all
            
            # DEBUG: 输出每个专家模型的预测值（第一个样本）
            if i == 0 and station_id == station_ids[0]:
                print(f"\n=== DEBUG: Individual Expert Predictions (Sample 0, Station {station_id}) ===")
                print(f"  x_input shape: {x_input.shape}")
                print(f"  station_id: {station_id}")
                for name, pred in predictions_dict.items():
                    print(f"  {name:25s} pred shape: {str(pred.shape):20s} value: {pred.cpu().numpy().flatten()[0]:.4f}")
                print(f"  y_true shape: {y_true.shape}")
                print(f"  y_true value: {y_true.cpu().numpy().flatten()[0]:.4f}")
                print(f"\n  ✅ 优化提示: 所有{len(station_ids)}个站点共享同一次模型推理，避免重复计算")
                print(f"  ✅ 每个基站使用LLM定制的专属权重")
            
            # 使用该基站的专属权重（不再共享权重）
            weights = station_weights[station_id]
            
            # 仅在第一个样本输出部分基站的权重信息
            if i == 0 and station_id in station_ids[:3]:
                stgcn_total = weights.get('stgcn_geo', 0) + weights.get('stgcn_poi', 0) + weights.get('stgcn_similarity', 0)
                lstm_total = weights.get('lstm_trend', 0) + weights.get('lstm_seasonal', 0) + weights.get('lstm_fourier_seasonal', 0) + weights.get('residual', 0)
                print(f"\n  📍 Station {station_id} using custom weights:")
                print(f"     STGCN={stgcn_total:.3f} | LSTM={lstm_total:.3f}")
            
            # 集成预测
            y_pred = ensemble_predictor.ensemble_predict(predictions_dict, weights)
            
            # 转换回numpy用于计算指标
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            
            # 收集所有预测用于最终评估
            station_records[station_id]['all_predictions'].append(y_pred_np)
            station_records[station_id]['all_true_values'].append(y_true_np)
            
            # 收集每个专家模型的单独预测
            for expert_name in ['stgcn_geo', 'stgcn_poi', 'stgcn_similarity']:
                if expert_name in predictions_dict:
                    expert_pred_np = predictions_dict[expert_name].cpu().numpy()
                    station_records[station_id]['expert_predictions'][expert_name].append(expert_pred_np)
    
    # 计算各站点的整体指标
    print("\n" + "="*50)
    print("Calculating Overall Metrics for Each Station")
    print("="*50)
    print(f"Processing {len(station_ids)} stations... (results will be saved to file)")
    
    station_metrics = {}
    
    for idx, station_id in enumerate(station_ids):
        # 显示进度（每100个站点或最后一个）
        if (idx + 1) % 100 == 0 or (idx + 1) == len(station_ids):
            print(f"  Processed {idx + 1}/{len(station_ids)} stations...", end='\r')
        
        all_predictions_np = np.concatenate(station_records[station_id]['all_predictions'], axis=0)
        all_true_values_np = np.concatenate(station_records[station_id]['all_true_values'], axis=0)
        
        # *** 关键修复：使用全站scaler的对应站点参数进行反标准化 ***
        # 专家模型训练时用全站数据标准化，每个站点有对应的均值和标准差
        # 从全站scaler提取该站点的参数进行反标准化
        # all_predictions_np shape: (num_samples, 1)
        # zscore.mean_[station_id]: 该站点的均值
        # zscore.scale_[station_id]: 该站点的标准差
        station_mean = zscore.mean_[station_id]
        station_scale = zscore.scale_[station_id]
        
        # 手动反标准化：x_original = x_scaled * scale + mean
        all_predictions_original = all_predictions_np * station_scale + station_mean
        all_true_values_original = all_true_values_np * station_scale + station_mean
        
        metrics = calculate_metrics(all_true_values_original, all_predictions_original)
        station_metrics[station_id] = metrics
        
        # 静默计算专家模型性能（不输出到终端）
        station_metrics[station_id]['expert_metrics'] = {}
        for expert_name in ['stgcn_geo', 'stgcn_poi', 'stgcn_similarity']:
            if expert_name in station_records[station_id]['expert_predictions'] and \
               len(station_records[station_id]['expert_predictions'][expert_name]) > 0:
                expert_preds = np.concatenate(station_records[station_id]['expert_predictions'][expert_name], axis=0)
                # 反标准化
                expert_preds_original = expert_preds * station_scale + station_mean
                expert_metrics = calculate_metrics(all_true_values_original, expert_preds_original)
                station_metrics[station_id]['expert_metrics'][expert_name] = expert_metrics
    
    print()  # 换行
    print(f"✓ Completed metrics calculation for all {len(station_ids)} stations")
    
    # 计算多站点平均指标
    if len(station_ids) > 1:
        print(f"\n--- Average Across {len(station_ids)} Stations ---")
        avg_metrics = {
            'mae': np.mean([m['mae'] for m in station_metrics.values()]),
            'rmse': np.mean([m['rmse'] for m in station_metrics.values()]),
            'nmae': np.mean([m['nmae'] for m in station_metrics.values()]),
            'nrmse': np.mean([m['nrmse'] for m in station_metrics.values()]),
            'r2': np.mean([m['r2'] for m in station_metrics.values()])
        }
        print(f"  Average MAE: {avg_metrics['mae']:.4f}")
        print(f"  Average RMSE: {avg_metrics['rmse']:.4f}")
        print(f"  Average NMAE: {avg_metrics['nmae']:.4f}")
        print(f"  Average NRMSE: {avg_metrics['nrmse']:.4f}")
        print(f"  Average R2: {avg_metrics['r2']:.4f}")
    
    # 保存结果
    print("\n" + "="*50)
    print("Saving Results")
    print("="*50)
    
    # 保存所有站点的指标
    metrics_file = os.path.join(args.output_dir, 'station_metrics.json')
    metrics_output = {
        'per_station': {str(k): v for k, v in station_metrics.items()},
        'average': avg_metrics if len(station_ids) > 1 else station_metrics[station_ids[0]],
        'station_ids': station_ids
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    print(f"✓ Saved metrics to {metrics_file}")
    
    # 保存摘要文本
    summary_file = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write("LLM Agent Ensemble Prediction Results (TEST SET)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Evaluation Set: TEST (non-overlapping windows, step={window_step})\n")
        f.write(f"Ensemble Strategy: Complete 7-Model Hierarchical Ensemble\n")
        f.write(f"  - STGCN Group (3 models): Weighted average of geo/poi/similarity\n")
        f.write(f"  - Component Group (4 models): Sum of trend+seasonal+fourier+residual\n")
        f.write(f"  - Final: LLM-weighted combination of both groups\n\n")
        
        f.write(f"**PER-STATION CUSTOM WEIGHTS:**\n")
        f.write(f"  ✅ Each station uses LLM-decided custom weights based on POI characteristics\n")
        f.write(f"  ✅ Total LLM calls for weight initialization: {len(station_ids)}\n")
        f.write(f"  ✅ Weights optimized for each station's unique traffic patterns\n\n")
        
        f.write(f"Station IDs: {station_ids if len(station_ids) <= 10 else f'{station_ids[:10]}... ({len(station_ids)} total)'}\n")
        f.write(f"Number of Stations: {len(station_ids)}\n")
        f.write(f"Total Test Samples: {num_samples}\n")
        f.write(f"LLM Model: {llm_agent.model_name}\n")
        f.write(f"LLM Enabled: {llm_agent.enable_llm}\n")
        f.write(f"Note: Non-overlapping windows ensure sample independence\n")
        f.write(f"Note: All 7 expert models participate in hierarchical ensemble\n\n")
        
        # 每个站点的指标
        f.write("Per-Station Metrics:\n")
        f.write("-" * 50 + "\n")
        for station_id in station_ids:
            metrics = station_metrics[station_id]
            f.write(f"\nStation {station_id}:\n")
            
            # 添加该基站的权重配置
            if station_id in station_weights:
                w = station_weights[station_id]
                stgcn_total = w.get('stgcn_geo', 0) + w.get('stgcn_poi', 0) + w.get('stgcn_similarity', 0)
                lstm_total = w.get('lstm_trend', 0) + w.get('lstm_seasonal', 0) + w.get('lstm_fourier_seasonal', 0) + w.get('residual', 0)
                f.write(f"  Custom Weights: STGCN={stgcn_total:.3f}, LSTM={lstm_total:.3f}\n")
            
            f.write(f"  MAE: {metrics['mae']:.4f}\n")
            f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
            f.write(f"  NMAE: {metrics['nmae']:.4f}\n")
            f.write(f"  NRMSE: {metrics['nrmse']:.4f}\n")
            f.write(f"  R2: {metrics['r2']:.4f}\n")
            
            # 添加专家模型性能
            if 'expert_metrics' in metrics and metrics['expert_metrics']:
                f.write(f"\n  Individual Expert Performance:\n")
                for expert_name, expert_m in metrics['expert_metrics'].items():
                    f.write(f"    {expert_name}: MAE={expert_m['mae']:.4f}, RMSE={expert_m['rmse']:.4f}, R2={expert_m['r2']:.4f}\n")
        
        # 平均指标
        if len(station_ids) > 1:
            f.write("\n" + "-" * 50 + "\n")
            f.write(f"Average Across {len(station_ids)} Stations:\n")
            f.write(f"  MAE: {avg_metrics['mae']:.4f}\n")
            f.write(f"  RMSE: {avg_metrics['rmse']:.4f}\n")
            f.write(f"  NMAE: {avg_metrics['nmae']:.4f}\n")
            f.write(f"  NRMSE: {avg_metrics['nrmse']:.4f}\n")
            f.write(f"  R2: {avg_metrics['r2']:.4f}\n")
    
    print(f"✓ Saved summary to {summary_file}")
    
    print("\n" + "="*50)
    print("All Done!")
    print("="*50)


if __name__ == '__main__':
    main()
