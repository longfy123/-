import json
import re
import time
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from enum import Enum
from typing import Dict, List
import openai

class LLMAgent:
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini",
                 enable_llm: bool = True, use_proxy: bool = False,
                 proxy_url: str = None, timeout: float = 30.0):
        self.enable_llm = enable_llm
        self.timeout = timeout
        client_kwargs = {
            "api_key": api_key,
            "base_url": "https://api.openai-proxy.org/v1",
            "timeout": timeout
        }
        if use_proxy and proxy_url:
            import httpx
            client_kwargs["http_client"] = httpx.Client(proxies=proxy_url)
        self.client = openai.OpenAI(**client_kwargs)
        self.model_name = model_name

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0

        self.expert_names = [
            'spatial_geo', 'spatial_poi', 'spatial_similarity',
            'linear_trend', 'lstm_seasonal', 'fourier_seasonal', 'residual'
        ]
        w = round(1.0 / len(self.expert_names), 6)
        self.current_weights = {name: w for name in self.expert_names}

        self.performance_history = []
        self.max_history_len = 10

        class EnsembleStrategy(Enum):
            BASE_WEIGHTED = "base_weighted"
            SCENE_ENHANCED = "scene_enhanced"
            DYNAMIC_BALANCE = "dynamic_balance"
            TEMPORAL_ADAPTIVE = "temporal_adaptive"

        self.EnsembleStrategy = EnsembleStrategy
        self.experience_summary = ""
        self.strategy_configs = {
            EnsembleStrategy.BASE_WEIGHTED: {"description": "Basic weighted average", "applicable_scenes": ["general"]},
            EnsembleStrategy.SCENE_ENHANCED: {"description": "Enhanced by scene features", "applicable_scenes": ["distinctive features"]},
            EnsembleStrategy.DYNAMIC_BALANCE: {"description": "Dynamic balance between two groups", "applicable_scenes": ["general"]},
            EnsembleStrategy.TEMPORAL_ADAPTIVE: {"description": "Adaptive to temporal features", "applicable_scenes": ["strong temporal patterns"]},
        }

        self.reliance_scores = {}
        self.scene_performance = defaultdict(list)
        self.scene_strategy_map = {}
        self.station_reliance_scores = {}

    # ──────────────────────────────────────────────────────────────
    # Scene feature extraction
    # ──────────────────────────────────────────────────────────────
    def extract_scene_features(self, timestamp: str, poi_info: str,
                               historical_seq: list = None) -> dict:
        dt = pd.to_datetime(timestamp)
        hour = dt.hour
        weekday = dt.weekday()
        is_weekend = weekday >= 5

        if 7 <= hour < 9:
            time_period = "morning_rush"
        elif 17 <= hour < 19:
            time_period = "evening_rush"
        elif 22 <= hour or hour < 6:
            time_period = "late_night"
        else:
            time_period = "regular"

        area_type = "mixed"
        poi_lower = poi_info.lower()
        if "offices" in poi_lower or "shopping" in poi_lower:
            area_type = "commercial"
        elif "residential" in poi_lower:
            area_type = "residential"
        elif "transportation" in poi_lower:
            area_type = "transportation"

        scene = {
            "time_period": time_period,
            "area_type": area_type,
            "hour": hour,
            "weekday": weekday,
            "is_weekend": is_weekend,
            "scene_key": time_period
        }

        if historical_seq and len(historical_seq) > 0:
            if len(historical_seq) >= 2:
                trend = historical_seq[-1] - historical_seq[0]
                if abs(trend) > 0.1 * np.mean(historical_seq):
                    scene["trend"] = "strong_up" if trend > 0 else "strong_down"
                else:
                    scene["trend"] = "stable"
            volatility = np.std(historical_seq) / (np.mean(historical_seq) + 1e-6)
            scene["volatility"] = "high" if volatility > 0.2 else "low"

        return scene

    # ──────────────────────────────────────────────────────────────
    # Reliance score helpers
    # ──────────────────────────────────────────────────────────────
    def get_reliance_score(self, expert_name: str, scene_key: str,
                           station_id: int = None) -> float:
        if station_id is not None:
            try:
                if (station_id in self.station_reliance_scores and
                        expert_name in self.station_reliance_scores[station_id] and
                        scene_key in self.station_reliance_scores[station_id][expert_name] and
                        self.station_reliance_scores[station_id][expert_name][scene_key]):
                    scores = self.station_reliance_scores[station_id][expert_name][scene_key]
                    weights = np.linspace(0.5, 1.0, len(scores))
                    weights = weights / weights.sum()
                    return float(np.average(scores, weights=weights))
            except Exception:
                pass
        try:
            if (expert_name in self.reliance_scores and
                    scene_key in self.reliance_scores[expert_name] and
                    self.reliance_scores[expert_name][scene_key]):
                scores = self.reliance_scores[expert_name][scene_key]
                return float(sum(scores) / len(scores))
        except Exception:
            pass
        return 0.5

    def _store_reliance(self, station_id: int, scene_key: str,
                        name: str, score: float):
        """Append score to both global and per-station stores (max 20 entries)."""
        if name not in self.reliance_scores:
            self.reliance_scores[name] = {}
        if scene_key not in self.reliance_scores[name]:
            self.reliance_scores[name][scene_key] = []
        self.reliance_scores[name][scene_key].append(score)
        if len(self.reliance_scores[name][scene_key]) > 20:
            self.reliance_scores[name][scene_key].pop(0)

        if station_id not in self.station_reliance_scores:
            self.station_reliance_scores[station_id] = {}
        if name not in self.station_reliance_scores[station_id]:
            self.station_reliance_scores[station_id][name] = {}
        if scene_key not in self.station_reliance_scores[station_id][name]:
            self.station_reliance_scores[station_id][name][scene_key] = []
        self.station_reliance_scores[station_id][name][scene_key].append(score)
        if len(self.station_reliance_scores[station_id][name][scene_key]) > 20:
            self.station_reliance_scores[station_id][name][scene_key].pop(0)

    def update_reliance_scores(self, station_id: int, scene: dict,
                               expert_predictions: Dict, true_value: float):
        """Math-formula reliance update (fallback)."""
        scene_key = scene["scene_key"]
        for expert_name in ('spatial_geo', 'spatial_poi', 'spatial_similarity'):
            if expert_name in expert_predictions:
                error = abs(expert_predictions[expert_name] - true_value)
                self._store_reliance(station_id, scene_key, expert_name, 1.0 / (1.0 + error))

        trend = expert_predictions.get('linear_trend', 0)
        lstm_s = expert_predictions.get('lstm_seasonal', 0)
        fourier_s = expert_predictions.get('fourier_seasonal', 0)
        residual = expert_predictions.get('residual', 0)
        seasonal = (lstm_s + fourier_s) / 2
        component_pred = trend + seasonal + residual
        comp_error = abs(component_pred - true_value)
        comp_score = 1.0 / (1.0 + comp_error)
        self._store_reliance(station_id, scene_key, 'component_group', comp_score)

        lstm_err = abs(lstm_s - true_value)
        fourier_err = abs(fourier_s - true_value)
        lstm_score = 1.0 / (1.0 + lstm_err)
        fourier_score = 1.0 / (1.0 + fourier_err)
        self._store_reliance(station_id, scene_key, 'lstm_seasonal', lstm_score)
        self._store_reliance(station_id, scene_key, 'fourier_seasonal', fourier_score)

        self.scene_performance[scene_key].append({
            'true_value': true_value,
            'predictions': expert_predictions.copy()
        })
        if len(self.scene_performance[scene_key]) > 100:
            self.scene_performance[scene_key].pop(0)

    def update_reliance_scores_with_llm(self, station_id: int, scene: dict,
                                        expert_predictions: Dict, true_value: float,
                                        timestamp: str, poi_info: str):
        """
        LLM-driven reliance score update (Formula 8 in the paper).
        Falls back to math formula if the LLM call fails.
        """
        scene_key = scene["scene_key"]

        spatial_errors = {n: abs(expert_predictions.get(n, 0) - true_value)
                        for n in ('spatial_geo', 'spatial_poi', 'spatial_similarity')}

        trend = expert_predictions.get('linear_trend', 0)
        lstm_s = expert_predictions.get('lstm_seasonal', 0)
        fourier_s = expert_predictions.get('fourier_seasonal', 0)
        residual = expert_predictions.get('residual', 0)
        seasonal = (lstm_s + fourier_s) / 2
        component_pred = trend + seasonal + residual
        comp_error = abs(component_pred - true_value)

        current_spatial_scores = {n: self.get_reliance_score(n, scene_key, station_id)
                                for n in ('spatial_geo', 'spatial_poi', 'spatial_similarity')}
        current_comp_score = self.get_reliance_score('component_group', scene_key, station_id)

        dt = pd.to_datetime(timestamp)
        hour = dt.hour
        weekday = dt.day_name()
        is_weekend = dt.dayofweek >= 5

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

        spatial_lines = ""
        for name in ('spatial_geo', 'spatial_poi', 'spatial_similarity'):
            pred = expert_predictions.get(name, 0)
            err = spatial_errors[name]
            cur = current_spatial_scores[name]
            spatial_lines += f"  - {name}: pred={pred:.4f}, error={err:.4f}, current_reliance={cur:.3f}\n"

        prompt = f"""You are an expert coordination agent for mobile base station traffic forecasting.
A prediction round has just completed. Update the reliance scores based on prediction errors and spatiotemporal context.

**Spatiotemporal Context:**
- Timestamp: {timestamp}
- Time period: {time_period}
- Day: {weekday} ({'Weekend' if is_weekend else 'Weekday'})
- Station POI distribution: {poi_info}

**Ground truth traffic value:** {true_value:.4f}

**Group A — spatial models (individual predictions vs ground truth):**
{spatial_lines}
**Group B — Decomposition models (evaluated as a combined unit):**
- trend={trend:.4f}, seasonal=(lstm={lstm_s:.4f}, fourier={fourier_s:.4f}), residual={residual:.4f}
- Combined prediction: {component_pred:.4f}, error={comp_error:.4f}, current_reliance={current_comp_score:.3f}

**Task:**
1. Update each spatial model's reliance score based on its individual error and scene context (adjust by no more than ±0.2)
2. Update Group B's single combined reliance score based on the combined prediction error (adjust by no more than ±0.2)
3. Also output the internal seasonal split ratio (lstm vs fourier weight, must sum to 1.0)

Output JSON only, no other text. All values must be pre-computed numeric literals (e.g. 0.72), never expressions (e.g. 0.70 + 0.02):
{{
    "spatial_scores": {{
        "spatial_geo": 0.70,
        "spatial_poi": 0.65,
        "spatial_similarity": 0.68
    }},
    "component_group_score": 0.55,
    "seasonal_split": {{
        "lstm_seasonal": 0.5,
        "fourier_seasonal": 0.5
    }}
}}
"""

        new_scores = None
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert coordination agent for traffic forecasting. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=400
            )
            if response.usage:
                self.total_prompt_tokens += response.usage.prompt_tokens
                self.total_completion_tokens += response.usage.completion_tokens
            self.total_requests += 1
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            # Evaluate any arithmetic expressions in JSON values (e.g. "0.70 + 0.02" -> 0.72,
            # "0.9565 / (0.9565 + 0.8070)" -> 0.542373)
            def _eval_expr(m):
                try:
                    return ': ' + str(round(eval(m.group(1)), 6))
                except Exception:
                    return m.group(0)
            content = re.sub(
                r':\s*([-+]?\d*\.?\d+(?:\s*[-+*/]\s*(?:\([-+]?\d*\.?\d+(?:\s*[-+*/]\s*[-+]?\d*\.?\d+)*\)|[-+]?\d*\.?\d+))+)',
                _eval_expr,
                content
            )
            # Extract first complete JSON object, stripping any trailing comments or text
            start = content.find('{')
            depth, end = 0, -1
            for i, ch in enumerate(content[start:], start):
                if ch == '{': depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            result = json.loads(content[start:end])
            new_scores = result
        except Exception as e:
            print(f"  Warning: LLM reliance update failed ({e}), falling back to math formula")
            print(f"  [DEBUG] Raw content: {repr(content)}")

        for name in ('spatial_geo', 'spatial_poi', 'spatial_similarity'):
            if new_scores and 'spatial_scores' in new_scores and name in new_scores['spatial_scores']:
                score = float(np.clip(new_scores['spatial_scores'][name], 0.0, 1.0))
            else:
                score = 1.0 / (1.0 + spatial_errors[name])
            self._store_reliance(station_id, scene_key, name, score)

        if new_scores and 'component_group_score' in new_scores:
            comp_score = float(np.clip(new_scores['component_group_score'], 0.0, 1.0))
        else:
            comp_score = 1.0 / (1.0 + comp_error)
        self._store_reliance(station_id, scene_key, 'component_group', comp_score)

        if new_scores and 'seasonal_split' in new_scores:
            split = new_scores['seasonal_split']
            lstm_w = float(np.clip(split.get('lstm_seasonal', 0.5), 0.0, 1.0))
            fourier_w = float(np.clip(split.get('fourier_seasonal', 0.5), 0.0, 1.0))
            total = lstm_w + fourier_w or 1.0
            self._store_reliance(station_id, scene_key, 'lstm_seasonal', lstm_w / total)
            self._store_reliance(station_id, scene_key, 'fourier_seasonal', fourier_w / total)
        else:
            lstm_err = abs(lstm_s - true_value)
            fourier_err = abs(fourier_s - true_value)
            lstm_score = 1.0 / (1.0 + lstm_err)
            fourier_score = 1.0 / (1.0 + fourier_err)
            self._store_reliance(station_id, scene_key, 'lstm_seasonal', lstm_score)
            self._store_reliance(station_id, scene_key, 'fourier_seasonal', fourier_score)

        self.scene_performance[scene_key].append({
            'true_value': true_value,
            'predictions': expert_predictions.copy()
        })
        if len(self.scene_performance[scene_key]) > 100:
            self.scene_performance[scene_key].pop(0)

    # ──────────────────────────────────────────────────────────────
    # Reliance report (used inside predict_batch_with_llm)
    # ──────────────────────────────────────────────────────────────
    def _build_reliance_report(self, station_id: int, scene: dict,
                               expert_predictions: Dict) -> str:
        report = []
        scene_key = scene["scene_key"]

        for expert in ('spatial_geo', 'spatial_poi', 'spatial_similarity'):
            pred_value = expert_predictions.get(expert, 0)
            scores = self.reliance_scores.get(expert, {}).get(scene_key, [])
            avg_score = sum(scores) / len(scores) if scores else 0.5
            report.append(f"{expert}: pred={pred_value:.4f}, reliability={avg_score:.3f}")

        trend = expert_predictions.get('linear_trend', 0)
        lstm_s = expert_predictions.get('lstm_seasonal', 0)
        fourier_s = expert_predictions.get('fourier_seasonal', 0)
        residual = expert_predictions.get('residual', 0)

        lstm_scores = self.reliance_scores.get('lstm_seasonal', {}).get(scene_key, [])
        fourier_scores = self.reliance_scores.get('fourier_seasonal', {}).get(scene_key, [])
        lstm_w = sum(lstm_scores) / len(lstm_scores) if lstm_scores else 0.5
        fourier_w = sum(fourier_scores) / len(fourier_scores) if fourier_scores else 0.5
        total_sw = lstm_w + fourier_w or 1.0
        seasonal = (lstm_w / total_sw) * lstm_s + (fourier_w / total_sw) * fourier_s

        component_pred = trend + seasonal + residual
        comp_scores = self.reliance_scores.get('component_group', {}).get(scene_key, [])
        comp_reliability = sum(comp_scores) / len(comp_scores) if comp_scores else 0.5
        report.append(f"component_group (trend+seasonal+residual): combined_pred={component_pred:.4f}, reliability={comp_reliability:.3f}")
        report.append(f"  seasonal detail: lstm={lstm_s:.4f}(w={lstm_w/total_sw:.2f}), fourier={fourier_s:.4f}(w={fourier_w/total_sw:.2f})")

        return "\n".join(report)

    # ──────────────────────────────────────────────────────────────
    # LLM prediction (Formula 7 in the paper)
    # ──────────────────────────────────────────────────────────────
    def predict_batch_with_llm(self, station_id: int, samples: List[Dict]) -> List[float]:
        """
        Batch LLM prediction: one API call predicts multiple time steps.
        Each sample must contain: {timestamp, poi_info, historical_sequence, expert_predictions}
        """
        if not self.enable_llm:
            return [
                sum(self.current_weights.get(name, 0) * s['expert_predictions'].get(name, 0)
                    for name in self.expert_names)
                for s in samples
            ]

        samples_text = ""
        for idx, s in enumerate(samples):
            scene = self.extract_scene_features(s['timestamp'], s['poi_info'], s['historical_sequence'])
            reliance_report = self._build_reliance_report(station_id, scene, s['expert_predictions'])
            dt = pd.to_datetime(s['timestamp'])
            hist = s['historical_sequence']
            hist_mean = float(np.mean(hist)) if hist else 0
            hist_trend = float(hist[-1] - hist[0]) if len(hist) > 1 else 0

            active_spatial_names = [n for n in self.expert_names
                                  if n in ('spatial_geo', 'spatial_poi', 'spatial_similarity')]
            spatial_reliance = {
                name: float(np.mean(self.reliance_scores.get(name, {}).get(scene['scene_key'], [0.5])))
                for name in active_spatial_names
            }
            total_w = sum(spatial_reliance.values()) or 1.0
            spatial_anchor = sum(
                spatial_reliance[n] / total_w * s['expert_predictions'].get(n, 0)
                for n in active_spatial_names
            )

            samples_text += f"""
--- Sample {idx+1} ---
Time: {s['timestamp']} ({dt.hour:02d}:00, {'Weekend' if dt.weekday()>=5 else 'Weekday'})
History (last 12 steps): {', '.join([f'{x:.2f}' for x in hist[-12:]])}
Mean: {hist_mean:.2f}, Trend: {'Rising' if hist_trend > 0 else 'Falling' if hist_trend < 0 else 'Stable'}
spatial weighted anchor: {spatial_anchor:.4f} (reliability-weighted average of the 3 spatial models)
Expert predictions & reliability:
{reliance_report}
"""

        active_spatial = [n for n in self.expert_names if n in ('spatial_geo', 'spatial_poi', 'spatial_similarity')]
        spatial_desc_map = {
            'spatial_geo': 'Spatial-temporal graph model using geographic proximity between stations',
            'spatial_poi': 'Spatial-temporal graph model using POI-based functional similarity between stations',
            'spatial_similarity': 'Spatial-temporal graph model using historical traffic pattern similarity',
        }
        spatial_lines = "\n".join(f"- {n}: {spatial_desc_map[n]}" for n in active_spatial)
        n_total_models = len(self.expert_names)

        prompt = f"""You are an AI agent responsible for predicting mobile base station traffic. Your role is to intelligently combine the outputs of {n_total_models} specialist forecasting models by assessing their reliability and producing a final traffic prediction.

**Background:**
Mobile base station traffic reflects the number of active users in the area. It varies by time of day, day of week, and the surrounding land use (POI mix). Each specialist model captures a different aspect of traffic patterns.

**Station ID:** {station_id}
**Surrounding area (500m radius POI distribution):** {samples[0]['poi_info']}

**The {n_total_models} specialist models and what they capture:**

Group A — spatial models (PRIMARY, spatial-temporal deep learning, generally most accurate):
{spatial_lines}

Group B — Component models (SUPPLEMENTARY, decomposition-based, used for fine-tuning only):
- linear_trend: Linear extrapolation of the trend component (predicts direction of change)
- lstm_seasonal: LSTM model trained on the seasonal (periodic) component of traffic
- fourier_seasonal: Fourier-based model trained on the seasonal component of traffic
- residual: Persistence model for the residual component (assumes residual stays constant)

**Weighting rules:**
- By default, assign 70-85% of total weight to Group A (spatial models combined)
- Only increase Group B weight above 15-30% if Group B reliability scores are significantly higher than Group A
- The "spatial weighted anchor" shown for each sample is the reliability-weighted average of the active spatial model(s). Empirically, predictions that stray far from this anchor tend to introduce noise rather than signal — treat it as a strong prior and only adjust modestly based on component model evidence or clear historical trend signals

Each model provides a prediction value and a reliability score based on its past accuracy in the current time scenario. Higher reliability means the model has been more accurate in similar past situations.

**Your task for the sample below:**
1. Read the historical traffic sequence and identify the current trend (rising/falling/stable) and volatility
2. Consider the time context (rush hour patterns differ from late night; weekends differ from weekdays) and the POI mix (e.g. office-heavy areas peak at commute times)
3. Weight Group A based on reliability scores and context
4. Use Group B predictions only as a minor adjustment within the spatial range
5. If validation MAE is provided, use it as additional evidence of each model's recent accuracy

{f"{self.experience_summary}" + chr(10) if self.experience_summary else ""}{samples_text}

Output format (JSON only, no other text):
{{
    "predictions": [value1]
}}
"""
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an intelligent ensemble agent for urban mobile traffic forecasting. You combine outputs from multiple specialist models by reasoning about their reliability and the current context. Always output valid JSON only, with no extra text or markdown."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=max(1000, 300 + len(samples) * 80)
                )
                if response.usage:
                    self.total_prompt_tokens += response.usage.prompt_tokens
                    self.total_completion_tokens += response.usage.completion_tokens
                self.total_requests += 1
                content = response.choices[0].message.content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                start = content.find('{')
                depth, end = 0, -1
                for i, ch in enumerate(content[start:], start):
                    if ch == '{': depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                result = json.loads(content[start:end])
                preds = result.get("predictions", [])

                validated_preds = []
                for i, pred in enumerate(preds):
                    if i >= len(samples):
                        break
                    s = samples[i]
                    scene = self.extract_scene_features(s['timestamp'], s['poi_info'], s['historical_sequence'])
                    active_spatial_names = [n for n in self.expert_names
                                          if n in ('spatial_geo', 'spatial_poi', 'spatial_similarity')]
                    spatial_reliance = {
                        name: float(np.mean(self.reliance_scores.get(name, {}).get(scene['scene_key'], [0.5])))
                        for name in active_spatial_names
                    }
                    total_w = sum(spatial_reliance.values()) or 1.0
                    spatial_anchor = sum(
                        spatial_reliance[n] / total_w * s['expert_predictions'].get(n, 0)
                        for n in active_spatial_names
                    )
                    # To ensure prediction stability, we apply a post-processing step that constrains
                    # the final output within a physically plausible range derived from the most
                    # reliable spatial-temporal models.
                    lower_bound = spatial_anchor * 0.8
                    upper_bound = spatial_anchor * 1.2
                    if pred < lower_bound or pred > upper_bound:
                        pred = max(lower_bound, min(upper_bound, pred))
                    validated_preds.append(float(pred))

                if len(validated_preds) < len(samples):
                    for s in samples[len(validated_preds):]:
                        fallback = sum(self.current_weights.get(name, 0) * s['expert_predictions'].get(name, 0)
                                       for name in self.expert_names)
                        validated_preds.append(float(fallback))

                return validated_preds

            except Exception as e:
                if attempt < 2:
                    time.sleep(0.5)
                else:
                    print(f"  LLM batch prediction failed after 3 attempts: {e}, falling back to weighted average")
                    return [sum(self.current_weights.get(name, 0) * s['expert_predictions'].get(name, 0)
                                for name in self.expert_names) for s in samples]

    # ──────────────────────────────────────────────────────────────
    # Experience summary (used across epochs)
    # ──────────────────────────────────────────────────────────────
    def build_experience_summary(self, reserved_records: List[Dict]) -> str:
        if not reserved_records:
            return ""
        scene_best = defaultdict(list)
        for r in reserved_records:
            scene_best[r['scene_key']].append(r['best_expert'])

        lines = ["[Accumulated experience from training samples]"]
        for scene_key, experts in scene_best.items():
            counter = Counter(experts)
            top = counter.most_common(2)
            top_str = ", ".join([f"{e}({c}/{len(experts)})" for e, c in top])
            lines.append(f"- {scene_key}: most accurate expert(s) = {top_str}")

        all_best = [r['best_expert'] for r in reserved_records]
        global_counter = Counter(all_best)
        global_top = global_counter.most_common(3)
        global_str = ", ".join([f"{e}({c})" for e, c in global_top])
        lines.append(f"- Overall best experts across all scenarios: {global_str}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────
    # Performance history (used by create_context_prompt / query_llm)
    # ──────────────────────────────────────────────────────────────
    def generate_expert_performance_description(self, expert_errors: Dict,
                                                ensemble_mae: float,
                                                ensemble_rmse: float,
                                                weights_used: Dict,
                                                timestamp: str) -> str:
        dt = pd.to_datetime(timestamp)
        hour = dt.hour
        weekday = dt.day_name()
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

        description = f"\n[{timestamp} | {weekday} | {time_period}]\n"
        description += f"Ensemble Performance: MAE={ensemble_mae:.4f}, RMSE={ensemble_rmse:.4f}\n"

        spatial_experts = ['spatial_geo', 'spatial_poi', 'spatial_similarity']
        spatial_maes = [expert_errors[e]['mae'] for e in spatial_experts if e in expert_errors]
        if spatial_maes:
            avg_spatial_mae = np.mean(spatial_maes)
            best_spatial = min(spatial_experts, key=lambda e: expert_errors.get(e, {}).get('mae', float('inf')))
            worst_spatial = max(spatial_experts, key=lambda e: expert_errors.get(e, {}).get('mae', 0))
            if avg_spatial_mae < ensemble_mae * 1.05:
                spatial_assessment = "EXCELLENT - spatial models performed very well, close to or better than ensemble"
            elif avg_spatial_mae < ensemble_mae * 1.15:
                spatial_assessment = "GOOD - spatial models performed well with reasonable accuracy"
            elif avg_spatial_mae < ensemble_mae * 1.30:
                spatial_assessment = "MODERATE - spatial models had acceptable but not outstanding performance"
            else:
                spatial_assessment = "POOR - spatial models struggled in this scenario, consider increasing component model weight"
            description += f"\nspatial Group Assessment: {spatial_assessment}\n"
            description += f"  - Best spatial: {best_spatial} (MAE={expert_errors[best_spatial]['mae']:.4f})\n"
            description += f"  - Worst spatial: {worst_spatial} (MAE={expert_errors[worst_spatial]['mae']:.4f})\n"
            description += f"  - Average spatial MAE: {avg_spatial_mae:.4f}\n"

        component_experts = ['linear_trend', 'lstm_seasonal', 'fourier_seasonal', 'residual']
        component_present = [e for e in component_experts if e in expert_errors]
        if component_present:
            component_maes = [expert_errors[e]['mae'] for e in component_present]
            avg_component_mae = np.mean(component_maes)
            if avg_component_mae < avg_spatial_mae * 0.95:
                component_assessment = "OUTSTANDING - Component models outperformed spatial, increase their weight"
            elif avg_component_mae < avg_spatial_mae * 1.10:
                component_assessment = "COMPETITIVE - Component models matched spatial performance"
            elif avg_component_mae < avg_spatial_mae * 1.25:
                component_assessment = "SUPPLEMENTARY - Component models provide useful complementary information"
            else:
                component_assessment = "WEAK - Component models underperformed, keep weight low"
            description += f"\nComponent Group Assessment: {component_assessment}\n"
            description += f"  - Average Component MAE: {avg_component_mae:.4f}\n"

        spatial_total_weight = sum(weights_used.get(e, 0) for e in spatial_experts)
        component_total_weight = sum(weights_used.get(e, 0) for e in component_experts)
        description += f"\nWeights Used: spatial={spatial_total_weight:.2f} | Component={component_total_weight:.2f}\n"

        if "POOR" in spatial_assessment or "OUTSTANDING" in component_assessment:
            description += "Strategy Hint: Consider increasing component model weight for similar scenarios\n"
        elif "EXCELLENT" in spatial_assessment and "WEAK" in component_assessment:
            description += "Strategy Hint: spatial dominance confirmed, maintain high spatial weight\n"
        else:
            description += "Strategy Hint: Current balance seems appropriate, minor adjustments may help\n"
        return description

    def add_performance_record(self, expert_errors: Dict, ensemble_mae: float,
                               ensemble_rmse: float, weights_used: Dict, timestamp: str):
        description = self.generate_expert_performance_description(
            expert_errors, ensemble_mae, ensemble_rmse, weights_used, timestamp)
        self.performance_history.append({
            'timestamp': timestamp,
            'description': description,
            'ensemble_mae': ensemble_mae,
            'ensemble_rmse': ensemble_rmse
        })
        if len(self.performance_history) > self.max_history_len:
            self.performance_history = self.performance_history[-self.max_history_len:]

    def get_best_strategy_for_scene(self, scene_key: str) -> str:
        return self.scene_strategy_map.get(scene_key, "BASE_WEIGHTED")
