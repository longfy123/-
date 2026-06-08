import numpy as np
import torch
from typing import Dict


class EnsemblePredictor:
    """Hierarchical ensemble predictor combining 7 expert models."""

    def __init__(self, expert_models: Dict, device, dataset='shanghai'):
        self.expert_models = expert_models
        self.device = device
        self.dataset = dataset
        self.n_vertex = 1042 if dataset == 'shanghai' else 1000
        self.n_his = 12

    # ──────────────────────────────────────────────────────────────
    # Per-model prediction methods
    # ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_spatial(self, model, x):
        x_input = x.unsqueeze(1)          # [B, 1, n_his, V]
        pred = model(x_input)             # [B, 1, 1, V]
        return pred.squeeze(1).squeeze(1) # [B, V]

    @torch.no_grad()
    def predict_lstm(self, model, x):
        pred = model(x)                   # [B, pred_len, V]
        if len(pred.shape) == 3 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        return pred                       # [B, V]

    def predict_linear(self, model, x):
        x_np = x.cpu().numpy()            # [B, n_his, V]
        pred_np = model.predict(x_np)     # [B, 1, V]
        pred_np = pred_np.squeeze(1)      # [B, V]
        return torch.from_numpy(pred_np).to(x.device).float()

    def predict_fourier(self, model, x):
        x_np = x.cpu().numpy()            # [B, n_his, V]
        predictions_list = []
        for b in range(x_np.shape[0]):
            pred_single = model.predict_step(x_np[b], n_pred=1)  # [1, V]
            predictions_list.append(pred_single[0])
        pred_np = np.stack(predictions_list, axis=0)              # [B, V]
        return torch.from_numpy(pred_np).to(x.device).float()

    @torch.no_grad()
    def predict_residual(self, model, x):
        model_type = type(model).__name__

        if model == 'persistence' or model_type == 'str':
            return x[:, -1, :]            # persistence: last observed value

        if 'DiffSTG' in model_type or hasattr(model, 'epsilon_theta'):
            batch_size, n_his, n_vertex = x.shape
            n_pred = 1
            x_history = x.unsqueeze(-1)                                          # [B, n_his, V, 1]
            x_future_zeros = torch.zeros(batch_size, n_pred, n_vertex, 1, device=x.device)
            x_with_future = torch.cat([x_history, x_future_zeros], dim=1)        # [B, n_his+1, V, 1]
            x_diff = x_with_future.permute(0, 3, 2, 1)                           # [B, 1, V, n_his+1]
            total_steps = n_his + n_pred
            pos_w = torch.zeros((batch_size, total_steps), dtype=torch.long, device=x.device)
            pos_d = torch.arange(total_steps, device=x.device).unsqueeze(0).repeat(batch_size, 1) % 48
            try:
                pred_full = model((x_diff, pos_w, pos_d), n_samples=1)           # [B, n_samples, F, V, T]
                if len(pred_full.shape) == 5:
                    pred = pred_full[:, 0, 0, :, -1]
                else:
                    pred = pred_full[:, 0, :, -1]
            except Exception as e:
                print(f"    Warning: DiffSTG prediction failed ({e}), using zero prediction")
                pred = torch.zeros(batch_size, n_vertex, device=x.device)
            return pred

        return x[:, -1, :]  # fallback

    # ──────────────────────────────────────────────────────────────
    # Collect all expert predictions
    # ──────────────────────────────────────────────────────────────
    def get_all_predictions(self, x, x_trend=None, x_seasonal=None,
                            x_residual=None, station_id=0):
        """
        Args:
            x:          [B, n_his, V] full sequence (for spatial)
            x_trend:    [B, n_his, V] trend component
            x_seasonal: [B, n_his, V] seasonal component
            x_residual: [B, n_his, V] residual component
            station_id: target station index (None = keep all stations)
        Returns:
            dict of {expert_name: tensor [B, V] or [B, 1]}
        """
        predictions = {}

        predictions['spatial_geo'] = self.predict_spatial(self.expert_models['spatial_geo'], x)
        predictions['spatial_poi'] = self.predict_spatial(self.expert_models['spatial_poi'], x)
        predictions['spatial_similarity'] = self.predict_spatial(self.expert_models['spatial_similarity'], x)

        x_trend_input = x_trend if x_trend is not None else x
        predictions['linear_trend'] = self.predict_linear(self.expert_models['linear_trend'], x_trend_input)

        x_seasonal_input = x_seasonal if x_seasonal is not None else x
        predictions['lstm_seasonal'] = self.predict_lstm(self.expert_models['lstm_seasonal'], x_seasonal_input)
        predictions['fourier_seasonal'] = self.predict_fourier(self.expert_models['fourier_seasonal'], x_seasonal_input)

        x_residual_input = x_residual if x_residual is not None else x
        predictions['residual'] = self.predict_residual(self.expert_models['residual'], x_residual_input)

        if station_id is not None:
            for key in predictions:
                if predictions[key].shape[-1] > 1:
                    predictions[key] = predictions[key][:, station_id:station_id+1]

        return predictions

    # ──────────────────────────────────────────────────────────────
    # Hierarchical ensemble
    # ──────────────────────────────────────────────────────────────
    def ensemble_predict(self, predictions: Dict, weights: Dict, station_id=None):
        """
        Hierarchical combination:
          - Group A: spatial models → weighted average
          - Group B: trend + weighted_avg(lstm_seasonal, fourier_seasonal) + residual → sum
          - Final: weighted combination of Group A and Group B
        """
        spatial_names = ['spatial_geo', 'spatial_poi', 'spatial_similarity']

        # Group A
        spatial_preds, spatial_weights = [], []
        for name in spatial_names:
            if name in predictions:
                spatial_preds.append(predictions[name])
                spatial_weights.append(weights.get(name, 1.0 / 3))
        spatial_weight_sum = sum(spatial_weights) or 1.0
        spatial_weights = [w / spatial_weight_sum for w in spatial_weights]
        spatial_ensemble = sum(w * p for w, p in zip(spatial_weights, spatial_preds)) if spatial_preds else None

        # Group B — seasonal weighted average
        lstm_w = weights.get('lstm_seasonal', 0)
        fourier_w = weights.get('fourier_seasonal', 0)
        seasonal_total = lstm_w + fourier_w
        seasonal_pred = None
        if seasonal_total > 0:
            if 'lstm_seasonal' in predictions and 'fourier_seasonal' in predictions:
                seasonal_pred = (lstm_w * predictions['lstm_seasonal'] +
                                 fourier_w * predictions['fourier_seasonal']) / seasonal_total
            elif 'lstm_seasonal' in predictions:
                seasonal_pred = predictions['lstm_seasonal']
            elif 'fourier_seasonal' in predictions:
                seasonal_pred = predictions['fourier_seasonal']

        component_pred = predictions.get('linear_trend')
        if seasonal_pred is not None:
            component_pred = seasonal_pred if component_pred is None else component_pred + seasonal_pred
        if 'residual' in predictions:
            component_pred = predictions['residual'] if component_pred is None else component_pred + predictions['residual']

        # Combine groups
        if spatial_ensemble is not None and component_pred is not None:
            spatial_vals = [weights.get(n, 0) for n in spatial_names if n in weights]
            seasonal_rep = (weights.get('lstm_seasonal', 0) + weights.get('fourier_seasonal', 0)) / 2
            comp_vals = [weights.get('linear_trend', 0), seasonal_rep, weights.get('residual', 0)]
            spatial_total_weight = sum(spatial_vals) / len(spatial_vals) if spatial_vals else 0
            component_total_weight = sum(comp_vals) / len(comp_vals)
            total = spatial_total_weight + component_total_weight
            if total > 0:
                sf = spatial_total_weight / total
                cf = component_total_weight / total
            else:
                sf, cf = 0.9, 0.1
            return sf * spatial_ensemble + cf * component_pred
        elif spatial_ensemble is not None:
            return spatial_ensemble
        else:
            return component_pred
