import os
import sys
import numpy as np
import torch
import scipy.sparse as sp

diffstg_path = '/root/MoELLM/diffusion'
if diffstg_path not in sys.path:
    sys.path.insert(0, diffstg_path)
from script import utility
from model import models
from lstm_seasonal_predictor import LSTMPredictor as LSTMSeasonalPredictor
from linear_trend_predictor import LinearExtrapolationModel
from fourier_seasonal_predictor import FourierSeasonalPredictor

try:
    from algorithm.diffstg.model import DiffSTG
    from algorithm.diffstg.ugnet import UGnet
    DIFFSTG_AVAILABLE = True
except ImportError:
    DIFFSTG_AVAILABLE = False

class ExpertModelLoader:
    def __init__(self, device, dataset='shanghai'):
        self.device = device
        self.dataset = dataset
        self.n_vertex = 1042 if dataset == 'shanghai' else 1000
        self.n_his = 12
        self.n_pred = 1
        self.expert_models = {}

    def load_stgcn_model(self, model_path, adj_type='distance'):
        dataset_path = f'/root/MoELLM/data/{self.dataset}'
        if adj_type == 'distance':
            adj = sp.load_npz(os.path.join(dataset_path, 'adj_geo.npz'))
        elif adj_type == 'function':
            adj = sp.load_npz(os.path.join(dataset_path, 'adj_poi.npz'))
        elif adj_type == 'pattern':
            adj = sp.load_npz(os.path.join(dataset_path, 'adj_similarity.npz'))

        adj = adj.tocsc()
        gso = utility.calc_gso(adj, 'sym_norm_lap')
        gso = utility.calc_chebynet_gso(gso)
        gso = gso.toarray().astype(np.float32)
        gso_tensor = torch.from_numpy(gso).to(self.device)

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
        blocks = [[1], [64, 16, 64], [64, 16, 64], [128, 128], [1]]
        model = models.STGCNChebGraphConv(args, blocks, self.n_vertex).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def load_linear_trend_model(self, model_path):
        state = torch.load(model_path, map_location=self.device)
        model_config = state['model_config']
        model = LinearExtrapolationModel(
            n_stations=model_config['n_stations'],
            seq_len=model_config['seq_len'],
            pred_len=model_config['pred_len']
        )

        return model

    def load_lstm_seasonal_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint.get('model_config', {})
        input_dim = model_config.get('input_dim', self.n_vertex)
        hidden_dim = model_config.get('hidden_dim', 128)
        num_layers = model_config.get('num_layers', 3)
        dropout = model_config.get('dropout', 0.1)
        pred_len = model_config.get('pred_len', 1)
        model = LSTMSeasonalPredictor(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, pred_len=pred_len
        ).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        return model

    def load_fourier_seasonal_model(self, model_path):
        model = FourierSeasonalPredictor.load(model_path)

        return model

    def load_residual_model(self, model_path):
        if model_path is None or not os.path.exists(model_path):
            return 'persistence'
        model = torch.load(model_path, map_location=self.device, weights_only=False)
        model.eval()
        return model

    def load_all_models(self):
        dataset_path = f'/root/MoELLM/data/{self.dataset}'
        self.expert_models['stgcn_geo'] = self.load_stgcn_model(
            os.path.join(dataset_path, 'distance.pt'), 'distance')
        self.expert_models['stgcn_poi'] = self.load_stgcn_model(
            os.path.join(dataset_path, 'function.pt'), 'function')
        self.expert_models['stgcn_similarity'] = self.load_stgcn_model(
            os.path.join(dataset_path, 'pattern.pt'), 'pattern')
        self.expert_models['linear_trend'] = self.load_linear_trend_model(
            os.path.join(dataset_path, 'Linear_trend.pt'))
        self.expert_models['lstm_seasonal'] = self.load_lstm_seasonal_model(
            os.path.join(dataset_path, 'LSTM_seasonal.pt'))
        self.expert_models['fourier_seasonal'] = self.load_fourier_seasonal_model(
            os.path.join(dataset_path, 'Fourier_seasonal.pt'))
        self.expert_models['residual'] = self.load_residual_model(
            os.path.join(dataset_path, 'residual.pt'))
        return self.expert_models
