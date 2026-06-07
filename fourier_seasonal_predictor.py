import numpy as np
import pandas as pd
import torch
from scipy import fft


class FourierSeasonalPredictor:

    def __init__(self, n_components=10):
        self.n_components = n_components

    def save(self, path):
        torch.save({
            'n_components': self.n_components,
            'model_type': 'FourierSeasonalPredictor'
        }, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path)
        return cls(n_components=state['n_components'])

    def fit(self, data):
        return self

    def predict_step(self, x, n_pred=1):
        seq_len = x.shape[0]
        n_stations = x.shape[1]
        predictions = np.zeros((n_pred, n_stations))

        for station_idx in range(n_stations):
            station_data = x[:, station_idx]
            fft_vals = fft.fft(station_data)

            k = self.n_components
            fft_filtered = np.zeros_like(fft_vals, dtype=complex)
            fft_filtered[:k] = fft_vals[:k]
            fft_filtered[-k:] = fft_vals[-k:]

            extended_len = seq_len + n_pred
            fft_extended = np.zeros(extended_len, dtype=complex)

            for i in range(k):
                if i < extended_len:
                    fft_extended[i] = fft_filtered[i] * extended_len / seq_len
            for i in range(k):
                if i > 0:
                    fft_extended[-i] = fft_filtered[-i] * extended_len / seq_len

            reconstructed = fft.ifft(fft_extended).real
            predictions[:, station_idx] = reconstructed[seq_len:seq_len + n_pred]

        return predictions

    def predict_dataset(self, data, seq_len, pred_len):
        n_timesteps = data.shape[0]
        window_size = seq_len + pred_len
        n_samples = n_timesteps - window_size + 1

        predictions_list = []
        actuals_list = []

        for idx in range(n_samples):
            x = data[idx:idx + seq_len]
            y = data[idx + seq_len:idx + window_size]
            predictions_list.append(self.predict_step(x, n_pred=pred_len))
            actuals_list.append(y)

        return np.array(predictions_list), np.array(actuals_list)


def main():
    seq_len = 12
    pred_len = 1
    n_components = 10
    window_size = 12
    n_his = seq_len

    df = pd.read_csv('/root/MoELLM/data/nanjing/traffic_seasonal.csv', header=None)
    data = df.iloc[:, :].values.astype(np.float32)

    n_timesteps, n_stations = data.shape
    data_col = n_timesteps

    len_test     = 8  * window_size + n_his
    len_val      = 4  * window_size + n_his
    len_reserved = 16 * window_size + n_his
    len_train    = data_col - len_reserved - len_val - len_test
    len_train_full = len_train + len_reserved

    val_data  = data[len_train_full:len_train_full + len_val]
    test_data = data[len_train_full + len_val:]

    predictor = FourierSeasonalPredictor(n_components=n_components)

    val_pred, val_actual   = predictor.predict_dataset(val_data,  seq_len, pred_len)
    test_pred, test_actual = predictor.predict_dataset(test_data, seq_len, pred_len)

    model_path = 'Fourier_seasonal_nanjing.pt'
    predictor.save(model_path)


if __name__ == '__main__':
    main()
