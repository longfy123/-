import numpy as np
import pandas as pd
import torch

class LinearExtrapolationModel:
    def __init__(self, n_stations, seq_len=12, pred_len=1):
        self.n_stations = n_stations
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model_type = "LinearExtrapolation"

    def predict(self, history):
        if len(history.shape) == 2:
            diff = history[-1, :] - history[-2, :]
            pred = history[-1, :] + diff
            return pred.reshape(1, -1)
        else:
            batch_size = history.shape[0]
            predictions = np.zeros((batch_size, self.pred_len, self.n_stations))
            for b in range(batch_size):
                diff = history[b, -1, :] - history[b, -2, :]
                predictions[b, 0, :] = history[b, -1, :] + diff
            return predictions

    def to_dict(self):
        return {
            'model_type': self.model_type,
            'n_stations': self.n_stations,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len
        }

    @classmethod
    def from_dict(cls, config):
        return cls(
            n_stations=config['n_stations'],
            seq_len=config['seq_len'],
            pred_len=config['pred_len']
        )


def evaluate_model(model, data, seq_len):
    n_timesteps, n_stations = data.shape
    n_samples = n_timesteps - seq_len
    predictions = []
    actuals = []
    for i in range(n_samples):
        history = data[i:i+seq_len, :]
        actual = data[i+seq_len:i+seq_len+1, :]
        pred = model.predict(history)
        predictions.append(pred)
        actuals.append(actual)
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    return predictions, actuals


def main():
    seq_len = 12
    pred_len = 1
    window_size = 12
    n_his = seq_len

    df = pd.read_csv('/root/MoELLM/data/nanjing/traffic_trend.csv', header=None)
    data = df.iloc[:, :].values.astype(np.float32)

    n_timesteps, n_stations = data.shape
    data_col = n_timesteps

    len_test     = 8  * window_size + n_his
    len_val      = 4  * window_size + n_his
    len_reserved = 16 * window_size + n_his
    len_train    = data_col - len_reserved - len_val - len_test
    len_train_full = len_train + len_reserved

    train_data = data[:len_train]
    val_data   = data[len_train_full:len_train_full + len_val]
    test_data  = data[len_train_full + len_val:]

    model = LinearExtrapolationModel(n_stations=n_stations, seq_len=seq_len, pred_len=pred_len)

    val_pred, val_actual = evaluate_model(model, val_data, seq_len)
    test_pred, test_actual = evaluate_model(model, test_data, seq_len)

    model_path = 'Linear_trend.pt'
    torch.save({
        'model_config': model.to_dict(),
        'seq_len': seq_len,
        'pred_len': pred_len,
        'n_stations': n_stations,
        'data_info': {
            'train_size': int(len_train),
            'reserved_size': int(len_reserved),
            'val_size': int(len_val),
            'test_size': int(len_test),
            'n_timesteps': int(n_timesteps),
            'n_stations': int(n_stations)
        }
    }, model_path)

if __name__ == '__main__':
    main()
