import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.window_size = seq_len + pred_len

    def __len__(self):
        return self.data.shape[0] - self.window_size + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.window_size]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.1, pred_len=1):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.input_dim = input_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, input_dim * pred_len)

    def forward(self, x):
        batch_size = x.size(0)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out.view(batch_size, self.pred_len, self.input_dim)


def train_model(model, train_loader, val_loader, device, epochs=100, lr=0.001, patience=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(x_batch), y_batch).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                model.load_state_dict(best_model_state)
                break

    return model


def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            predictions.append(model(x_batch).cpu().numpy())
            actuals.append(y_batch.numpy())
    return np.concatenate(predictions, axis=0), np.concatenate(actuals, axis=0)


def main():
    seq_len = 12
    pred_len = 1
    batch_size = 4
    hidden_dim = 128
    num_layers = 3
    dropout = 0.1
    epochs = 100
    lr = 0.0005
    window_size = 12
    n_his = seq_len

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv('/root/MoELLM/data/nanjing/traffic_seasonal.csv', header=None)
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

    train_loader = DataLoader(TimeSeriesDataset(train_data, seq_len, pred_len), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TimeSeriesDataset(val_data,   seq_len, pred_len), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TimeSeriesDataset(test_data,  seq_len, pred_len), batch_size=batch_size, shuffle=False)

    model = LSTMPredictor(
        input_dim=n_stations,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        pred_len=pred_len
    ).to(device)

    model = train_model(model, train_loader, val_loader, device, epochs=epochs, lr=lr, patience=10)

    test_pred, test_actual = evaluate_model(model, test_loader, device)

    model_path = 'LSTM_seasonal.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': n_stations,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'pred_len': pred_len
        },
        'seq_len': seq_len,
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
