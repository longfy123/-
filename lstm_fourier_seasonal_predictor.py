import logging
import os
import gc
import argparse
import math
import random
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy import fft

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from script import earlystopping


def set_env(seed):
    """设置随机种子以保证实验可重复性"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def extract_seasonal_fourier(data, n_components=10, period=336):
    """
    使用傅里叶变换提取季节性分量
    
    Args:
        data: 时间序列数据 [time_steps, n_features]
        n_components: 保留的主要频率成分数量
        period: 主要周期（用于识别关键频率）
    
    Returns:
        seasonal: 季节性分量
        frequencies: 主要频率
        amplitudes: 对应振幅
    """
    n_samples, n_features = data.shape
    seasonal = np.zeros_like(data)
    
    print(f"\n使用傅里叶变换提取季节性分量...")
    print(f"  数据长度: {n_samples}")
    print(f"  主要周期: {period} (一周)")
    print(f"  保留频率成分数: {n_components}")
    
    # 对每个特征单独进行傅里叶分析
    all_frequencies = []
    all_amplitudes = []
    
    for i in range(n_features):
        series = data[:, i]
        
        # 1. 去除趋势（简单移动平均）
        window_size = min(period, len(series) // 2)
        if window_size % 2 == 0:
            window_size += 1
        kernel = np.ones(window_size) / window_size
        trend = np.convolve(series, kernel, mode='same')
        detrended = series - trend
        
        # 2. 傅里叶变换
        fft_values = fft.fft(detrended)
        fft_freq = fft.fftfreq(len(detrended))
        
        # 3. 只保留正频率部分
        positive_freq_idx = fft_freq > 0
        fft_values_pos = fft_values[positive_freq_idx]
        fft_freq_pos = fft_freq[positive_freq_idx]
        
        # 4. 计算功率谱（振幅）
        power = np.abs(fft_values_pos)
        
        # 5. 找出最强的n_components个频率
        top_indices = np.argsort(power)[-n_components:][::-1]
        top_frequencies = fft_freq_pos[top_indices]
        top_amplitudes = power[top_indices]
        
        all_frequencies.append(top_frequencies)
        all_amplitudes.append(top_amplitudes)
        
        # 6. 重构季节性信号（只使用主要频率成分）
        seasonal_fft = np.zeros_like(fft_values)
        for idx in top_indices:
            # 保留正频率和对应的负频率
            seasonal_fft[positive_freq_idx][idx] = fft_values[positive_freq_idx][idx]
            # 找到对应的负频率索引
            neg_idx = np.where(fft_freq == -fft_freq_pos[idx])[0]
            if len(neg_idx) > 0:
                seasonal_fft[neg_idx[0]] = fft_values[neg_idx[0]]
        
        # 7. 逆变换得到季节性分量
        seasonal[:, i] = fft.ifft(seasonal_fft).real
    
    # 统计主要周期
    avg_frequencies = np.mean(all_frequencies, axis=0)
    avg_periods = 1 / avg_frequencies
    
    print(f"\n识别到的主要周期（按重要性排序）:")
    for j, (freq, period_len) in enumerate(zip(avg_frequencies[:5], avg_periods[:5])):
        print(f"  {j+1}. 周期={period_len:.1f}个时间步 (频率={freq:.6f})")
    
    return seasonal, all_frequencies, all_amplitudes


class LSTMPredictor(nn.Module):
    """LSTM时间序列预测模型（结合移动平均特征）"""
    def __init__(self, input_size, output_size, seq_len, hidden_size=64, num_layers=2, dropout=0.1):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.output_size = output_size
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        prediction = self.fc(last_output)
        return prediction


def get_parameters():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(description='LSTM with Fourier Seasonal Component Prediction')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='shanghai', choices=['shanghai', 'nanjing'])
    parser.add_argument('--n_his', type=int, default=12, help='历史时间步长')
    parser.add_argument('--n_pred', type=int, default=1, help='预测时间步长')
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=4, help='LSTM层数')
    parser.add_argument('--droprate', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0001, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1000, help='epochs, default as 1000')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
    parser.add_argument('--ma_window', type=int, default=25, help='移动平均窗口大小')
    parser.add_argument('--n_fourier_components', type=int, default=5, help='傅里叶分量数')
    parser.add_argument('--seasonal_period', type=int, default=168, help='季节性周期（1天=48个30分钟时间步）')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    set_env(args.seed)

    if args.enable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        gc.collect()
    
    return args, device


def load_time_series_data(dataset_name, len_train_full, len_train, len_val):
    """加载时间序列数据"""
    dataset_path = '/root/stgcn原版/data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    
    # 修改：添加 header=None
    vel_df = pd.read_csv(os.path.join(dataset_path, 'vel_with_timestamp.csv'), header=None)
    
    # 判断第一列是否是时间戳（更智能的判断）
    first_col = vel_df.iloc[:, 0]
    has_timestamp = False
    
    # 检查第一列是否可能是时间戳
    if pd.api.types.is_string_dtype(first_col):
        # 如果是字符串类型，检查是否包含常见时间格式
        sample = str(first_col.iloc[0]) if len(first_col) > 0 else ""
        if ':' in sample or '-' in sample or '/' in sample or sample.isdigit() == False:
            has_timestamp = True
    elif pd.api.types.is_datetime64_any_dtype(first_col):
        has_timestamp = True
    
    if has_timestamp and vel_df.shape[1] > 1:
        # 有时间戳列
        timestamps = first_col.values
        vel = vel_df.iloc[:, 1:].values
        print(f"检测到时间戳列，数据形状: {vel.shape}")
    else:
        # 没有时间戳列
        timestamps = None
        vel = vel_df.values
        print(f"未检测到时间戳列，数据形状: {vel.shape}")
    
    # 数据集划分
    train = vel[:len_train]
    reserved = vel[len_train:len_train_full]
    val = vel[len_train_full:len_train_full + len_val]
    test = vel[len_train_full + len_val:]
    
    print(f"数据划分完成: 训练集={len(train)}, 保留集={len(reserved)}, 验证集={len(val)}, 测试集={len(test)}")
    
    return train, reserved, val, test, timestamps


def calculate_moving_average(data, window_size=5):
    """计算移动平均特征"""
    n_features = data.shape[1]
    ma_data = np.zeros_like(data)
    
    for i in range(len(data)):
        if i < window_size - 1:
            ma_data[i] = np.mean(data[:i+1], axis=0)
        else:
            ma_data[i] = np.mean(data[i-window_size+1:i+1], axis=0)
    
    enhanced_data = np.concatenate([data, ma_data], axis=1)
    return enhanced_data


def data_transform(data, n_his, n_pred, use_ma=False, ma_window=5):
    """将数据转换为时间序列格式"""
    if use_ma:
        data = calculate_moving_average(data, ma_window)
    
    len_record = len(data)
    num = len_record - n_his - n_pred + 1
    
    x = np.zeros([num, n_his, data.shape[1]])
    y = np.zeros([num, data.shape[1] // 2 if use_ma else data.shape[1]])
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :] = data[head:tail]
        if use_ma:
            n_original_features = data.shape[1] // 2
            y[i] = data[tail + n_pred - 1, :n_original_features]
        else:
            y[i] = data[tail + n_pred - 1]
    
    return x, y


def data_preparation(args, device):
    """数据准备和预处理（使用傅里叶变换提取季节性分量）"""
    dataset_path = '/root/stgcn原版/data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    
    # 修改：添加 header=None
    vel_df = pd.read_csv(os.path.join(dataset_path, 'vel_with_timestamp.csv'), header=None)
    
    # 更智能地判断是否有时间戳列
    first_col = vel_df.iloc[:, 0]
    has_timestamp = False
    
    if pd.api.types.is_string_dtype(first_col):
        sample = str(first_col.iloc[0]) if len(first_col) > 0 else ""
        if ':' in sample or '-' in sample or '/' in sample or sample.isdigit() == False:
            has_timestamp = True
    elif pd.api.types.is_datetime64_any_dtype(first_col):
        has_timestamp = True
    
    if has_timestamp and vel_df.shape[1] > 1:
        # 包含时间戳列
        data_col = len(vel_df)
        n_features = vel_df.shape[1] - 1
        print(f"检测到时间戳列，数据形状: {vel_df.shape}, 特征数: {n_features}")
    else:
        # 没有时间戳列
        data_col = len(vel_df)
        n_features = vel_df.shape[1]
        print(f"未检测到时间戳列，数据形状: {vel_df.shape}, 特征数: {n_features}")
    
    print(f"总时间步数: {data_col}")
    
    # 数据集划分比例
    val_and_test_rate = 0.15
    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train_full = int(data_col - len_val - len_test)
    len_reserved = int(math.floor(len_train_full * 0.20))
    len_train = len_train_full - len_reserved
    
    print(f"数据集划分: 总样本数={data_col}, 训练集={len_train}, 保留集={len_reserved}, 验证集={len_val}, 测试集={len_test}")
    print(f"特征数量: {n_features}")
    
    # 计算周期描述
    hours = args.seasonal_period * 0.5
    days = hours / 24
    print(f"季节性周期: {args.seasonal_period}个时间步 ({hours:.1f}小时 / {days:.2f}天)")
    print(f"傅里叶分量数: {args.n_fourier_components}")
    
    # 加载数据（使用修改后的函数）
    train, reserved, val, test, timestamps = load_time_series_data(
        args.dataset, len_train_full, len_train, len_val
    )
    
    # 验证数据维度
    print(f"数据加载验证: train.shape={train.shape}, val.shape={val.shape}, test.shape={test.shape}")
    
    # Z-score标准化
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)
    
    # ========== 使用傅里叶变换提取季节性分量 ==========
    print(f"\n{'='*60}")
    print(f"使用傅里叶变换提取季节性分量")
    print(f"{'='*60}")
    
    train_seasonal, train_freq, train_amp = extract_seasonal_fourier(
        train, n_components=args.n_fourier_components, period=args.seasonal_period
    )
    val_seasonal, _, _ = extract_seasonal_fourier(
        val, n_components=args.n_fourier_components, period=args.seasonal_period
    )
    test_seasonal, _, _ = extract_seasonal_fourier(
        test, n_components=args.n_fourier_components, period=args.seasonal_period
    )
    
    print(f"\n傅里叶季节性提取完成！")
    print(f"  训练集季节性形状: {train_seasonal.shape}")
    print(f"  验证集季节性形状: {val_seasonal.shape}")
    print(f"  测试集季节性形状: {test_seasonal.shape}")
    
    print(f"\n季节性分量统计:")
    print(f"  范围: [{train_seasonal.min():.4f}, {train_seasonal.max():.4f}]")
    print(f"  均值: {train_seasonal.mean():.4f}")
    print(f"  标准差: {train_seasonal.std():.4f}")
    
    # ========== 添加移动平均特征 ==========
    print(f"\n正在添加移动平均特征，窗口大小: {args.ma_window}")
    
    x_train, y_train = data_transform(train_seasonal, args.n_his, args.n_pred, use_ma=True, ma_window=args.ma_window)
    x_val, y_val = data_transform(val_seasonal, args.n_his, args.n_pred, use_ma=True, ma_window=args.ma_window)
    x_test, y_test = data_transform(test_seasonal, args.n_his, args.n_pred, use_ma=True, ma_window=args.ma_window)
    
    print(f"\n添加MA特征后：")
    print(f"  输入特征数: {x_train.shape[2]} (原始: {n_features}, MA窗口: {args.ma_window})")
    print(f"  输出特征数: {y_train.shape[1]}")
    
    # 转换为Tensor
    x_train = torch.FloatTensor(x_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    x_val = torch.FloatTensor(x_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    x_test = torch.FloatTensor(x_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    # 创建数据加载器
    train_data = utils.data.TensorDataset(x_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_data = utils.data.TensorDataset(x_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    
    return n_features, zscore, train_iter, val_iter, test_iter


def prepare_model(args, n_features, device):
    """准备模型、损失函数、优化器等"""
    input_features = n_features * 2  # 原始 + MA
    
    model = LSTMPredictor(
        input_size=input_features,
        output_size=n_features,
        seq_len=args.n_his,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.droprate
    ).to(device)
    
    loss_fn = nn.MSELoss()
    
    es = earlystopping.EarlyStopping(
        delta=0.0,
        patience=args.patience,
        path=f'LSTM_Fourier_seasonal_{args.dataset}.pt',
        verbose=True
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay_rate
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma
    )
    
    return model, loss_fn, es, optimizer, scheduler


def train_epoch(model, loss_fn, optimizer, train_iter):
    """训练一个epoch"""
    model.train()
    train_loss = 0.0
    
    for x, y in train_iter:
        optimizer.zero_grad()
        y_pred = model(x)
        l = loss_fn(y_pred, y)
        l.backward()
        optimizer.step()
        train_loss += l.item() * x.size(0)
    
    train_loss = train_loss / len(train_iter.dataset)
    return train_loss


def val_epoch(model, loss_fn, val_iter):
    """验证一个epoch"""
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for x, y in val_iter:
            y_pred = model(x)
            l = loss_fn(y_pred, y)
            val_loss += l.item() * x.size(0)
    
    val_loss = val_loss / len(val_iter.dataset)
    return val_loss


def train(args, model, loss_fn, optimizer, scheduler, es, train_iter, val_iter):
    """训练模型"""
    print("\n开始训练...")
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loss_fn, optimizer, train_iter)
        val_loss = val_epoch(model, loss_fn, val_iter)
        scheduler.step()
        
        if torch.cuda.is_available():
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000
        else:
            gpu_mem_alloc = 0
        
        print(f'Epoch: {epoch:03d} | Lr: {optimizer.param_groups[0]["lr"]:.10f} | '
              f'Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f} | '
              f'GPU occupy: {gpu_mem_alloc:.2f} MiB')
        
        es(val_loss, model)
        if es.early_stop:
            print('Early stopping')
            break
    
    model.load_state_dict(torch.load(es.path))


def test(args, model, loss_fn, test_iter, zscore):
    """测试模型"""
    model.eval()
    
    y_true_list = []
    y_pred_list = []
    mae_list = []
    mse_list = []
    
    with torch.no_grad():
        for x, y in test_iter:
            y_pred = model(x)
            y_np = y.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            
            mae = np.abs(y_np - y_pred_np)
            mse = (y_np - y_pred_np) ** 2
            
            y_true_list.append(y_np)
            y_pred_list.append(y_pred_np)
            mae_list.append(mae)
            mse_list.append(mse)
    
    mae_array = np.concatenate(mae_list, axis=0)
    mse_array = np.concatenate(mse_list, axis=0)
    y_true_array = np.concatenate(y_true_list, axis=0)
    y_pred_array = np.concatenate(y_pred_list, axis=0)
    
    MAE = np.mean(mae_array)
    RMSE = np.sqrt(np.mean(mse_array))
    
    # 使用标准差归一化
    y_std = np.std(y_true_array)
    NMAE = MAE / y_std * 100
    NRMSE = RMSE / y_std * 100
    
    # 计算R²
    ss_res = np.sum((y_true_array - y_pred_array) ** 2)
    ss_tot = np.sum((y_true_array - np.mean(y_true_array)) ** 2)
    R2 = 1 - (ss_res / ss_tot)
    
    test_MSE = np.mean(mse_array)
    
    print('\n' + '='*60)
    print(f'傅里叶季节性预测测试结果 - Dataset: {args.dataset}')
    print('='*60)
    print(f'Test MSE:  {test_MSE:.6f}')
    print(f'MAE:       {MAE:.6f}')
    print(f'RMSE:      {RMSE:.6f}')
    print(f'NMAE:      {NMAE:.2f}%')
    print(f'NRMSE:     {NRMSE:.2f}%')
    print(f'R²:        {R2:.6f}')
    print(f'{"="*60}\n')
    
    results_file = f'LSTM_Fourier_seasonal_{args.dataset}_results.txt'
    
    # 计算实际时间
    hours = args.seasonal_period * 0.5
    days = hours / 24
    period_desc = f"{args.seasonal_period} time steps ({hours:.1f} hours / {days:.2f} days)"
    
    with open(results_file, 'w') as f:
        f.write('LSTM + Fourier Seasonal Component Prediction Results\n')
        f.write(f'Dataset: {args.dataset}\n')
        f.write(f'Fourier Components: {args.n_fourier_components}\n')
        f.write(f'Seasonal Period: {period_desc}\n')
        f.write('='*60 + '\n')
        f.write(f'Test MSE:  {test_MSE:.6f}\n')
        f.write(f'MAE:       {MAE:.6f}\n')
        f.write(f'RMSE:      {RMSE:.6f}\n')
        f.write(f'NMAE:      {NMAE:.2f}%\n')
        f.write(f'NRMSE:     {NRMSE:.2f}%\n')
        f.write(f'R²:        {R2:.6f}\n')
    
    print(f'结果已保存到 {results_file}')
    
    return MAE, RMSE, NMAE, NRMSE, R2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    print("="*60)
    print("LSTM + 傅里叶变换季节性预测")
    print("="*60)
    print("说明：使用傅里叶变换提取时间序列的主要周期性成分")
    print("      然后使用LSTM预测傅里叶季节性分量")
    print("="*60)
    
    args, device = get_parameters()
    
    print("\n准备数据...")
    n_features, zscore, train_iter, val_iter, test_iter = data_preparation(args, device)
    
    print("\n准备模型...")
    model, loss_fn, es, optimizer, scheduler = prepare_model(args, n_features, device)
    
    print(f"\n模型结构:")
    print(model)
    print(f"\n模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n开始训练LSTM傅里叶季节性预测模型...")
    train(args, model, loss_fn, optimizer, scheduler, es, train_iter, val_iter)
    
    print("\n测试傅里叶季节性预测模型...")
    test(args, model, loss_fn, test_iter, zscore)
    
    print("\n训练完成!")
    print("模型文件: LSTM_Fourier_seasonal_{}.pt".format(args.dataset))
    print("结果文件: LSTM_Fourier_seasonal_{}_results.txt".format(args.dataset))
