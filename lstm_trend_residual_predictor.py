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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils

from script import earlystopping


def extract_trend_component(data, window_size=48):
    """
    从时间序列中提取趋势分量
    
    Args:
        data: 时间序列数据 [time_steps, n_features]
        window_size: 移动平均窗口大小（48 = 1天，30分钟间隔）
    
    Returns:
        trend: 趋势分量 [time_steps, n_features]
    """
    n_samples, n_features = data.shape
    trend = np.zeros_like(data)
    
    # 对每个特征单独提取趋势
    for i in range(n_features):
        series = data[:, i]
        
        # 使用移动平均提取趋势
        window = min(window_size, len(series) // 2)
        if window % 2 == 0:
            window += 1
        
        # 使用卷积实现移动平均
        kernel = np.ones(window) / window
        trend_i = np.convolve(series, kernel, mode='same')
        
        # 处理边界
        half_window = window // 2
        for j in range(half_window):
            trend_i[j] = np.mean(series[:j+half_window+1])
            trend_i[-(j+1)] = np.mean(series[-(j+half_window+1):])
        
        trend[:, i] = trend_i
    
    return trend


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


class LSTMResidualPredictor(nn.Module):
    """LSTM残差预测模型（用于预测持久化基线的修正量）"""
    def __init__(self, input_size, output_size, seq_len, hidden_size=64, num_layers=2, dropout=0.1):
        super(LSTMResidualPredictor, self).__init__()
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
        
        # 全连接层 - 输出残差修正量
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # LSTM输出
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        last_output = self.dropout(last_output)
        # 预测残差修正量
        residual = self.fc(last_output)  # (batch_size, output_size)
        return residual


def get_parameters():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(description='LSTM Residual Correction for Time Series Trend Prediction')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='shanghai', choices=['shanghai', 'nanjing'])
    parser.add_argument('--n_his', type=int, default=12, help='历史时间步长')
    parser.add_argument('--n_pred', type=int, default=1, help='预测时间步长')
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=3, help='LSTM层数')
    parser.add_argument('--droprate', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0001, help='weight decay (L2 penalty)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000, help='epochs, default as 1000')
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
    parser.add_argument('--ma_window', type=int, default=5, help='移动平均窗口大小')
    parser.add_argument('--trend_window', type=int, default=48, help='趋势提取窗口（1天=48个30分钟）')
    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # 设置随机种子以保证实验可重复性
    set_env(args.seed)

    # 设置设备 (CUDA or CPU)
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
    
    # 读取带时间戳的数据
    vel_df = pd.read_csv(os.path.join(dataset_path, 'vel_with_timestamp.csv'))
    
    # 提取时间戳和数据
    timestamps = vel_df.iloc[:, 0].values if vel_df.shape[1] > 1 else None
    vel = vel_df.iloc[:, 1:].values if vel_df.shape[1] > 1 else vel_df.values
    
    # 数据集划分
    train = vel[:len_train]
    reserved = vel[len_train:len_train_full]
    val = vel[len_train_full:len_train_full + len_val]
    test = vel[len_train_full + len_val:]
    
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


def data_transform_with_baseline(data, n_his, n_pred, use_ma=False, ma_window=5):
    """
    将数据转换为时间序列格式，同时计算持久化基线
    
    Returns:
        x: 输入序列
        y_true: 真实值
        y_baseline: 持久化基线预测（用t-1预测t）
        residual: 需要学习的残差（y_true - y_baseline）
    """
    # 如果使用移动平均，添加MA特征
    if use_ma:
        data_enhanced = calculate_moving_average(data, ma_window)
    else:
        data_enhanced = data
    
    len_record = len(data)
    num = len_record - n_his - n_pred + 1
    
    x = np.zeros([num, n_his, data_enhanced.shape[1]])
    y_true = np.zeros([num, data.shape[1]])  # 原始特征维度
    y_baseline = np.zeros([num, data.shape[1]])
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :] = data_enhanced[head:tail]
        
        # 真实值
        y_true[i] = data[tail + n_pred - 1]
        
        # 持久化基线：使用历史序列的最后一个时刻作为预测
        y_baseline[i] = data[tail - 1]
    
    # 计算残差（真实值 - 基线预测）
    residual = y_true - y_baseline
    
    return x, y_true, y_baseline, residual


def data_preparation(args, device):
    """数据准备和预处理（提取趋势分量）"""
    # 获取数据集大小
    dataset_path = '/root/stgcn原版/data'
    dataset_path = os.path.join(dataset_path, args.dataset)
    vel_df = pd.read_csv(os.path.join(dataset_path, 'vel_with_timestamp.csv'))
    
    # 确定是否包含时间戳列
    if vel_df.shape[1] > 1 and pd.api.types.is_string_dtype(vel_df.iloc[:, 0]):
        data_col = len(vel_df)
        n_features = vel_df.shape[1] - 1
    else:
        data_col = len(vel_df)
        n_features = vel_df.shape[1]
    
    # 数据集划分比例
    val_and_test_rate = 0.15
    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train_full = int(data_col - len_val - len_test)
    len_reserved = int(math.floor(len_train_full * 0.20))
    len_train = len_train_full - len_reserved
    
    print(f"数据集划分: 训练集={len_train}, 保留集={len_reserved}, 验证集={len_val}, 测试集={len_test}")
    print(f"特征数量: {n_features}")
    print(f"趋势提取窗口: {args.trend_window}个时间步")
    
    # 加载数据
    train, reserved, val, test, timestamps = load_time_series_data(
        args.dataset, len_train_full, len_train, len_val
    )
    
    # Z-score标准化
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val = zscore.transform(val)
    test = zscore.transform(test)
    
    # ========== 提取趋势分量 ==========
    print(f"\n正在从时间序列中提取趋势分量（窗口={args.trend_window}）...")
    
    train_trend = extract_trend_component(train, args.trend_window)
    val_trend = extract_trend_component(val, args.trend_window)
    test_trend = extract_trend_component(test, args.trend_window)
    
    print(f"趋势提取完成！")
    print(f"  训练集趋势形状: {train_trend.shape}")
    print(f"  验证集趋势形状: {val_trend.shape}")
    print(f"  测试集趋势形状: {test_trend.shape}")
    
    # 统计趋势分量的信息
    print(f"\n趋势分量统计:")
    print(f"  范围: [{train_trend.min():.4f}, {train_trend.max():.4f}]")
    print(f"  均值: {train_trend.mean():.4f}")
    print(f"  标准差: {train_trend.std():.4f}")
    
    # ========== 准备训练数据（持久化基线 + LSTM残差修正）==========
    print(f"\n准备训练数据（基线 + LSTM残差修正）...")
    print(f"添加移动平均特征，窗口大小: {args.ma_window}")
    
    # 转换数据并计算基线和残差
    x_train, y_train_true, y_train_baseline, residual_train = data_transform_with_baseline(
        train_trend, args.n_his, args.n_pred, use_ma=True, ma_window=args.ma_window
    )
    x_val, y_val_true, y_val_baseline, residual_val = data_transform_with_baseline(
        val_trend, args.n_his, args.n_pred, use_ma=True, ma_window=args.ma_window
    )
    x_test, y_test_true, y_test_baseline, residual_test = data_transform_with_baseline(
        test_trend, args.n_his, args.n_pred, use_ma=True, ma_window=args.ma_window
    )
    
    print(f"\n数据准备完成：")
    print(f"  输入特征数: {x_train.shape[2]} (原始: {n_features}, MA窗口: {args.ma_window})")
    print(f"  输出特征数: {y_train_true.shape[1]}")
    print(f"  残差统计: 均值={residual_train.mean():.6f}, 标准差={residual_train.std():.6f}")
    
    # 转换为Tensor
    x_train = torch.FloatTensor(x_train).to(device)
    residual_train = torch.FloatTensor(residual_train).to(device)
    y_train_baseline = torch.FloatTensor(y_train_baseline).to(device)
    y_train_true = torch.FloatTensor(y_train_true).to(device)
    
    x_val = torch.FloatTensor(x_val).to(device)
    residual_val = torch.FloatTensor(residual_val).to(device)
    y_val_baseline = torch.FloatTensor(y_val_baseline).to(device)
    y_val_true = torch.FloatTensor(y_val_true).to(device)
    
    x_test = torch.FloatTensor(x_test).to(device)
    residual_test = torch.FloatTensor(residual_test).to(device)
    y_test_baseline = torch.FloatTensor(y_test_baseline).to(device)
    y_test_true = torch.FloatTensor(y_test_true).to(device)
    
    # 创建数据加载器（LSTM学习预测残差）
    train_data = utils.data.TensorDataset(x_train, residual_train, y_train_baseline, y_train_true)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    
    val_data = utils.data.TensorDataset(x_val, residual_val, y_val_baseline, y_val_true)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    
    test_data = utils.data.TensorDataset(x_test, residual_test, y_test_baseline, y_test_true)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    
    return n_features, zscore, train_iter, val_iter, test_iter


def prepare_model(args, n_features, device):
    """准备模型、损失函数、优化器等"""
    # 计算输入特征数（原始特征 + 移动平均特征）
    input_features = n_features * 2  # 原始 + MA
    
    # 创建LSTM残差预测模型
    model = LSTMResidualPredictor(
        input_size=input_features,
        output_size=n_features,  # 输出原始特征维度的残差
        seq_len=args.n_his,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.droprate
    ).to(device)
    
    # 损失函数
    loss_fn = nn.MSELoss()
    
    # Early Stopping
    es = earlystopping.EarlyStopping(
        delta=0.0,
        patience=args.patience,
        path=f'LSTM_Residual_trend_{args.dataset}.pt',
        verbose=True
    )
    
    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay_rate
    )
    
    # 学习率调度器
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
    
    for x, residual_true, y_baseline, y_true in train_iter:
        optimizer.zero_grad()
        
        # LSTM预测残差
        residual_pred = model(x)
        
        # 计算损失（预测残差 vs 真实残差）
        l = loss_fn(residual_pred, residual_true)
        
        # 反向传播
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
        for x, residual_true, y_baseline, y_true in val_iter:
            # LSTM预测残差
            residual_pred = model(x)
            
            # 计算损失（预测残差 vs 真实残差）
            l = loss_fn(residual_pred, residual_true)
            val_loss += l.item() * x.size(0)
    
    val_loss = val_loss / len(val_iter.dataset)
    return val_loss


def train(args, model, loss_fn, optimizer, scheduler, es, train_iter, val_iter):
    """训练模型"""
    print("\n开始训练...")
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_epoch(model, loss_fn, optimizer, train_iter)
        
        # 验证
        val_loss = val_epoch(model, loss_fn, val_iter)
        
        # 学习率调度
        scheduler.step()
        
        # GPU内存占用
        if torch.cuda.is_available():
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000
        else:
            gpu_mem_alloc = 0
        
        # 打印信息
        print(f'Epoch: {epoch:03d} | Lr: {optimizer.param_groups[0]["lr"]:.10f} | '
              f'Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f} | '
              f'GPU occupy: {gpu_mem_alloc:.2f} MiB')
        
        # Early Stopping
        es(val_loss, model)
        if es.early_stop:
            print('Early stopping')
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(es.path))


def test(args, model, loss_fn, test_iter, zscore):
    """测试模型"""
    model.eval()
    
    y_true_list = []
    y_pred_list = []
    y_baseline_list = []
    mae_list = []
    mse_list = []
    
    with torch.no_grad():
        for x, residual_true, y_baseline, y_true in test_iter:
            # LSTM预测残差
            residual_pred = model(x)
            
            # 最终预测 = 持久化基线 + LSTM残差修正
            y_pred = y_baseline + residual_pred
            
            # 转换为numpy
            y_true_np = y_true.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()
            y_baseline_np = y_baseline.cpu().numpy()
            
            # 计算误差
            mae = np.abs(y_true_np - y_pred_np)
            mse = (y_true_np - y_pred_np) ** 2
            
            # 保存结果
            y_true_list.append(y_true_np)
            y_pred_list.append(y_pred_np)
            y_baseline_list.append(y_baseline_np)
            mae_list.append(mae)
            mse_list.append(mse)
    
    # 合并所有批次的结果
    mae_array = np.concatenate(mae_list, axis=0)
    mse_array = np.concatenate(mse_list, axis=0)
    y_true_array = np.concatenate(y_true_list, axis=0)
    y_pred_array = np.concatenate(y_pred_list, axis=0)
    y_baseline_array = np.concatenate(y_baseline_list, axis=0)
    
    # 计算各项指标
    MAE = np.mean(mae_array)
    RMSE = np.sqrt(np.mean(mse_array))
    
    # 计算归一化指标（使用标准差）
    y_std = np.std(y_true_array)
    NMAE = MAE / y_std * 100
    NRMSE = RMSE / y_std * 100
    
    # 计算R²
    ss_res = np.sum((y_true_array - y_pred_array) ** 2)
    ss_tot = np.sum((y_true_array - np.mean(y_true_array)) ** 2)
    R2 = 1 - (ss_res / ss_tot)
    
    # 计算MSE
    test_MSE = np.mean(mse_array)
    
    # 同时计算持久化基线的性能
    mae_baseline = np.mean(np.abs(y_true_array - y_baseline_array))
    mse_baseline = np.mean((y_true_array - y_baseline_array) ** 2)
    rmse_baseline = np.sqrt(mse_baseline)
    ss_res_baseline = np.sum((y_true_array - y_baseline_array) ** 2)
    r2_baseline = 1 - (ss_res_baseline / ss_tot)
    
    # 打印结果
    print('\n' + '='*60)
    print(f'趋势预测测试结果 - Dataset: {args.dataset}')
    print('='*60)
    print('\n持久化基线（t-1预测t）:')
    print(f'  MAE:       {mae_baseline:.6f}')
    print(f'  RMSE:      {rmse_baseline:.6f}')
    print(f'  R²:        {r2_baseline:.6f}')
    print('\n基线 + LSTM残差修正:')
    print(f'  Test MSE:  {test_MSE:.6f}')
    print(f'  MAE:       {MAE:.6f}')
    print(f'  RMSE:      {RMSE:.6f}')
    print(f'  NMAE:      {NMAE:.2f}%')
    print(f'  NRMSE:     {NRMSE:.2f}%')
    print(f'  R²:        {R2:.6f}')
    print('\n性能提升:')
    mae_improvement = (mae_baseline - MAE) / mae_baseline * 100
    rmse_improvement = (rmse_baseline - RMSE) / rmse_baseline * 100
    r2_improvement = R2 - r2_baseline
    print(f'  MAE改善:   {mae_improvement:.2f}%')
    print(f'  RMSE改善:  {rmse_improvement:.2f}%')
    print(f'  R²提升:    {r2_improvement:.6f}')
    print(f'{"="*60}\n')
    
    # 保存结果到文件
    results_file = f'LSTM_Residual_trend_{args.dataset}_results.txt'
    with open(results_file, 'w') as f:
        f.write('Persistence Baseline + LSTM Residual Correction\n')
        f.write('Trend Component Prediction Results\n')
        f.write(f'Dataset: {args.dataset}\n')
        f.write('='*60 + '\n')
        f.write('\nPersistence Baseline (t-1 predicts t):\n')
        f.write(f'  MAE:       {mae_baseline:.6f}\n')
        f.write(f'  RMSE:      {rmse_baseline:.6f}\n')
        f.write(f'  R²:        {r2_baseline:.6f}\n')
        f.write('\nBaseline + LSTM Residual Correction:\n')
        f.write(f'  Test MSE:  {test_MSE:.6f}\n')
        f.write(f'  MAE:       {MAE:.6f}\n')
        f.write(f'  RMSE:      {RMSE:.6f}\n')
        f.write(f'  NMAE:      {NMAE:.2f}%\n')
        f.write(f'  NRMSE:     {NRMSE:.2f}%\n')
        f.write(f'  R²:        {R2:.6f}\n')
        f.write('\nPerformance Improvement:\n')
        f.write(f'  MAE improvement:   {mae_improvement:.2f}%\n')
        f.write(f'  RMSE improvement:  {rmse_improvement:.2f}%\n')
        f.write(f'  R² gain:           {r2_improvement:.6f}\n')
    
    print(f'结果已保存到 {results_file}')
    
    return MAE, RMSE, NMAE, NRMSE, R2


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    print("="*60)
    print("持久化基线 + LSTM残差修正")
    print("时间序列趋势预测")
    print("="*60)
    print("方法：使用t-1作为基线预测，LSTM学习并修正残差")
    print("="*60)
    
    # 获取参数
    args, device = get_parameters()
    
    # 数据准备
    print("\n准备数据...")
    n_features, zscore, train_iter, val_iter, test_iter = data_preparation(args, device)
    
    # 准备模型
    print("\n准备模型...")
    model, loss_fn, es, optimizer, scheduler = prepare_model(args, n_features, device)
    
    print(f"\n模型结构:")
    print(model)
    print(f"\n模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    print("\n开始训练LSTM残差预测模型...")
    train(args, model, loss_fn, optimizer, scheduler, es, train_iter, val_iter)
    
    # 测试模型
    print("\n测试趋势预测模型（基线 + LSTM修正）...")
    test(args, model, loss_fn, test_iter, zscore)
    
    print("\n训练完成!")
    print("模型文件: LSTM_Residual_trend_{}.pt".format(args.dataset))
    print("结果文件: LSTM_Residual_trend_{}_results.txt".format(args.dataset))
