import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import json
import os

def decompose_time_series(data_path='/root/stgcn原版/data/shanghai/vel_with_timestamp.csv', 
                          output_dir='/root/stgcn原版/data/shanghai',
                          period=336,  # 默认周期为336 (一周,假设30分钟间隔)
                          model='additive'):
    print(f"正在读取数据: {data_path}")
    # 读取数据
    df = pd.read_csv(data_path, header=None)
    print(f"数据形状: {df.shape}")
    print(f"时间步数: {df.shape[0]}, 站点数: {df.shape[1] - 1}")
    # 第一列是时间戳,去掉它,只保留站点数据
    station_data = df.iloc[:, 1:].values.astype(np.float64)
    n_timesteps, n_stations = station_data.shape
    # 初始化三个数组存储结果
    trend_array = np.zeros_like(station_data)
    seasonal_array = np.zeros_like(station_data)
    residual_array = np.zeros_like(station_data)
    
    # 对每个站点进行分解
    success_count = 0
    fail_count = 0
    
    for i in range(n_stations):
        if (i + 1) % 100 == 0:
            print(f"处理进度: {i+1}/{n_stations}")
        
        try:
            # 获取当前站点的时间序列
            series = station_data[:, i]
            
            # 检查是否有足够的非NaN值
            if np.isnan(series).sum() > len(series) * 0.5:
                print(f"警告: 站点 {i} 有过多缺失值,跳过")
                trend_array[:, i] = series
                seasonal_array[:, i] = 0
                residual_array[:, i] = 0
                fail_count += 1
                continue
            
            # 进行季节性分解
            result = seasonal_decompose(series, 
                                       model=model, 
                                       period=period,
                                       extrapolate_trend='freq')
            
            # 保存分解结果
            trend_array[:, i] = result.trend
            seasonal_array[:, i] = result.seasonal
            residual_array[:, i] = result.resid
            
            success_count += 1
            
        except Exception as e:
            print(f"站点 {i} 分解失败: {str(e)}")
            # 如果分解失败,保持原始数据作为趋势,季节性和残差为0
            trend_array[:, i] = series
            seasonal_array[:, i] = 0
            residual_array[:, i] = 0
            fail_count += 1
    
    print(f"\n分解完成!")
    print(f"成功: {success_count}, 失败: {fail_count}")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为CSV文件
    print("\n正在保存CSV文件...")
    
    # 使用原始的时间戳列作为索引
    timestamp_col = df.iloc[:, 0]
    
    # 添加索引列
    trend_df = pd.DataFrame(trend_array)
    trend_df.insert(0, 'timestamp', timestamp_col)
    trend_path = os.path.join(output_dir, 'vel_trend.csv')
    trend_df.to_csv(trend_path, index=False, header=False)
    print(f"趋势数据保存至: {trend_path}")
    
    seasonal_df = pd.DataFrame(seasonal_array)
    seasonal_df.insert(0, 'timestamp', timestamp_col)
    seasonal_path = os.path.join(output_dir, 'vel_seasonal.csv')
    seasonal_df.to_csv(seasonal_path, index=False, header=False)
    print(f"季节性数据保存至: {seasonal_path}")
    
    residual_df = pd.DataFrame(residual_array)
    residual_df.insert(0, 'timestamp', timestamp_col)
    residual_path = os.path.join(output_dir, 'vel_residual.csv')
    residual_df.to_csv(residual_path, index=False, header=False)
    print(f"残差数据保存至: {residual_path}")
    
    print("\nCSV文件保存完成!")
    
    # 保存分解统计信息
    stats = {
        'n_timesteps': int(n_timesteps),
        'n_stations': int(n_stations),
        'period': int(period),
        'model': model,
        'success_count': int(success_count),
        'fail_count': int(fail_count),
        'trend_mean': float(np.nanmean(trend_array)),
        'trend_std': float(np.nanstd(trend_array)),
        'seasonal_mean': float(np.nanmean(seasonal_array)),
        'seasonal_std': float(np.nanstd(seasonal_array)),
        'residual_mean': float(np.nanmean(residual_array)),
        'residual_std': float(np.nanstd(residual_array))
    }
    
    stats_path = os.path.join(output_dir, 'decompose_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    return trend_array, seasonal_array, residual_array, stats

if __name__ == '__main__':
    trend, seasonal, residual, stats = decompose_time_series(
        data_path='data/shanghai/vel_with_timestamp.csv',
        output_dir='data/shanghai',
        period=336,  # 30分钟间隔，一周的周期
        model='additive'
    )
    
    print("\n所有任务完成!")