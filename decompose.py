import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import os

def decompose_time_series(data_path='/root/MoELLM/data/nanjing/traffic.csv',
                          output_dir='/root/MoELLM/data/nanjing',
                          period=336,  # The default period is 336 (7 days, assuming a 30-minute interval)
                          model='additive'):
    df = pd.read_csv(data_path, header=None)
    station_data = df.iloc[:, :].values.astype(np.float64)
    n_timesteps, n_stations = station_data.shape

    trend_array = np.zeros_like(station_data)
    seasonal_array = np.zeros_like(station_data)
    residual_array = np.zeros_like(station_data)
    
    for i in range(n_stations):
        series = station_data[:, i]
        
        result = seasonal_decompose(series, 
                                   model=model, 
                                   period=period,
                                   extrapolate_trend='freq')
        
        trend_array[:, i] = result.trend
        seasonal_array[:, i] = result.seasonal
        residual_array[:, i] = result.resid
    
    os.makedirs(output_dir, exist_ok=True)

    trend_df = pd.DataFrame(trend_array)
    trend_path = os.path.join(output_dir, 'traffic_trend.csv')
    trend_df.to_csv(trend_path, index=False, header=False)

    seasonal_df = pd.DataFrame(seasonal_array)
    seasonal_path = os.path.join(output_dir, 'traffic_seasonal.csv')
    seasonal_df.to_csv(seasonal_path, index=False, header=False)

    residual_df = pd.DataFrame(residual_array)
    residual_path = os.path.join(output_dir, 'traffic_residual.csv')
    residual_df.to_csv(residual_path, index=False, header=False)

    return trend_array, seasonal_array, residual_array

if __name__ == '__main__':
    trend, seasonal, residual = decompose_time_series(
        data_path='data/nanjing/traffic.csv',
        output_dir='data/nanjing',
        period=336,  
        model='additive'
    )
    