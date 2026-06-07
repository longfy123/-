import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

def load_adj(dataset_name):
    dataset_path = '/root/MoELLM/data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    adj = adj.tocsc()
    
    if dataset_name == 'shanghai':
        n_vertex = 1042
    elif dataset_name == 'nanjing':
        n_vertex = 1000

    return adj, n_vertex


def load_data(dataset_name, len_train_full, len_train, len_val):
    dataset_path = '/root/MoELLM/data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    traffic = pd.read_csv(os.path.join(dataset_path, 'traffic.csv'))

    train = traffic[: len_train]
    reserved = traffic[len_train: len_train_full]
    val = traffic[len_train_full: len_train_full + len_val]
    test = traffic[len_train_full + len_val:]
    return train, val, test, reserved   
 

def data_transform_multistep(data, n_his, n_pred_ms, device):
    n_vertex = data.shape[1]
    len_record = len(data)
    window_size = n_his + n_pred_ms
    num_samples = len_record // window_size

    x = np.zeros([num_samples, n_pred_ms, 1, n_his, n_vertex])
    y = np.zeros([num_samples, n_pred_ms, n_vertex])

    for i in range(num_samples):
        base = i * window_size
        for j in range(n_pred_ms):
            x[i, j, 0] = data[base + j: base + j + n_his]
            y[i, j] = data[base + j + n_his]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)


def data_transform(data, n_his, n_pred, device):
    # produce data slices for x_data and y_data
    # using NON-OVERLAPPING windows for independent samples

    n_vertex = data.shape[1]
    len_record = len(data)
    
    window_size = n_his + n_pred
    num = (len_record - n_his - n_pred + 1) // window_size
    
    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, n_vertex])
    
    for i in range(num):
        head = i * window_size
        tail = head + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        y[i] = data[tail + n_pred - 1]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)
