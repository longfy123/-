import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

def compute_similarity_matrix(data, threshold=0.7):

    n_stations = data.shape[1]
    X = data.T - data.T.mean(axis=1, keepdims=True)
    
    # Calculate the correlation coefficient matrix
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1  
    corr = (X @ X.T) / (norm @ norm.T)
    corr = np.clip(corr, -1, 1)
    
    W = np.zeros((n_stations, n_stations))
    W[corr >= threshold] = corr[corr >= threshold]
    np.fill_diagonal(W, 1)
    
    return W

def main(threshold=0.5, save_path='/root/MoELLM/data/nanjing/adj_similarity.npz'):
    df = pd.read_csv('/root/MoELLM/data/nanjing/traffic.csv', header=None)
    data = np.nan_to_num(df.values.astype(np.float32))
    
    # Calculate the similarity matrix
    W = compute_similarity_matrix(data, threshold)
    
    # save
    sp.save_npz(save_path, sp.csc_matrix(W))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.7)
    args = parser.parse_args()
    main(threshold=args.threshold)
