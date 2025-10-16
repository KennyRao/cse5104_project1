# main.py
# Author: Weikai Rao 519139

import numpy as np
import pandas as pd

def rescale_data(X_train, X_test, method='standardization'):
    """
    Rescale the data using either normalization or standardization, default is standardization.
    If method is not recognized, use standardization
    """
    normalization_methods = ['normalization', 'min-max', 'minmax', 'min_max', 'min max', 'normal', 'norm']
    if method in normalization_methods:
        min_vals = X_train.min(axis=0)
        max_vals = X_train.max(axis=0)
        val_range = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
        X_train_scaled = (X_train - min_vals) / val_range
        X_test_scaled = (X_test - min_vals) / val_range
        return X_train_scaled, X_test_scaled
    else:
        mean_vals = X_train.mean(axis=0)
        std_vals = X_train.std(axis=0)
        std_vals = np.where(std_vals == 0, 1, std_vals)
        X_train_scaled = (X_train - mean_vals) / std_vals
        X_test_scaled = (X_test - mean_vals) / std_vals
        return X_train_scaled, X_test_scaled

def main():
    data_file = 'Concrete_Data.csv'
    df = pd.read_csv(data_file)  # 1030 rows × 9 columns, header is column names
    print(f"Data shape: {df.shape}")
    
    test_df = df.iloc[500:630]  # 130 rows × 9 columns
    train_df = pd.concat([df.iloc[0:500], df.iloc[630:]])  # 900 rows × 9 columns
    
    feature_cols = df.columns[:-1]  # predictors (8 columns)
    target_col = df.columns[-1]  # response: "Compressive Strength"
    
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df[target_col].to_numpy(dtype=float)
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    X_train_scaled, X_test_scaled = rescale_data(X_train, X_test, method='standardization')

if __name__ == "__main__":
    main()
