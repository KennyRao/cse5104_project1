# main.py
# Author: Weikai Rao 519139

import numpy as np
import pandas as pd

def rescale_data(X_train, X_test, method='standardization'):
    """
    Rescale the data using either normalization or standardization, default is standardization.
    If method is not recognized, do not rescale the data and return original data.
    """
    normalization_methods = ['normalization', 'min-max', 'minmax', 'min_max', 'min max', 'normal', 'norm']
    standardization_methods = ['standardization', 'standardize', 'std', 'z-score', 'zscore', 'z score', 'standard']
    if method in normalization_methods:
        min_vals = X_train.min(axis=0)
        max_vals = X_train.max(axis=0)
        val_range = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
        X_train_scaled = (X_train - min_vals) / val_range
        X_test_scaled = (X_test - min_vals) / val_range
        return X_train_scaled, X_test_scaled
    elif method in standardization_methods:
        mean_vals = X_train.mean(axis=0)
        std_vals = X_train.std(axis=0)
        std_vals = np.where(std_vals == 0, 1, std_vals)
        X_train_scaled = (X_train - mean_vals) / std_vals
        X_test_scaled = (X_test - mean_vals) / std_vals
        return X_train_scaled, X_test_scaled
    else:
        return X_train, X_test

def mse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean((y_true - y_pred)**2))

def variance_explained(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    var_y = float(np.var(y_true))
    if var_y == 0:
        return 0.0
    return 1.0 - mse(y_true, y_pred) / var_y

def gd_univariate(X, y, init_m=1, init_b=1, alpha=0.05, max_iter=20000, tolerance=1e-6):
    """
    Perform gradient descent to learn m and b by minimizing the cost function.
    X: numpy array of shape (n_samples,)
    y: numpy array of shape (n_samples,)
    """
    m = init_m
    b = init_b
    n = len(y)
    for i in range(max_iter):
        m_gradient = 0.0
        b_gradient = 0.0
        for j in range(n):
            x = X[j]
            error = y[j] - (m * x + b)
            m_gradient += (-2.0) * x * error
            b_gradient += (-2.0) * error
        new_m = m - alpha * (m_gradient/n)
        new_b = b - alpha * (b_gradient/n)
        if abs(new_m - m) < tolerance and abs(new_b - b) < tolerance:
            break
        m = new_m
        b = new_b
    return m, b

def predict_univariate(X, m, b):
    return m * X + b

def fit_and_evaluate_univariate(X_train, y_train, X_test, y_test, rescale_method='standardization', init_m=1, init_b=1, alpha=0.05, max_iter=20000, tolerance=1e-6):
    X_train_scaled, X_test_scaled = rescale_data(X_train, X_test, method=rescale_method)
    m, b = gd_univariate(X_train_scaled, y_train, init_m, init_b, alpha, max_iter, tolerance)
    y_train_pred = predict_univariate(X_train_scaled, m, b)
    y_test_pred = predict_univariate(X_test_scaled, m, b)
    train_mse = mse(y_train, y_train_pred)
    test_mse = mse(y_test, y_test_pred)
    train_ve = variance_explained(y_train, y_train_pred)
    test_ve = variance_explained(y_test, y_test_pred)
    return m, b, train_mse, test_mse, train_ve, test_ve

def main():
    data_file = 'Concrete_Data.csv'
    my_header = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age', 'Concrete compressive strength']
    df = pd.read_csv(data_file)  # 1030 rows × 9 columns, header is column names
    df.columns = my_header
    print(f"Data shape: {df.shape}")
    
    test_df = df.iloc[500:630]  # 130 rows × 9 columns
    train_df = pd.concat([df.iloc[0:500], df.iloc[630:]])  # 900 rows × 9 columns
    
    feature_cols = df.columns[:-1]  # predictors (8 columns)
    target_col = df.columns[-1]  # response: "Compressive Strength"
    
    # Q1.1 Use normalized/standardized values for the predictors
    for col in feature_cols:
        X_train = train_df[col].to_numpy(dtype=float)
        y_train = train_df[target_col].to_numpy(dtype=float)
        X_test = test_df[col].to_numpy(dtype=float)
        y_test = test_df[target_col].to_numpy(dtype=float)
        
        m, b, train_mse, test_mse, train_ve, test_ve = fit_and_evaluate_univariate(X_train, y_train, X_test, y_test)
        print(f"Predictor: {col}")
        print(f"  m={m}, b={b}")
        print(f"  Train MSE: {train_mse}, Test MSE: {test_mse}")
        print(f"  Train Variance Explained: {train_ve}, Test Variance Explained: {test_ve}")
    
    # Q1.2 Use original values for the predictors
    for col in feature_cols:
        X_train = train_df[col].to_numpy(dtype=float)
        y_train = train_df[target_col].to_numpy(dtype=float)
        X_test = test_df[col].to_numpy(dtype=float)
        y_test = test_df[target_col].to_numpy(dtype=float)
        
        m, b, train_mse, test_mse, train_ve, test_ve = fit_and_evaluate_univariate(X_train, y_train, X_test, y_test, rescale_method=None)
        print(f"(Original) Predictor: {col}")
        print(f"  m={m}, b={b}")
        print(f"  Train MSE: {train_mse}, Test MSE: {test_mse}")
        print(f"  Train Variance Explained: {train_ve}, Test Variance Explained: {test_ve}")
    

if __name__ == "__main__":
    main()
