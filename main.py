# main.py
# Author: Weikai Rao 519139

import numpy as np
import pandas as pd

def rescale_data(X_train, X_test, method='standardization'):
    """
    Rescale the data using either normalization or standardization, default is standardization.
    If method is not recognized, do not rescale the data and return original data.
    Return the rescaled X_train, X_test, and the method used (or None if no rescaling done).
    """
    normalization_methods = ['normalization', 'min-max', 'minmax', 'min_max', 'min max', 'normal', 'norm']
    standardization_methods = ['standardization', 'standardize', 'std', 'z-score', 'zscore', 'z score', 'standard']
    if method in normalization_methods:
        min_vals = X_train.min(axis=0)
        max_vals = X_train.max(axis=0)
        val_range = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
        X_train_scaled = (X_train - min_vals) / val_range
        X_test_scaled = (X_test - min_vals) / val_range
        return X_train_scaled, X_test_scaled, method
    elif method in standardization_methods:
        mean_vals = X_train.mean(axis=0)
        std_vals = X_train.std(axis=0)
        std_vals = np.where(std_vals == 0, 1, std_vals)
        X_train_scaled = (X_train - mean_vals) / std_vals
        X_test_scaled = (X_test - mean_vals) / std_vals
        return X_train_scaled, X_test_scaled, method
    else:
        return X_train, X_test, None

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

def get_safe_alpha_for_raw_univariate(X):
    """
    Return a safe learning rate alpha for univariate data X that are not rescaled.
    """
    X = np.asarray(X, dtype=np.float64).ravel()
    n = X.size
    if n == 0:
        return 1e-3
    LipschitzConstant = (2.0 / n) * (np.sum(X * X, dtype=np.float64) + n)
    if LipschitzConstant <= 0 or not np.isfinite(LipschitzConstant):
        return 1e-3
    return 0.5 / LipschitzConstant

def get_safe_alpha_for_raw_multivariate(X):
    """
    Return a safe learning rate alpha for multivariate data X that are not rescaled.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n == 0:
        return 1e-3
    A = np.c_[X, np.ones(n, dtype=np.float64)]  # add a column of 1s for intercept
    smax = np.linalg.svd(A, compute_uv=False, hermitian=False)[0]  # largest singular value
    LipschitzConstant = (2.0 / n) * (smax ** 2)
    if not np.isfinite(LipschitzConstant) or LipschitzConstant <= 0:
        return 1e-3
    return 1.0 / LipschitzConstant

def gd_univariate(X, y, init_m=1, init_b=1, alpha=0.05, max_iter=20000, tolerance=1e-6, patience=200):
    """
    Perform gradient descent to learn m and b by minimizing the cost function.
    Return the learned m, b, and number of iterations used.
    """
    X = np.asarray(X, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    inv_n = 1.0 / n

    m = float(init_m)
    b = float(init_b)

    last_best = np.inf
    stale = 0

    for i in range(max_iter):
        errors = (m * X + b) - y  # shape (n,)
        loss = float(np.dot(errors, errors) * inv_n)  # MSE

        # Early stopping based on loss improvement
        if last_best - loss < tolerance:
            stale += 1
            if stale >= patience:
                return m, b, i + 1
        else:
            last_best = loss
            stale = 0

        # Vectorized gradients
        grad_m = 2.0 * inv_n * np.dot(X, errors)
        grad_b = 2.0 * inv_n * errors.sum()

        # Parameter update
        delta_m = -alpha * grad_m
        delta_b = -alpha * grad_b
        m += delta_m
        b += delta_b

        # Numerical safety & small-step convergence
        if not (np.isfinite(m) and np.isfinite(b)):
            return m, b, i + 1
        if max(abs(delta_m), abs(delta_b)) < tolerance:
            return m, b, i + 1

    return m, b, max_iter

def gd_multivariate(X, y, init_w=None, init_b=0.0, alpha=0.05, max_iter=20000, tolerance=1e-6, patience=200):
    """
    Perform gradient descent to learn weights w and bias b by minimizing the cost function.
    Return the learned w (as a numpy array), b (as a float), and number of iterations used.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = X.shape
    inv_n = 1.0 / n

    w = np.zeros(p, dtype=np.float64) if init_w is None else np.asarray(init_w, dtype=np.float64).ravel()
    b = float(init_b)

    last_best = np.inf
    stale = 0

    for i in range(max_iter):
        errors = (X @ w + b) - y
        loss = float(np.dot(errors, errors) * inv_n)

        if last_best - loss < tolerance:
            stale += 1
            if stale >= patience:
                return w, b, i + 1
        else:
            stale = 0
            last_best = loss

        grad_w = 2.0 * inv_n * (X.T @ errors)
        grad_b = 2.0 * inv_n * errors.sum()

        dw = -alpha * grad_w
        db = -alpha * grad_b
        w += dw
        b += db

        if not (np.all(np.isfinite(w)) and np.isfinite(b)):
            return w, b, i + 1
        if max(np.max(np.abs(dw)), abs(db)) < tolerance:
            return w, b, i + 1

    return w, b, max_iter

def predict_multivariate(X, w, b):
    return np.asarray(X, dtype=np.float64) @ np.asarray(w, dtype=np.float64) + float(b)

def predict_univariate(X, m, b):
    return m * X + b

def fit_and_evaluate_univariate(X_train, y_train, X_test, y_test, rescale_method='standardization', init_m=1, init_b=1, alpha=0.05, max_iter=20000, tolerance=1e-6, patience=200):
    """
    Fit a univariate linear regression model using gradient descent on the training data,
    and evaluate it on both training and test data.
    Return the learned m, b, training MSE, test MSE, training variance explained, test variance explained, and number of iterations used.
    """
    X_train_scaled, X_test_scaled, method = rescale_data(X_train, X_test, method=rescale_method)
    
    if method is None:  # no rescaling done
        safe_alpha = get_safe_alpha_for_raw_univariate(X_train_scaled)
        alpha = min(alpha, safe_alpha)
        init_m = 0.0
        init_b = 0.0
    
    m, b, iter_used = gd_univariate(X_train_scaled, y_train, init_m, init_b, alpha, max_iter, tolerance, patience)
    
    y_train_pred = predict_univariate(X_train_scaled, m, b)
    y_test_pred = predict_univariate(X_test_scaled, m, b)
    
    train_mse = mse(y_train, y_train_pred)
    test_mse = mse(y_test, y_test_pred)
    
    train_ve = variance_explained(y_train, y_train_pred)
    test_ve = variance_explained(y_test, y_test_pred)
    
    return m, b, train_mse, test_mse, train_ve, test_ve, iter_used

def fit_and_evaluate_multivariate(X_train, y_train, X_test, y_test, rescale_method='standardization', init_w=None, init_b=0.0, alpha=0.05, max_iter=20000, tolerance=1e-6, patience=200):
    """
    Fit a multivariate linear regression model using gradient descent on the training data,
    and evaluate it on both training and test data.
    Return the learned w (as a numpy array), b (as a float), training MSE, test MSE, training variance explained, test variance explained, and number of iterations used.
    """
    X_train_scaled, X_test_scaled, method = rescale_data(X_train, X_test, method=rescale_method)
    
    if method is None:  # no rescaling done
        safe_alpha = get_safe_alpha_for_raw_multivariate(X_train_scaled)
        alpha = min(alpha, safe_alpha)
        init_b = 0.0
    
    w, b, iter_used = gd_multivariate(X_train_scaled, y_train, init_w, init_b, alpha, max_iter, tolerance, patience)
    
    y_train_pred = predict_multivariate(X_train_scaled, w, b)
    y_test_pred = predict_multivariate(X_test_scaled, w, b)
    
    train_mse = mse(y_train, y_train_pred)
    test_mse = mse(y_test, y_test_pred)
    
    train_ve = variance_explained(y_train, y_train_pred)
    test_ve = variance_explained(y_test, y_test_pred)
    
    return w, b, train_mse, test_mse, train_ve, test_ve, iter_used

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
        
        m, b, train_mse, test_mse, train_ve, test_ve, iter_used = fit_and_evaluate_univariate(X_train, y_train, X_test, y_test)
        print(f"Predictor: {col}")
        print(f"  m={m}, b={b}")
        print(f"  Train MSE: {train_mse}, Test MSE: {test_mse}")
        print(f"  Train Variance Explained: {train_ve}, Test Variance Explained: {test_ve}")
        print(f"  Iterations used: {iter_used}")
    
    # Q1.2 Use original values for the predictors
    for col in feature_cols:
        X_train = train_df[col].to_numpy(dtype=float)
        y_train = train_df[target_col].to_numpy(dtype=float)
        X_test = test_df[col].to_numpy(dtype=float)
        y_test = test_df[target_col].to_numpy(dtype=float)
        
        m, b, train_mse, test_mse, train_ve, test_ve, iter_used = fit_and_evaluate_univariate(X_train, y_train, X_test, y_test, rescale_method=None, max_iter=100000)
        print(f"Predictor (Unscaled): {col}")
        print(f"  m={m}, b={b}")
        print(f"  Train MSE: {train_mse}, Test MSE: {test_mse}")
        print(f"  Train Variance Explained: {train_ve}, Test Variance Explained: {test_ve}")
        print(f"  Iterations used: {iter_used}")
    

if __name__ == "__main__":
    main()
