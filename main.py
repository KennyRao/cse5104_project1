# main.py
# Author: Weikai Rao 519139

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

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
    X_train_scaled, X_test_scaled, method_used = rescale_data(X_train, X_test, method=rescale_method)
    
    if method_used is None:  # no rescaling done
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
    X_train_scaled, X_test_scaled, method_used = rescale_data(X_train, X_test, method=rescale_method)
    
    if method_used is None:  # no rescaling done
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

def one_step_update_multivariate(X, y, alpha=0.1, init_w=None, init_b=1.0):
    """
    Perform one step of gradient descent update for multivariate linear regression.
    Return the updated weights w (as a numpy array) and bias b (as a float).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = X.shape
    w = np.ones(p, dtype=np.float64) if init_w is None else np.asarray(init_w, dtype=np.float64).ravel()
    b = float(init_b)
    inv_n = 1.0 / n

    r = (X @ w + b) - y
    grad_w = 2.0 * inv_n * (X.T @ r)
    grad_b = 2.0 * inv_n * r.sum()

    w_new = w - alpha * grad_w
    b_new = b - alpha * grad_b
    return w_new, b_new

def prepare_features(X_train, X_test, method='raw'):
    """
    Prepare features by applying the specified rescaling method.
    Supported methods: 'raw', 'standardization', 'normalization', 'log'.
    """
    method = (method or '').lower()
    if method in ['raw', 'none']:
        return X_train, X_test, 'raw'
    if method in ['standardization', 'standardize', 'std', 'z-score', 'zscore', 'z score', 'standard']:
        X_train_scaled, X_test_scaled, _ = rescale_data(X_train, X_test, method='standardization')
        return X_train_scaled, X_test_scaled, 'standardization'
    if method in ['normalization', 'min-max', 'minmax', 'min_max', 'min max', 'normal', 'norm']:
        X_train_scaled, X_test_scaled, _ = rescale_data(X_train, X_test, method='normalization')
        return X_train_scaled, X_test_scaled, 'normalization'
    if method in ['log', 'log1p', 'log(x+1)', 'log(x + 1)']:
        # log transform only
        return np.log1p(X_train), np.log1p(X_test), 'log1p'
    # default: raw
    return X_train, X_test, 'raw'

def statsmodels_fit_and_evaluate(X_train, y_train, X_test, y_test, feature_cols, target_name="y"):
    """
    Fit a multivariate linear regression model using statsmodels OLS,
    and evaluate it on both training and test data.
    Return a dictionary containing parameters, p-values, t-values, training MSE, test MSE, training variance explained, test variance explained, and the fitted result object.
    """
    X_train_df = pd.DataFrame(X_train, columns=feature_cols)
    X_test_df = pd.DataFrame(X_test,  columns=feature_cols)
    y_train_s  = pd.Series(y_train, name=target_name)
    
    # add intercept
    X_train_const = sm.add_constant(X_train_df, has_constant='add')
    X_test_const = sm.add_constant(X_test_df, has_constant='add')
    
    # fit model
    model = sm.OLS(y_train_s, X_train_const)
    results = model.fit()
    
    # predict
    y_pred_train = results.predict(X_train_const)
    y_pred_test = results.predict(X_test_const)
    
    # evaluate
    train_mse = mse(y_train, y_pred_train)
    test_mse = mse(y_test, y_pred_test)
    train_ve = variance_explained(y_train, y_pred_train)
    test_ve = variance_explained(y_test, y_pred_test)
    
    names = list(X_train_const.columns)
    params  = pd.Series(np.asarray(results.params), index=names)
    pvalues = pd.Series(np.asarray(results.pvalues), index=names)
    tvalues = pd.Series(np.asarray(results.tvalues), index=names)
    return {
        "params": params,
        "pvalues": pvalues,
        "tvalues": tvalues,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_ve": train_ve,
        "test_ve": test_ve,
        "result": results,
        }

def run_multivariate_ols_analysis(X_train, y_train, X_test, y_test, feature_cols, method_tag):
    X_train_prep, X_test_prep, method_used = prepare_features(X_train, X_test, method_tag)
    results = statsmodels_fit_and_evaluate(X_train_prep, y_train, X_test_prep, y_test, feature_cols=feature_cols, target_name="Concrete compressive strength")

    p_vals = results["pvalues"]
    pv_features = p_vals.loc[feature_cols] if set(feature_cols).issubset(p_vals.index) else pd.Series({name: p_vals.get(name, np.nan) for name in feature_cols})

    print(f"\n[Part B — {method_used}] Multivariate OLS (train-fit, test-eval)")
    print("  Train: MSE={:.6f}  VE={:.6f}".format(results["train_mse"], results["train_ve"]))
    print("  Test : MSE={:.6f}  VE={:.6f}".format(results["test_mse"],  results["test_ve"]))
    print("  p-values (features):")
    for k, v in pv_features.items():
        print(f"    {k}: {v:.6g}")

    return results, pv_features

def pvalues_for_gd_solution(X, y, w, b, feature_names=None, robust=None):
    """
    Compute t-stat p-values for (w, b), assuming OLS model y ≈ Xw + b.
    robust: None (classical), or 'HC0'/'HC1'/'HC2'/'HC3' for robust SEs.
    Returns a DataFrame with rows [features..., 'const'] and columns ['coef','se','t','p'].
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = X.shape
    A = np.c_[X, np.ones(n)]
    theta = np.r_[np.asarray(w, dtype=np.float64).ravel(), float(b)]

    r = y - A @ theta
    k = p + 1
    df = max(n - k, 1)

    XtX = A.T @ A
    XtX_inv = np.linalg.pinv(XtX)

    if robust is None:
        sigma2 = float(r @ r) / df
        cov = sigma2 * XtX_inv
    else:
        if robust.upper() in ("HC0", "HC1"):
            meat = A.T @ (r[:, None] * r[:, None] * A)
            if robust.upper() == "HC1":
                meat *= n / df
        elif robust.upper() in ("HC2", "HC3"):
            h = np.sum(A * (A @ XtX_inv), axis=1)
            if robust.upper() == "HC2":
                w_i = (r**2) / np.clip(1 - h, 1e-12, None)
            else:  # HC3
                w_i = (r**2) / np.clip((1 - h)**2, 1e-12, None)
            meat = A.T @ (w_i[:, None] * A)
        else:
            raise ValueError("robust must be one of None, 'HC0','HC1','HC2','HC3'")
        cov = XtX_inv @ meat @ XtX_inv

    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    tvals = theta / np.where(se == 0, np.inf, se)
    pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), df))

    names = list(feature_names) + ["const"] if feature_names is not None else [f"x{i+1}" for i in range(p)] + ["const"]
    return pd.DataFrame({"coef": theta, "se": se, "t": tvals, "p": pvals}, index=names)

def gd_multivariate_with_history(X, y, init_w=None, init_b=0.0, alpha=0.05, max_iter=20000, tolerance=1e-6, patience=200):
    """
    Same as gd_multivariate but also returns a per-iteration training MSE history.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = X.shape
    inv_n = 1.0 / n

    w = np.zeros(p, dtype=np.float64) if init_w is None else np.asarray(init_w, dtype=np.float64).ravel()
    b = float(init_b)

    last_best = np.inf
    stale = 0
    mse_history = []

    for i in range(max_iter):
        errors = (X @ w + b) - y
        loss = float(np.dot(errors, errors) * inv_n)  # MSE
        mse_history.append(loss)

        # Early stopping based on loss improvement
        if last_best - loss < tolerance:
            stale += 1
            if stale >= patience:
                return w, b, mse_history
        else:
            last_best = loss
            stale = 0

        # Gradients and update
        grad_w = 2.0 * inv_n * (X.T @ errors)
        grad_b = 2.0 * inv_n * errors.sum()

        w -= alpha * grad_w
        b -= alpha * grad_b

        # Numerical safety & tiny step convergence
        if not (np.all(np.isfinite(w)) and np.isfinite(b)):
            return w, b, mse_history
        if max(np.max(np.abs(alpha * grad_w)), abs(alpha * grad_b)) < tolerance:
            return w, b, mse_history

    return w, b, mse_history


def save_loss_curve(loss_history, out_path="gd_loss_curve.png", title="GD Training Loss (MSE) per Iteration"):
    """
    Save a simple line plot of MSE vs iteration to disk (PNG by default).
    """
    fig, ax = plt.subplots()
    ax.plot(range(1, len(loss_history) + 1), loss_history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

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
        X_train = train_df[col].to_numpy(dtype=np.float64)
        y_train = train_df[target_col].to_numpy(dtype=np.float64)
        X_test = test_df[col].to_numpy(dtype=np.float64)
        y_test = test_df[target_col].to_numpy(dtype=np.float64)
        
        m, b, train_mse, test_mse, train_ve, test_ve, iter_used = fit_and_evaluate_univariate(X_train, y_train, X_test, y_test)
        print(f"\nPredictor: {col}")
        print(f"  m={m}, b={b}")
        print(f"  Train MSE: {train_mse}, Test MSE: {test_mse}")
        print(f"  Train Variance Explained: {train_ve}, Test Variance Explained: {test_ve}")
        print(f"  Iterations used: {iter_used}")
    
    # Q1.2 Use original values for the predictors
    for col in feature_cols:
        X_train = train_df[col].to_numpy(dtype=np.float64)
        y_train = train_df[target_col].to_numpy(dtype=np.float64)
        X_test = test_df[col].to_numpy(dtype=np.float64)
        y_test = test_df[target_col].to_numpy(dtype=np.float64)
        
        m, b, train_mse, test_mse, train_ve, test_ve, iter_used = fit_and_evaluate_univariate(X_train, y_train, X_test, y_test, rescale_method=None, max_iter=100000)
        print(f"\nPredictor (Unscaled): {col}")
        print(f"  m={m}, b={b}")
        print(f"  Train MSE: {train_mse}, Test MSE: {test_mse}")
        print(f"  Train Variance Explained: {train_ve}, Test Variance Explained: {test_ve}")
        print(f"  Iterations used: {iter_used}")
    
    # Q2.1 Code test 1
    X1 = np.array([[3.0, 4.0, 5.0]])
    y1 = np.array([4.0])
    init_w = np.ones(X1.shape[1], dtype=np.float64)
    w1, b1 = one_step_update_multivariate(X1, y1, alpha=0.1, init_w=init_w, init_b=1.0)
    print("\nQ2.1 Code test 1:")
    print(f"  New m_1: {w1[0]}")
    print(f"  New m_2: {w1[1]}")
    print(f"  New m_3: {w1[2]}")
    print(f"  New b:   {b1}")
    
    # Q2.2 Code test 2
    X2 = np.array([[ 3, 4, 4],
               [ 4, 2, 1],
               [10, 2, 5],
               [ 3, 4, 5],
               [11, 1, 1]], dtype=np.float64)
    y2 = np.array([3, 2, 8, 4, 5], dtype=np.float64)
    init_w = np.ones(X2.shape[1], dtype=np.float64)
    w2, b2 = one_step_update_multivariate(X2, y2, alpha=0.1, init_w=init_w, init_b=1.0)
    print("\nQ2.2 Code test 2:")
    print(f"  New m_1: {w2[0]}")
    print(f"  New m_2: {w2[1]}")
    print(f"  New m_3: {w2[2]}")
    print(f"  New b:   {b2}")
    
    # Q2.3 Use normalized/standardized values for the predictors
    X_train = train_df[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_df[target_col].to_numpy(dtype=np.float64)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float64)
    y_test = test_df[target_col].to_numpy(dtype=np.float64)
    
    ws, b, train_mse, test_mse, train_ve, test_ve, iter_used = fit_and_evaluate_multivariate(X_train, y_train, X_test, y_test)
    print("\nQ2.3 Multivariate with rescaling:")
    print(f"  m={ws}")
    print(f"  b={b}")
    print(f"  Train MSE: {train_mse}, Test MSE: {test_mse}")
    print(f"  Train Variance Explained: {train_ve}, Test Variance Explained: {test_ve}")
    print(f"  Iterations used: {iter_used}")
    
    # Q2.4 Use original values for the predictors
    ws_raw, b_raw, train_mse_raw, test_mse_raw, train_ve_raw, test_ve_raw, iter_used_raw = fit_and_evaluate_multivariate(X_train, y_train, X_test, y_test, rescale_method=None, max_iter=100000)
    print("\nQ2.4 Multivariate without rescaling:")
    print(f"  m={ws_raw}")
    print(f"  b={b_raw}")
    print(f"  Train MSE: {train_mse_raw}, Test MSE: {test_mse_raw}")
    print(f"  Train Variance Explained: {train_ve_raw}, Test Variance Explained: {test_ve_raw}")
    print(f"  Iterations used: {iter_used_raw}")
    
    # Part B
    # Q1
    results_raw, pvals_raw = run_multivariate_ols_analysis(X_train, y_train, X_test, y_test, feature_cols, method_tag='raw')
    results_scaled, pvals_scaled = run_multivariate_ols_analysis(X_train, y_train, X_test, y_test, feature_cols, method_tag='standardization')
    results_log, pvals_log = run_multivariate_ols_analysis(X_train, y_train, X_test, y_test, feature_cols, method_tag='log1p')
    
    # Q2
    # Raw model:
    pval_table_raw = pvalues_for_gd_solution(X_train, y_train, ws_raw, b_raw, feature_names=list(feature_cols), robust=None)
    print("\nP-values for my GD model (raw features):")
    print(pval_table_raw.to_string(float_format=lambda v: f"{v:.6g}"))
    
    # Generate and save loss curve for GD with standardized features
    X_train_std, _, _ = rescale_data(X_train, X_train, method='standardization')

    _, _, loss_hist = gd_multivariate_with_history(X_train_std, y_train, alpha=0.05, max_iter=5000, tolerance=1e-6, patience=200)
    png_path = save_loss_curve(
        loss_hist,
        out_path="gd_loss_curve.png",
        title="GD Multivariate (Standardized) — Training MSE per Iteration"
    )
    print(f"Saved loss curve to {png_path}")

if __name__ == "__main__":
    main()
