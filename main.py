import numpy as np
import pandas as pd

def main():
    data_file = 'Concrete_Data.csv'
    df = pd.read_csv(data_file, header=None)  # row 0 is header
    
    test_df = df.iloc[501:631]  # 130 rows × 9 columns
    train_df = pd.concat([df.iloc[1:501], df.iloc[631:]])  # 900 rows × 9 columns
    
    feature_cols = df.columns[:-1]  # all columns except the last one "Compressive Strength"
    target_col = df.columns[-1]  # "Compressive Strength" is the response variable.
    
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df[target_col].to_numpy(dtype=float)
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

if __name__ == "__main__":
    main()
