
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def preprocess_data(df: pd.DataFrame, target_col: str, test_size: float, random_state: int) -> Tuple:

    print("2. Preprocessing data...")
    
  
    df_clean = df.dropna(subset=[target_col]).copy()
    print(f"   - Shape after dropping NaNs in target: {df_clean.shape}")


    df_clean[target_col] = df_clean[target_col].apply(lambda x: 1 if x.lower() == 'positive' else 0)

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    

    X_log = np.log2(X.astype(float) + 1)
    

    X_train_log, X_test_log, y_train, y_test = train_test_split(
        X_log, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y 
    )
    

    scaler = StandardScaler()
    

    scaler.fit(X_train_log) 
    

    X_train_scaled = pd.DataFrame(scaler.transform(X_train_log), index=X_train_log.index, columns=X_train_log.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_log), index=X_test_log.index, columns=X_test_log.columns)
    

    X_scaled_full = pd.DataFrame(scaler.transform(X_log), index=X_log.index, columns=X_log.columns)

    print(f"   - Training set shape: {X_train_scaled.shape}")
    print(f"   - Test set shape: {X_test_scaled.shape}")


    return X_train_scaled, X_test_scaled, y_train, y_test, X_scaled_full 