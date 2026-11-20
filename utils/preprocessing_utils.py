import pandas as pd
import numpy as np
import os
from typing import Tuple,List
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
#os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

def oversample_minority_with_groups(X_df: pd.DataFrame, y: pd.Series, group_series: pd.Series, seed: int = 42):
    """Oversample the minority class while preserving group alignment via a temporary column."""
    Xy = pd.concat([
        X_df.reset_index(drop=True),
        pd.Series(group_series, name="group").reset_index(drop=True)
    ], axis=1)

    oversample = RandomOverSampler(sampling_strategy='minority', random_state=seed)
    Xy_res, y_res = oversample.fit_resample(Xy, y)

    groups_res = Xy_res["group"]
    X_res = Xy_res.drop(columns=["group"])
    return X_res, y_res, groups_res


def fit_transform_gender(train_df: pd.DataFrame, *other_dfs: pd.DataFrame):
    """Fit a LabelEncoder on train_df['gender'] and transform in provided dataframes.
    Returns the transformed dataframes in the same order.
    """
    enc = LabelEncoder()
    enc.fit(train_df['gender'])
    train_df.loc[:, 'gender'] = enc.transform(train_df['gender'])
    train_df['gender'] = train_df['gender'].astype('int64')
    transformed = []
    for df in other_dfs:
        df.loc[:, 'gender'] = enc.transform(df['gender'])
        df['gender'] = df['gender'].astype('int64') 
        transformed.append(df)
    return (train_df, *transformed)


def fit_transform_scaler(X_train, X_test, X_val=None):
    """Fit StandardScaler on training data and transform all datasets.
    
    Args:
        X_train: Training features
        X_test: Test features  
        X_val: Validation features (optional)
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, X_val_scaled)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_test_scaled, X_val_scaled
    else:
        return X_train_scaled, X_test_scaled
