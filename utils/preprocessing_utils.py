import pandas as pd
import numpy as np
import pickle
import os
from typing import Tuple,List
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
#os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

def split_cv_folds (target_cohort:str, num_folds:int, saved_data_path:str, seed: int = None)  -> Tuple[np.ndarray, np.ndarray, np.ndarray] :
    
    cohort = pd.read_csv(f'{saved_data_path}/cohorts/'+target_cohort+'.csv.gz', compression='gzip', parse_dates = ['admittime','dischtime'])
    
    subject_labels = cohort.groupby("subject_id")['label'].max().reset_index()

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state= seed)
    for fold, (train_idx, test_idx) in enumerate(skf.split(subject_labels, subject_labels['label'])):
        test = subject_labels.loc[test_idx]
        train = subject_labels.loc[train_idx]

        # split validation
        train_idx, val_idx = train_test_split(train.index, test_size=0.2, random_state=seed, stratify=train['label'])
        val = subject_labels.loc[val_idx]
        train = subject_labels.loc[train_idx]

        train_sub = list(train['subject_id'])
        val_sub = list(val['subject_id'])
        test_sub = list(test['subject_id'])

        train_hadms = np.array(cohort.loc[cohort['subject_id'].isin(train_sub),['subject_id','hadm_id']])
        val_hadms = np.array(cohort.loc[cohort['subject_id'].isin(val_sub),['subject_id','hadm_id']])
        test_hadms = np.array(cohort.loc[cohort['subject_id'].isin(test_sub),['subject_id','hadm_id']])

        if seed:
            save_path = f'{saved_data_path}/multi_seed_training/folds/{target_cohort}/seed_{seed}'
        else:
            save_path = f'{saved_data_path}/folds/{target_cohort}'
            
            
        os.makedirs(save_path, exist_ok=True)

        pickle.dump([train_hadms, val_hadms, test_hadms], 
                    open(os.path.join(save_path, f'fold_{fold}.pkl'),'wb'))
        
        
        common_hadm_ids = set(train_hadms[:, 1]).intersection(set(test_hadms[:, 1]))
        common_subject_ids = set(train_hadms[:, 0]).intersection(set(test_hadms[:, 0]))
        if common_hadm_ids:
            print('There are common admissions in test and train')
        if common_subject_ids:
            print('There are common patients in test and train')
            
    print("Splitied folds saved!")
    return train_hadms, val_hadms, test_hadms

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


def select_fold_top_features(target_cohort: str, input_features: str, saved_data_path: str, 
                           num_features: int = 100, num_folds: int = 5) -> None:
    """Select top features per fold based on frequency in positive cases.
    
    This function analyzes the frequency of lab features in positive cases for each fold
    and saves the top N features for each fold.
    
    Args:
        target_cohort: Name of the target cohort
        input_features: Name of the input features file
        saved_data_path: Path to saved data directory
        num_features: Number of top features to select per fold (default: 100)
        num_folds: Number of CV folds (default: 5)
    """
    # Load cohort and labs data
    cohort = pd.read_csv(f'{saved_data_path}/cohorts/{target_cohort}.csv.gz',
                        compression='gzip', parse_dates=['admittime', 'dischtime'])
    labs = pd.read_csv(f'{saved_data_path}/features/{input_features}.csv.gz',
                      compression='gzip', header=0)
    
    # Create output directory
    save_path = f'{saved_data_path}/top_features/{target_cohort}/'
    os.makedirs(save_path, exist_ok=True)
    
    # Merge labs with labels
    labs_with_labels = labs.merge(
        cohort[['hadm_id', 'label']],
        on='hadm_id',
        how='left'
    )
    pos_labs = labs_with_labels[labs_with_labels['label'] == 1]
    
    # Process each fold
    CV_folds_path = f'{saved_data_path}/folds/{target_cohort}'
    
    for fold in range(num_folds):
        # Load fold data
        with open(os.path.join(CV_folds_path, f'fold_{fold}.pkl'), 'rb') as f:
            train_ids, val_ids, test_ids = pickle.load(f)
        
        # Combine train and validation for feature selection
        train_hids = np.vstack([train_ids, val_ids])
        fold_pos_labs = pos_labs[pos_labs['hadm_id'].isin(train_hids[:, 1])]
        
        # Remove duplicates and calculate frequency
        fold_pos_labs = fold_pos_labs.drop_duplicates(subset=['itemid', 'hadm_id'])
        freq = (
            fold_pos_labs.groupby('itemid')['hadm_id']
            .nunique()
            .reset_index(name='unique_admissions')
            .sort_values(by='unique_admissions', ascending=False)
        )
        
        # Select top features
        selected_features = freq[:num_features]['itemid'].tolist()
        
        # Save selected features for this fold
        with open(os.path.join(save_path, f'fold_{fold}_features.pkl'), 'wb') as out_f:
            pickle.dump(selected_features, out_f)
    
    print(f"Top {num_features} features selected and saved for {num_folds} folds!")