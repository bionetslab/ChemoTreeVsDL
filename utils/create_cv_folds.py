import os
import pickle
import argparse
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import StratifiedKFold, train_test_split
from config.constants import PROJECT_ROOT, RANDOM_SEED, num_folds

def split_cv_folds (target_cohort:str, num_folds:int, saved_data_path:str, seed: int = None)  -> Tuple[np.ndarray, np.ndarray, np.ndarray] :

    cohort = pd.read_csv(f'{saved_data_path}/cohorts/'+target_cohort+'.csv.gz', compression='gzip', parse_dates = ['admittime','dischtime'])

    subject_labels = cohort.groupby("subject_id")['label'].max().reset_index()

    effective_seed = RANDOM_SEED if seed is None else seed
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state= effective_seed)
    for fold, (train_idx, test_idx) in enumerate(skf.split(subject_labels, subject_labels['label'])):
        test = subject_labels.loc[test_idx]
        train = subject_labels.loc[train_idx]

        # split validation
        train_idx, val_idx = train_test_split(train.index, test_size=0.2, random_state=effective_seed, stratify=train['label'])
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['MIMIC_IV', 'UKEr'])
    parser.add_argument('--cohorts', nargs='+', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[None])
    args = parser.parse_args()

    saved_data_path = PROJECT_ROOT / args.dataset / 'saved_data'
    for cohort in args.cohorts:
        for seed in args.seeds:
            split_cv_folds(cohort, num_folds, saved_data_path, seed)


if __name__ == "__main__":
    main()
