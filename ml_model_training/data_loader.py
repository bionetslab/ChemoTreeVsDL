"""
Data loading and processing module for ML/NN training.
"""

import pickle
from re import sub
from typing import Tuple, Optional, Any, List

from utils.preprocessing_utils import fit_transform_gender, oversample_minority_with_groups
from feature_processing.ml_feature_matrix_builder import FeatureExtractorFactory
from utils.io_utils import load_pickle, save_pickle, fold_file


class DataLoader:
    """Handles data loading and processing for machine learning models."""
    
    def __init__(self, 
                 target_cohort: str,
                 feature_combination_method: str,
                 training_data_type: str,
                 feature_selection_boolen: bool,
                 oversampling_method: str,
                 saved_data_path: str,
                 paths: Any,
                 feat_type: str = 'standard',
                 agg_interval: int = 24):
        """
        Initialize the data loader.
        
        Args:
            target_cohort: Name of the target cohort
            feature_combination_method: Method for combining features
            training_data_type: Type of training data
            feature_selection_boolen: Whether to apply feature selection
            oversampling_method: Method for oversampling
            saved_data_path: Path to saved data
            paths: Path configuration object
            feat_type: Feature type ('standard' for traditional, 'V', 'M', 'D', 'VD', 'VM', 'MD', 'VMD' for timeseries)
            agg_interval: Aggregation interval in hours (default: 24)
        """
        self.target_cohort = target_cohort
        self.feature_combination_method = feature_combination_method
        self.training_data_type = training_data_type
        self.feature_selection_boolen = feature_selection_boolen
        self.oversampling_method = oversampling_method
        self.saved_data_path = saved_data_path
        self.paths = paths
        self.feat_type = feat_type
        self.agg_interval = agg_interval
        
        #set feature_extractor_type based on feat_type
        if feat_type == 'standard':
            self.feature_extractor_type = 'traditional'
        else:
            # For timeseries features, use 'timeseries' extractor
            self.feature_extractor_type = 'timeseries'
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractorFactory.create_extractor(self.feature_extractor_type)
    

    def load_and_process_data(self, fold: int, load_training_data_boolen: bool = False) -> Optional[Tuple]:
        """
        Load and process training data for a specific fold.
        
        Args:
            fold: Fold number for cross-validation
            load_training_data_boolen: Whether to load pre-saved data
            
        Returns:
            Tuple of (X_train, Y_train, X_test, Y_test, X_val, Y_val, subj_ids_train)
        """
        if load_training_data_boolen:
            return self._load_pre_saved_data(fold)
        else:
            return self._extract_and_process_data(fold)
    
    def _load_pre_saved_data(self, fold: int) -> Optional[Tuple]:
        """Load previously saved training data."""
        print("Loading previously saved training data...")
        base = self.paths.training
        x_train_path = fold_file(base, 'X_train', fold)
        
        if not x_train_path.exists():
            print('Training data not found!')
            return None
        # Load all pre-saved data
        X_train = load_pickle(fold_file(base, 'X_train', fold))
        Y_train = load_pickle(fold_file(base, 'Y_train', fold))
        X_test = load_pickle(fold_file(base, 'X_test', fold))
        Y_test = load_pickle(fold_file(base, 'Y_test', fold))
        X_val = load_pickle(fold_file(base, 'X_val', fold))
        Y_val = load_pickle(fold_file(base, 'Y_val', fold))
        subj_ids_train = load_pickle(fold_file(base, 'Sub_train', fold))
        subj_ids_val = load_pickle(fold_file(base, 'Sub_val', fold))
        
        return (X_train, Y_train, X_test, Y_test, X_val, Y_val, subj_ids_train, subj_ids_val)

    def _extract_and_process_data(self, fold: int) -> Tuple:
        """Extract and process fresh data from folds."""
        # Load fold data
        with open(self.paths.folds / f'fold_{fold}.pkl', 'rb') as f:
            train_ids, val_ids, test_ids = pickle.load(f)
            
        print(f'Feature extractor type: {self.feature_extractor_type}')
        print(f'Feature type: {self.feat_type}')
        if self.feature_selection_boolen:
            print(f'Feature selection applied (Top 100 features in cancer chemo cohort), Threshold: 100')
        # Extract features and labels for training split first (to get itemids and bins)
        X_train, Y_train, subj_ids_train, hadm_ids_train, train_itemids, train_bins = self._extract_data_split(train_ids, "training", fold)
        # Extract features for test and val splits using training itemids and bins
        X_test, Y_test, subj_ids_test, hadm_ids_test = self._extract_data_split(test_ids, "test", fold, itemids=train_itemids, bins=train_bins)
        X_val, Y_val, subj_ids_val, hadm_ids_val = self._extract_data_split(val_ids, "validation", fold, itemids=train_itemids, bins=train_bins)
        
        
        # Display dataset information
        #self._display_dataset_info(X_train, Y_train, X_test, Y_test, X_val, Y_val)
        
        # Apply oversampling if needed
        #if self.oversampling_method == 'minority':
        #    X_train, Y_train, subj_ids_train = self._apply_minority_oversampling(X_train, Y_train, subj_ids_train)
        
        # Validate data for leakage
        self._validate_data_leakage(train_ids, test_ids)
        self._validate_data_leakage(val_ids, test_ids)
        
        # Apply gender transformation if needed
        if "DEMO" in self.training_data_type:
            X_train, X_test, X_val = fit_transform_gender(X_train, X_test, X_val)

        # Save processed data AFTER all transformations
        self._save_processed_data(X_train, Y_train, X_test, Y_test, X_val, Y_val, subj_ids_train, subj_ids_val, fold)
        
        return X_train, Y_train, X_test, Y_test, X_val, Y_val, subj_ids_train, subj_ids_val
    
    def _extract_data_split(self, ids, split_name: str, fold: int, itemids=None, bins=None):
        """Extract data for a specific split (train/test/val)."""
        #print(f'Extracting {split_name} data using {self.feature_extractor_type} extractor.')
        #print(f'Feature type: {self.feat_type}')
        
        result = self.feature_extractor.extract_features(
            target_cohort=self.target_cohort,
            ids=ids,
            feature_combination_method=self.feature_combination_method,
            training_data_types=self.training_data_type,
            fold=fold,
            feature_threshold=self.feature_selection_boolen,
            saved_data_path=self.saved_data_path,
            feat_type=self.feat_type,
            agg_interval=self.agg_interval,
            itemids=itemids,
            bins=bins
        )
        
        return result
    
    def _display_dataset_info(self, X_train, Y_train, X_test, Y_test, X_val, Y_val):
        """Display information about the dataset."""
        print(f'Dataset sizes - Train: {len(Y_train)}, Test: {len(Y_test)}, Validation: {len(Y_val)}')
        # Data loading completed
        print(f'Training data shape: {X_train.shape}')
        print('Training data preview:')
        print(X_train.head())
    
    
    def _validate_data_leakage(self, train_ids, test_ids):
        """Validate that there's no data leakage between train and test sets."""
        train_patients = set(train_ids[:, 0])
        test_patients = set(test_ids[:, 0])
        train_admissions = set(train_ids[:, 1])
        test_admissions = set(test_ids[:, 1])
        
        if train_patients.intersection(test_patients):
            print('WARNING: There are common patients in test and train sets!')
        
        if train_admissions.intersection(test_admissions):
            print('WARNING: There are common admissions in test and train sets!')
    
    def _save_processed_data(self, X_train, Y_train, X_test, Y_test, X_val, Y_val, subj_ids_train, subj_ids_val, fold):
        """Save processed data for future use."""
        base = self.paths.training
        save_pickle(X_train, fold_file(base, 'X_train', fold))
        save_pickle(Y_train, fold_file(base, 'Y_train', fold))
        save_pickle(X_test, fold_file(base, 'X_test', fold))
        save_pickle(Y_test, fold_file(base, 'Y_test', fold))
        save_pickle(X_val, fold_file(base, 'X_val', fold))
        save_pickle(Y_val, fold_file(base, 'Y_val', fold))
        save_pickle(subj_ids_train, fold_file(base, 'Sub_train', fold))
        save_pickle(subj_ids_val, fold_file(base, 'Sub_val', fold))


# Convenience function for backward compatibility
def load_and_process_data(target_cohort: str,
                         feature_combination_method: str,
                         training_data_type: str,
                         feature_selection_boolen: bool,
                         oversampling_method: str,
                         saved_data_path: str,
                         paths: Any,
                         fold: int,
                         load_training_data_boolen: bool = False,
                         feat_type: str = 'standard') -> Optional[Tuple]:
    """
    Convenience function for loading and processing data.
    
    This function creates a DataLoader instance and calls load_and_process_data.
    Useful for backward compatibility or when you don't need to reuse the loader.
    
    Args:
        feat_type: Feature type ('standard' for traditional, 'V', 'M', 'D', 'VD', 'VM', 'MD', 'VMD' for timeseries)
    """
    loader = DataLoader(
        target_cohort=target_cohort,
        feature_combination_method=feature_combination_method,
        training_data_type=training_data_type,
        feature_selection_boolen=feature_selection_boolen,
        oversampling_method=oversampling_method,
        saved_data_path=saved_data_path,
        paths=paths,
        feat_type=feat_type
    )
    
    return loader.load_and_process_data(fold, load_training_data_boolen)
