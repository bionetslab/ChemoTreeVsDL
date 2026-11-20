"""
Feature extraction utilities for different data representations.

"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Tuple, List, Optional, Dict, Any
from abc import ABC, abstractmethod
from config.constants import TOP_FEATURES_PATH


class BaseFeatureExtractor(ABC):
    """Base class for feature extractors."""
    
    @abstractmethod
    def extract_features(self, target_cohort: str, ids: np.ndarray, 
                        feature_combination_method: str, training_data_types: List[str], 
                        fold: int, feature_threshold: bool, saved_data_path: str,
                        feat_type: str = None, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Extract features for given admission IDs.
        
        Returns:
            X_df: Feature matrix
            y_df: Labels
            subj_ids: Subject IDs
            hadm_ids: Admission IDs
        """
        pass


class TimeSeriesExtractor(BaseFeatureExtractor):
    """Time series feature extraction (original getXplus approach)."""
    
    def extract_features(self, target_cohort: str, ids: np.ndarray, 
                        feature_combination_method: str, training_data_types: List[str], 
                        fold: int, feature_threshold: bool, saved_data_path: str,
                        feat_type: str = "VMD", itemids: list = None, bins: list = None, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Extract time series features (VMD - Values, Missingness, Delta)."""
        
        all_raw = []
        all_demo = []
        hadm_ids = []
        subject_ids = []
        agg_interval = kwargs.get('agg_interval', 24)

        for subject_id, hadm_id in ids:
            raw_path = f"{saved_data_path}/processed_admission_features_csv/{target_cohort}/agg_interval_{agg_interval}h/{hadm_id}/raw_labs.csv"
            if not os.path.exists(raw_path):
                continue
            all_raw.append(pd.read_csv(raw_path, header=0))
            all_demo.append(pd.read_csv(f"{saved_data_path}/processed_admission_features_csv/{target_cohort}/agg_interval_{agg_interval}h/{hadm_id}/demo.csv", header=0))
            hadm_ids.append(hadm_id)
            subject_ids.append(subject_id)
            
        if not all_raw:
            return pd.DataFrame(), pd.DataFrame(), pd.Series([], name="subject_id"), pd.Series([], name="hadm_id")
            
        data = pd.concat(all_raw, ignore_index=True).rename(columns={"int": "bin"})
        demo = pd.concat(all_demo, keys=hadm_ids).reset_index(level=1, drop=True)
        
        labels = pd.read_csv(f'{saved_data_path}/processed_admission_features_csv/{target_cohort}/agg_interval_{agg_interval}h/labels.csv', header=0)
        y_df = labels.set_index("hadm_id").loc[hadm_ids, "label"].reset_index(drop=True)

        if feature_threshold:
            with open(TOP_FEATURES_PATH, 'rb') as f:
                selected_features = pickle.load(f)
            #print("Feature selection applied (Top 100 features in cancer chemo cohort), Threshold:", len(selected_features))
            data = data[data["itemid"].isin(selected_features)]

        # Extract itemids and bins from training data or use provided ones
        if itemids is None and bins is None:
            # Training: extract itemids and bins
            x_df, m_df, delta_df, all_itemids, all_bins = self._compute_vmd_features(data, target_cohort, return_itemids_bins=True)
        else:
            # Test/Val: use provided itemids and bins
            x_df, m_df, delta_df = self._compute_vmd_features(data, target_cohort, itemids=itemids, bins=bins)


        # Reindex to ensure correct admission order
        x_df = x_df.reindex(hadm_ids)
        m_df = m_df.reindex(hadm_ids)
        delta_df = delta_df.reindex(hadm_ids)

        if feature_combination_method == "concatenate":
            x_df.columns = [f"X_{itemid}_{bin}" for (bin, itemid) in x_df.columns]
            m_df.columns = [f"M_{itemid}_{bin}" for (bin, itemid) in m_df.columns]
            delta_df.columns = [f"D_{itemid}_{bin}" for (bin, itemid) in delta_df.columns]

        else:
            x_df = x_df.groupby(axis=1, level="itemid").mean()
            x_df.columns = [f"X_{itemid}" for itemid in x_df.columns]
            m_df = m_df.groupby(axis=1, level="itemid").mean()
            m_df.columns = [f"M_{itemid}" for itemid in m_df.columns]
            delta_df = delta_df.groupby(axis=1, level="itemid").mean()
            delta_df.columns = [f"D_{itemid}" for itemid in delta_df.columns]

        # Combine features based on feat_type
        X_df = self._combine_feature_types(x_df, m_df, delta_df, demo, feat_type)
        if itemids is None and bins is None:
            # Training: return itemids and bins
            return X_df, y_df, pd.Series(subject_ids, name="subject_id"), pd.Series(hadm_ids, name="hadm_id"), all_itemids, all_bins
        else:
            # Test/Val: return only features
            return X_df, y_df, pd.Series(subject_ids, name="subject_id"), pd.Series(hadm_ids, name="hadm_id")
    
    def _compute_vmd_features(self, data: pd.DataFrame, target_cohort, return_itemids_bins=False, itemids=None, bins=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute Values, Missingness, and Delta features from raw lab data."""
        
        all_admissions = data["hadm_id"].unique()
        
        # Use provided itemids and bins or extract from current data
        if itemids is not None and bins is not None:
            all_items = itemids
            all_bins = bins
        else:
            all_items = data["itemid"].unique()
            all_bins = sorted(data["bin"].unique())
        # In uker dataset some splits have missing itemids add them for consistency
        '''if "uker" in target_cohort:
            expected_items = [i for i in range(1, 49) if i != 39] # based on training data
            missing_items = set(expected_items) - set(all_items)
            if missing_items:
                print(f"Adding missing item IDs: {sorted(missing_items)}")
                all_items = sorted(set(all_items) | set(expected_items))'''
    
        # X: average values within bins
        x_values = (
            data.groupby(["hadm_id", "itemid", "bin"])["value"]
            .mean()
        )
        
        # Ensure all bins are present before unstack
        x_values = x_values.reindex(
            pd.MultiIndex.from_product([all_admissions, all_items, all_bins], names=["hadm_id", "itemid", "bin"])
        )
        
        x_values = x_values.unstack(level="bin")
        
        # Impute missing values (forward/backward fill) after adding missing bins
        x_values = x_values.ffill(axis=1).bfill(axis=1)
        
        # Fill remaining missing values with 0
        x_values = x_values.fillna(0)
        x_df = x_values.unstack(level="itemid")
        
        # M: missingness
        m_values = (
            data.groupby(["hadm_id", "itemid", "bin"])["value"]
            .count()
        )
        
        # Ensure all bins are present before unstack
        m_values = m_values.reindex(
            pd.MultiIndex.from_product([all_admissions, all_items, all_bins], names=["hadm_id", "itemid", "bin"])
        )
        
        m_values = m_values.unstack(level="bin")
        
        # Fill missing values with 0 (no observations = 0 count)
        m_values = m_values.fillna(0).astype(int)
        m_df = m_values.unstack(level="itemid")
        
        # Delta: time since last observed
        delta = np.zeros_like(m_values.values, dtype=float)
        delta[:, 0] = 1 - m_values.values[:, 0]
        
        for t in range(1, m_values.shape[1]):
            prev = delta[:, t - 1]
            obs = m_values.values[:, t]
            delta[:, t] = np.where(obs == 1, 0, 1 + prev)
        
        delta = delta / m_values.shape[1]
        delta_df = pd.DataFrame(delta, index=m_values.index, columns=m_values.columns)
        delta_df = delta_df.unstack(level="itemid")
        
        if return_itemids_bins:
            extracted_itemids = sorted(list(all_items))
            extracted_bins = sorted(list(all_bins))
            return x_df, m_df, delta_df, extracted_itemids, extracted_bins
        else:
            return x_df, m_df, delta_df
    
    def _combine_feature_types(self, x_df: pd.DataFrame, m_df: pd.DataFrame, delta_df: pd.DataFrame, 
                             demo: pd.DataFrame, feat_type: str) -> pd.DataFrame:
        """Combine different feature types based on feat_type parameter."""
        if feat_type == "V":
            return pd.concat([x_df, demo], axis=1).reset_index(drop=True)
        elif feat_type == "M":
            return pd.concat([m_df, demo], axis=1).reset_index(drop=True)
        elif feat_type == "D":
            return pd.concat([delta_df, demo], axis=1).reset_index(drop=True)
        elif feat_type == "VD":
            return pd.concat([x_df, delta_df, demo], axis=1).reset_index(drop=True)
        elif feat_type == "VM":
            return pd.concat([x_df, m_df, demo], axis=1).reset_index(drop=True)
        elif feat_type == "MD":
            return pd.concat([m_df, delta_df, demo], axis=1).reset_index(drop=True)
        elif feat_type == "VMD":
            return pd.concat([x_df, m_df, delta_df, demo], axis=1).reset_index(drop=True)
        else:
            raise ValueError(
                f"Invalid feat_type: {feat_type}. "
                "Must be one of:'V', 'M', 'D', 'VD', 'VM', 'MD', 'VMD'."
            )




class FeatureExtractorFactory:
    """Factory class to create appropriate feature extractors."""
    
    _extractors = {
        'timeseries': TimeSeriesExtractor,
    }
    
    @classmethod
    def create_extractor(cls, extractor_type: str) -> BaseFeatureExtractor:
        """Create a feature extractor of the specified type."""
        if extractor_type not in cls._extractors:
            raise ValueError(f"Unknown extractor type: {extractor_type}. "
                           f"Available types: {list(cls._extractors.keys())}")
        
        return cls._extractors[extractor_type]()
    
    @classmethod
    def get_available_extractors(cls) -> List[str]:
        """Get list of available extractor types."""
        return list(cls._extractors.keys())
