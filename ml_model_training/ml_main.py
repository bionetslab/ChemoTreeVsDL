"""
Main entry point for ML model training.
"""

import argparse
import pandas as pd
import numpy as np
import random
from datetime import datetime
from pathlib import Path
from config.constants import PROJECT_ROOT, RANDOM_SEED
from utils.io_utils import build_paths
from ml_model_training.data_loader import DataLoader
from ml_model_training.ml_trainer import MLTrainer
from ml_model_training.ml_evaluator import Result_evaluation
from utils.preprocessing_utils import oversample_minority_with_groups
from ts_model_training.logger import Logger

# Set all random seeds for maximum reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class MLTrainingPipeline:
    """Main pipeline for ML model training."""
    
    def __init__(self, target_cohort: str, saved_data_path: str, model_type: str,
                 feature_combination_method: str, oversampling_method: str,
                 feature_selection_boolen: bool, training_data_type: list,
                 load_training_data_boolen: bool, grid_search_boolen: bool,
                 num_folds: int = 5,
                 feat_type: str = 'VMD',
                 agg_interval: int = 24,
                 days_before_discharge: int = 14,
                 prefix: str = '',
                 random_seed: int = None,
                 action: str = 'train'):
        
        # Store configuration
        self.target_cohort = target_cohort
        self.saved_data_path = saved_data_path
        self.model_type = model_type
        self.feature_combination_method = feature_combination_method
        self.oversampling_method = oversampling_method
        self.feature_selection_boolen = feature_selection_boolen
        self.training_data_type = training_data_type
        self.load_training_data_boolen = load_training_data_boolen
        self.grid_search_boolen = grid_search_boolen
        self.num_folds = num_folds
        self.feat_type = feat_type
        self.agg_interval = agg_interval
        self.days_before_discharge = days_before_discharge
        self.prefix = prefix
        self.random_seed = random_seed
        
        # Build paths
        base = Path(self.saved_data_path)
        self.paths = build_paths(
            base=base,
            cohort=self.target_cohort,
            feature_method=self.feature_combination_method,
            oversampling=self.oversampling_method,
            feature_selection=self.feature_selection_boolen,
            grid_search=self.grid_search_boolen,
            feat_type=self.feat_type,
            agg_interval=self.agg_interval,
            prefix=self.prefix,
            random_seed=self.random_seed
        )

        # Log file saved alongside the run's outputs (same Logger as the TS pipeline)
        if action == 'prepare':
            self.logger = Logger(str(self.paths.training), 'prepare_log.txt')
        else:
            self.logger = Logger(str(self.paths.results), f'{self.model_type}_log.txt')
        self.log = self.logger.write
        self.log(f"\n########## New run: action={action}, model={self.model_type} ##########")
        self.log(f"Log file: {self.logger.log}")

        # Initialize components
        self.data_loader = DataLoader(
            target_cohort=self.target_cohort,
            feature_combination_method=self.feature_combination_method,
            training_data_type=self.training_data_type,
            feature_selection_boolen=self.feature_selection_boolen,
            oversampling_method=self.oversampling_method,
            saved_data_path=self.saved_data_path,
            paths=self.paths,
            feat_type=self.feat_type,
            agg_interval=self.agg_interval,
            days_before_discharge=self.days_before_discharge,
            logger=self.logger
        )

        self.trainer = MLTrainer(
            model_type=self.model_type,
            feature_combination_method=self.feature_combination_method,
            grid_search=self.grid_search_boolen,
            results_path=str(self.paths.outputs),
            cohort=self.target_cohort,
            logger=self.logger
        )

        self.evaluator = Result_evaluation(logger=self.logger)


    def _print_configuration(self):
        """Log training configuration."""
        rows = [
            ("Target Cohort", self.target_cohort),
            ("Model Type", self.model_type),
            ("Oversampling", self.oversampling_method),
            ("Feature Selection", self.feature_selection_boolen),
            ("Training Data Types", self.training_data_type),
            ("Feature Method", self.feature_combination_method),
            ("Grid Search", self.grid_search_boolen),
            ("Number of Folds", self.num_folds),
            ("Load Training Data", self.load_training_data_boolen),
            ("Temporal resolution", self.agg_interval),
            ("Feature Type", self.feat_type),
            ("Prefix", self.prefix),
        ]
        lines = ["=" * 60, "MACHINE LEARNING TRAINING CONFIGURATION", "=" * 60,
                 f"{'Parameter':<30} {'Value':<30}", "-" * 60]
        lines += [f"{name:<30} {str(value):<30}" for name, value in rows]
        lines += ["=" * 60]
        self.log("\n" + "\n".join(lines))
        self.log("Starting cross-validation...")

    def prepare_training_data(self):
        """Build feature matrices for all folds."""
        for fold in range(self.num_folds):
            self.log(f"Building feature matrices for fold {fold}...")
            data_per_fold = self.data_loader.load_and_process_data(fold, load_training_data_boolen=False)

            if data_per_fold is None:
                self.log(f'training data not build for fold {fold}!')
                break

        self.log("Feature matrices building completed!")
    
    def train(self):
        """Main training loop for machine learning models."""
        
        self._print_configuration()
        
        for fold in range(self.num_folds):
            self.log("========================== FOLD {0:2d} =========================".format(fold))

            # Load and process data
            data_per_fold = self.data_loader.load_and_process_data(fold, self.load_training_data_boolen)

            if data_per_fold is None:
                self.log(f'Training data not found for fold {fold}!')
                break
            
            X_train, Y_train, X_test, Y_test, X_val, Y_val, subj_ids_train, subj_ids_val = data_per_fold
            
            # Combine train and validation sets for tree-based models
            X_train = pd.concat([X_train, X_val], ignore_index=True)
            Y_train = pd.concat([Y_train, Y_val], ignore_index=True)
            subj_ids_train = pd.concat([subj_ids_train, subj_ids_val], ignore_index=True)
            
            
            # Apply oversampling if needed
            if self.oversampling_method == 'minority':
                self.log("Oversampling applied.")
                X_train, Y_train, subj_ids_train = oversample_minority_with_groups(
                    X_train, Y_train, subj_ids_train, seed=RANDOM_SEED
                )
                #print(f'Positive samples in train after oversampling: {Y_train.sum()}')
            
            # Train model and get results
            test_metrics, best_params = self.trainer.train_model(X_train, Y_train, X_test, Y_test, fold, subj_ids_train)
            
            self.evaluator.evaluation_metrics(test_metrics, best_params, fold, self.model_type)
            self.log(f"Fold {fold} done.")

        # Save comprehensive results
        self.evaluator.save_results(self.model_type, self.paths.results)


def main():
    """Main entry point for ML training."""
    parser = argparse.ArgumentParser(description="Train ML model")
    
    parser.add_argument('--dataset', type=str, default='MIMIC_IV', help='Dataset name')
    parser.add_argument('--cohort', type=str, default='mimic_cohort_NF_30_days', help='Target cohort')
    parser.add_argument('--model_type', type=str, 
                       choices=['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Xgboost', 'CatBoost'], 
                       default='Random Forest')
    parser.add_argument('--features', nargs='+', default=['LAB', 'DEMO'], 
                       help='List of feature types (e.g., LAB DEMO)')
    parser.add_argument('--feature_method', type=str, choices=['concatenate', 'aggregate'], 
                       default='concatenate')
    parser.add_argument('--oversampling', type=str, default='minority')
    parser.add_argument('--grid_search', type=lambda x: bool(int(x)), default=True)
    parser.add_argument('--feature_selection', type=lambda x: bool(int(x)), default=False)
    parser.add_argument('--load_data', type=lambda x: bool(int(x)), default=False)
    parser.add_argument('--feat_type', type=str, 
                       choices=["V", "M", "D", "VMD", "VM", "VD", "MD"], 
                       default="VMD")
    parser.add_argument('--agg_interval', type=int, choices=[3, 6, 12, 24], default=24,
                    help='Aggregation interval in hours (e.g., 3, 6, 12, 24)')
    parser.add_argument('--days_before_discharge', type=int, default=14)
    parser.add_argument('--num_folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--prefix', type=str, default=None,
                       help='Prefix for output files')
    parser.add_argument('--action', type=str, default='train', 
                       choices=['prepare', 'train'],
                       help='Action to perform: prepare or train')

    args = parser.parse_args()
    if args.prefix is None:
        args.prefix = datetime.now().strftime("%d%m%y")

    saved_data_path = PROJECT_ROOT / args.dataset / 'saved_data'

    # Initialize training pipeline
    pipeline = MLTrainingPipeline(
        target_cohort=args.cohort,
        saved_data_path=saved_data_path,
        model_type=args.model_type,
        feature_combination_method=args.feature_method,
        oversampling_method=args.oversampling,
        feature_selection_boolen=args.feature_selection,
        training_data_type=args.features,
        load_training_data_boolen=args.load_data,
        grid_search_boolen=args.grid_search,
        num_folds=args.num_folds,
        feat_type=args.feat_type,
        agg_interval = args.agg_interval,
        days_before_discharge=args.days_before_discharge,
        prefix=args.prefix,
        action=args.action
    )

    # Log the full command-line configuration and resolved paths
    pipeline.log("Command-line arguments:\n" + "\n".join(f"  {k}: {v}" for k, v in vars(args).items()))
    pipeline.log("Paths:\n" + "\n".join(f"  {k}: {v}" for k, v in vars(pipeline.paths).items()))

    # Execute the requested action
    if args.action == 'prepare':
        pipeline.prepare_training_data()
    elif args.action == 'train':
        pipeline.train()


if __name__ == "__main__":
    main()
