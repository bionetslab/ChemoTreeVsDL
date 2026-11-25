"""Result paths of the old FLabNet pipeline, as used in its analysis notebook (analyse_results.ipynb,
analyse_feature_importance.ipynb). Only used to test the plot scripts on the old results:

    OLD_RESULTS_HOME=/home/jovyan/data_common/FLabNet-pipeline python -m plots.<script>

Each new experiment prefix is mapped to the old run that the paper figures were made from.
"""
import os
from pathlib import Path

OLD_RESULTS_HOME = os.environ.get("OLD_RESULTS_HOME")

# new model keys -> old ML file names
ML_OLD_NAMES = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
                "gradient_boosting": "Gradient Boosting", "xgboost": "Xgboost", "catboost": "CatBoost"}
ML_NEW_KEYS = {v: k for k, v in ML_OLD_NAMES.items()}


def _base(dataset, seed=None):
    base = Path(OLD_RESULTS_HOME) / dataset / "saved_data"
    return base / "multi_seed_training" if seed is not None else base


# grid search folder names of the old ML results on disk (older notebook/code versions wrote grid_True / grid_False)
GRID_DIRS = {True: "grid_nested", False: "grid_False"}


def _ml_dir(dataset, cohort, grid, feat_type, agg_interval=None, seed=None, outputs=False):
    fs = dataset == "MIMIC_IV"
    path = (_base(dataset, seed) / ("outputs" if outputs else "results") / cohort / "non_time_series" / "concatenate"
            / "minority" / f"feature_selection_{fs}" / GRID_DIRS[grid] / f"feat_type_{feat_type}")
    if agg_interval is not None:
        path = path / f"agg_interval_{agg_interval}h"
    if seed is not None:
        path = path / f"seed_{seed}"
    return path


def ml_results_file(dataset, cohort, model, prefix, seed=None):
    name = f"{ML_OLD_NAMES[model]}_results.csv"
    if prefix == "full_run":
        return _ml_dir(dataset, cohort, True, "VMD") / "250925" / name
    if prefix.startswith("feat_type_"):
        ft = prefix[len("feat_type_"):]
        return _ml_dir(dataset, cohort, True, "standard" if ft == "V" else ft) / "250925" / name
    if prefix.startswith("agg_int_"):
        return _ml_dir(dataset, cohort, True, "VMD", agg_interval=int(prefix[len("agg_int_"):])) / "101025" / name
    if prefix == "multi_seed":
        return _ml_dir(dataset, cohort, False, "VMD", agg_interval=24, seed=seed) / "101025" / name
    raise ValueError(f"No old path for ML prefix {prefix}")


def ml_feature_importance_file(dataset, cohort, model, fold):
    return (_ml_dir(dataset, cohort, True, "VMD", agg_interval=24, outputs=True) / "231125"
            / f"{ML_OLD_NAMES[model]}_feature_importance_fold{fold}.csv")


def ml_tm_file(dataset, cohort, model):
    return _ml_dir(dataset, cohort, False, "VMD", agg_interval=24) / "time_memory_ml_test" / f"{ML_OLD_NAMES[model]}_tm.csv"


def ts_results_file(dataset, cohort, model, prefix, fold, grid, seed=None, train_mode="standard"):
    dd = "uker" if dataset == "UKEr" else "mimic"
    old_prefix = {
        "full_run": "081025",
        "drop_minutes": "drop_minutes_191025",
        "tm_gpu": "tm_gpu_271025",
        "tm_cpu": "tm_cpu_271025",
        "pt_none": "pt_161025_none",
        "pt_all_freeze": f"pt_191025_all_{dd}_freeze",
        "pt_all_all_weights": f"pt_191025_all_{dd}_all_weights",
        "pt_cohort_freeze": "pt_161025_cohort_freeze",
        "multi_seed": "multi_seed_091025",
    }.get(prefix)
    if prefix.startswith("agg_int_"):
        old_prefix = f"{prefix}_191025"
    elif prefix.startswith("feat_type_"):
        old_prefix = f"221025_{prefix}"
    elif prefix == "pt_cohort_all_weights":
        # the notebook took this run from the multi-seed training (seed 42)
        seed, old_prefix = 42, "multi_seed_091025_pt_161025_cohort_all_weights"
    if old_prefix is None:
        raise ValueError(f"No old path for TS prefix {prefix}")

    path = _base(dataset, seed) / "results" / cohort / "time_series" / train_mode / model
    if seed is not None:
        path = path / f"seed_{seed}"
    return path / old_prefix / f"fold_{fold}" / f"grid_{grid}" / "results_final.csv"
