"""Read training results from saved_data for the paper tables and figures.

ML runs (training/ml/ml_main.py) write one results_final.csv per run with one row per fold plus an "Average" row:
    saved_data/<dataset>/results/<cohort>/non_time_series/<model>/<prefix>/results_final.csv
TS runs (training/ts/main.py) write one results_final.csv per fold with a "split" column (rows are appended on reruns):
    saved_data/<dataset>/results/<cohort>/time_series/<train_mode>/<model>/<prefix>/fold_<k>/grid_<grid>/results_final.csv
Multi-seed runs are under saved_data/<dataset>/multi_seed_training/results/..., with seed_<n>/ before <prefix>.

Missing result files are skipped (as in the original analysis notebooks) and listed by report_missing().
With OLD_RESULTS_HOME set, results are read from the old pipeline instead (see plots/old_paths.py).
"""
from pathlib import Path

import pandas as pd

from config.constants import SAVED_DATA_DIR
from plots import old_paths

FIGURES_DIR = SAVED_DATA_DIR / "figures"
NUM_FOLDS = 5

# names and orders as in the original analysis notebook (analyse_results.ipynb)
exps = [("MIMIC_IV", "mimic_cohort_aplasia_45_days"), ("MIMIC_IV", "mimic_cohort_NF_30_days"),
        ("UKEr", "uker_cohort_aplasia_45_days"), ("UKEr", "uker_cohort_NF_30_days")]
cohorts = ['mimic_cohort_aplasia_45_days', 'uker_cohort_aplasia_45_days', 'mimic_cohort_NF_30_days', 'uker_cohort_NF_30_days']
cohort_names = {
    "mimic_cohort_NF_30_days": "Neutropenic fever (MIMIC-IV)",
    "mimic_cohort_aplasia_45_days": "Aplasia (MIMIC-IV)",
    "uker_cohort_NF_30_days": "Neutropenic fever (UKEr)",
    "uker_cohort_aplasia_45_days": "Aplasia (UKEr)"
}
models = ["mlp", "gru", "lstm", "tcn", "sand", "grud", "interpnet", "strats"]
tree_models = ["logistic_regression", "random_forest", "xgboost", "gradient_boosting", "catboost"]
all_models = [("temporal", m) for m in models] + [("tree-based", m) for m in tree_models]
model_names = {
    "gru": "GRU", "lstm": "LSTM", "tcn": "TCN", "sand": "SAnD", "grud": "GRU-D", "strats": "STraTS", "interpnet": "InterpNet", "mlp": "MLP",
    "random_forest": "RF", "xgboost": "XGB", "gradient_boosting": "GB", "catboost": "CatB", "logistic_regression": "LR"
}
print_metrics = ['auroc', 'auprc', 'f1', 'recall']
metrics_name_dict = {'auroc': 'AUC-ROC', 'auprc': 'AUC-PR', 'f1': 'F1-score', 'recall': 'Recall'}

_missing = []


def results_base(dataset, cohort, seed=None):
    base = SAVED_DATA_DIR / dataset
    if seed is not None:
        base = base / "multi_seed_training"
    return base / "results" / cohort


def ml_run_dir(dataset, cohort, model, prefix, seed=None):
    path = results_base(dataset, cohort, seed) / "non_time_series" / model
    if seed is not None:
        path = path / f"seed_{seed}"
    return path / prefix


def ts_results_path(dataset, cohort, model, prefix, fold, grid="nested", seed=None, train_mode="standard"):
    path = results_base(dataset, cohort, seed) / "time_series" / train_mode / model
    if seed is not None:
        path = path / f"seed_{seed}"
    return path / prefix / f"fold_{fold}" / f"grid_{grid}" / "results_final.csv"


def read_ml(dataset, cohort, model, prefix, seed=None):
    """Per-fold rows of an ML run (without the "Average" row), metrics as float; None if missing."""
    if old_paths.OLD_RESULTS_HOME:
        path = old_paths.ml_results_file(dataset, cohort, model, prefix, seed)
    else:
        path = ml_run_dir(dataset, cohort, model, prefix, seed) / "results_final.csv"
    if not path.exists():
        _missing.append(path)
        return None
    df = pd.read_csv(path)
    if old_paths.OLD_RESULTS_HOME:
        df["Model"] = df["Model"].replace(old_paths.ML_NEW_KEYS)
    df = df[df["Fold"] != "Average"].copy()
    df["Fold"] = df["Fold"].astype(int)
    for m in ["AUC-ROC", "AUC-PRC", "f1_score"]:
        df[m] = df[m].astype(float)
    df["cohort"] = cohort
    return df


def read_ts(dataset, cohort, model, prefix, grid="nested", seed=None, train_mode="standard"):
    """Test rows of a TS run, one per fold (the last one if the file was appended on reruns); None if no fold exists."""
    rows = []
    for fold in range(NUM_FOLDS):
        if old_paths.OLD_RESULTS_HOME:
            path = old_paths.ts_results_file(dataset, cohort, model, prefix, fold, grid, seed, train_mode)
        else:
            path = ts_results_path(dataset, cohort, model, prefix, fold, grid, seed, train_mode)
        if not path.exists():
            _missing.append(path)
            continue
        df = pd.read_csv(path)
        test = df[df["split"] == "test"]
        if not test.empty:
            rows.append(test.iloc[[-1]])
    return pd.concat(rows, ignore_index=True) if rows else None


def concat(dfs):
    dfs = [d for d in dfs if d is not None]
    if not dfs:
        raise SystemExit("No results found.\n" + missing_text())
    return pd.concat(dfs, ignore_index=True)


def missing_text():
    root = SAVED_DATA_DIR.parent
    return "\n".join(f"  {Path(p).relative_to(root) if Path(p).is_relative_to(root) else p}" for p in _missing)


def report_missing():
    if _missing:
        print(f"\n{len(_missing)} result files missing (skipped):\n" + missing_text())


def save(fig_or_plt, name, formats=("svg",), **kwargs):
    """Save a figure to saved_data/figures/<name>.<fmt>."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = FIGURES_DIR / f"{name}.{fmt}"
        fig_or_plt.savefig(out, dpi=300, **kwargs)
        print(f"Saved {out}")
