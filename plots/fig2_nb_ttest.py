"""Fig. 2 alternative: pairwise one-sided Nadeau-Bengio corrected resampled t-tests with Benjamini-Hochberg correction,
per cohort on 5 CV split seeds x 5 folds (multi-seed run, best parameters, VMD features, 24h bins).
Shown as a dominance matrix: number of cohorts in which the row model is significantly better (p < 0.05) than the column model.
Ported as is from analyse_results.ipynb. Not in the published paper (reviewers R2.8 / R3.5).

Needs:  bash experiments/exp_multi_seed.sh, and the default folds saved_data/<dataset>/folds/<cohort>/fold_0.pkl (n_train, n_test)
Usage:  python -m plots.fig2_nb_ttest
Output: saved_data/figures/nb_ttest2.pdf
"""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import t
from statsmodels.stats.multitest import multipletests

from config.constants import SAVED_DATA_DIR
from plots.fig2_stats import load_results
from plots.load import cohorts, exps, report_missing, save


def nadeau_bengio_ttest(a, b, n_train, n_test):
    diff = a - b
    k = len(diff)
    mean_diff = diff.mean()
    corrected_std = np.sqrt(np.var(diff, ddof=1) * (1/k + n_test / n_train))
    t_stat = mean_diff / corrected_std
    p_val = t.sf(t_stat, k-1)  # one-sided
    return p_val


def compute_p_vals_nb_ttest(pivot, n_train_test, correction=True):
    n_train, n_test = n_train_test
    models = pivot.columns
    n_models = len(models)

    p_matrix = np.ones((n_models, n_models))
    p_values = []

    for i in range(n_models):
        for j in range(i+1, n_models):
            a = pivot.iloc[:, i].values
            b = pivot.iloc[:, j].values
            # Always test the model with higher mean against the lower
            if a.mean() >= b.mean():
                p_values.append(nadeau_bengio_ttest(a, b, n_train, n_test))
            else:
                p_values.append(nadeau_bengio_ttest(b, a, n_train, n_test))

    if correction:
        _, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
    else:
        p_corrected = p_values

    # Fill symmetric matrix
    k = 0
    for i in range(n_models):
        for j in range(i+1, n_models):
            p_matrix[i, j] = p_corrected[k]
            p_matrix[j, i] = p_corrected[k]
            k += 1

    return p_matrix


def train_test_sizes():
    l_dict = {}
    for dataset, cohort in exps:
        file_path = SAVED_DATA_DIR / dataset / "folds" / cohort / "fold_0.pkl"
        if file_path.exists():
            with open(file_path, "rb") as f:
                train, val, test = pickle.load(f)
                ntrainids = len(train) + len(val)
                ntestids = len(test)
                l_dict[cohort] = (ntrainids, ntestids)
            print(f"{cohort}: trainids {ntrainids}, test ids {ntestids}")
        else:
            print(f"File not found: {file_path}")
    return l_dict


def main():
    combined_df = load_results()
    l_dict = train_test_sizes()

    models_ordered_dict = {}
    all_metric_p_matrices = {}
    run_cohorts = [c for c in cohorts if c in set(combined_df["cohort"]) and c in l_dict]
    for sel_metric in ["AUC-ROC", "AUC-PRC", "f1_score"]:
        models_ordered = combined_df.groupby("Model")[sel_metric].mean().sort_values(ascending=False).index
        models_ordered_dict[sel_metric] = models_ordered
        # comparison per cohort
        all_p_matrices = []
        for cohort in run_cohorts:
            df_cohort = combined_df[combined_df["cohort"] == cohort]
            # pivot to have folds as rows, models as columns
            pivot = df_cohort.pivot(index=["Fold", "Seed"], columns="Model", values=sel_metric)[models_ordered]
            p_matrix = compute_p_vals_nb_ttest(pivot, l_dict[cohort], True)
            all_p_matrices.append(p_matrix)
        all_metric_p_matrices[sel_metric] = all_p_matrices

    dominance_matrices = {}
    for sel_metric in ["AUC-ROC", "AUC-PRC", "f1_score"]:
        models_ordered = models_ordered_dict[sel_metric]
        dominance_matrix = pd.DataFrame(0, index=models_ordered, columns=models_ordered)
        for cohort, p_matrix in zip(run_cohorts, all_metric_p_matrices[sel_metric]):  # loop over cohorts
            df_cohort = combined_df[combined_df["cohort"] == cohort]
            pivot = df_cohort.pivot(index=["Fold", "Seed"], columns="Model", values=sel_metric)[models_ordered]

            for i, m1 in enumerate(models_ordered):
                for j, m2 in enumerate(models_ordered):
                    if i > j:
                        # check significance
                        if p_matrix[i, j] < 0.05:
                            # check actual mean difference
                            mean_diff = pivot[m1].mean() - pivot[m2].mean()
                            if mean_diff > 0:
                                dominance_matrix.loc[m1, m2] += 1
                            else:
                                dominance_matrix.loc[m2, m1] += 1
        dominance_matrices[sel_metric] = dominance_matrix

    cmap = LinearSegmentedColormap.from_list(
        "white_to_seagreen",
        ["#fffaf0", "#2e8b57"]  # cream/white to seagreen
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False)

    for ax, (sel_metric, dominance_matrix) in zip(axes, dominance_matrices.items()):
        sns.heatmap(
            dominance_matrix,
            annot=True,
            fmt="d",
            cmap=cmap,
            ax=ax,
            square=True,
            cbar=False
        )

        ax.set_title(sel_metric)
        ax.set_xlabel("")
        ax.set_ylabel("Model" if ax == axes[0] else "")

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")

    plt.tight_layout()
    save(plt, "nb_ttest2")
    plt.close()
    report_missing()


if __name__ == "__main__":
    main()
