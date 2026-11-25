"""Fig. 2: pairwise one-sided Wilcoxon tests with Benjamini-Hochberg correction between all models,
on 5 CV split seeds x 5 folds x 4 cohorts (multi-seed run, best parameters, VMD features, 24h bins).
Ported as is from analyse_results.ipynb (cells 129, 142, 144, 145, 147, make_p_val_cmap from cell 122).

Needs:  bash experiments/exp_multi_seed.sh
Usage:  python -m plots.fig2_stats
Output: saved_data/figures/wilcoxon_onesided_asimmetric.svg
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from plots.load import all_models, concat, exps, model_names, print_metrics, read_ml, read_ts, report_missing, save

PREFIX = "multi_seed"
SEEDS = [4, 15, 42, 82, 95]  # as in experiments/exp_multi_seed.sh


def make_p_val_cmap():
    # Define a gradient of reds for small values
    reds = np.linspace([0.7, 0.1, 0.1], [1.0, 0.7, 0.7], 4)  # 4 shades from dark to light red

    # Light blue for values > 0.05
    blues = np.array([[0.7, 0.85, 1.0]])  # single light blue

    # Combine into one colormap
    colors = np.vstack([reds, blues])
    cmap = ListedColormap(colors)

    # Define boundaries
    bounds = [0, 0.0001, 0.0005, 0.001, 0.05, 1.0]  # must have len(bounds) = len(colors) + 1
    norm = BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def compute_one_sided_p_vals(pivot, correction=True):
    """
    Compute pairwise one-sided Wilcoxon tests.
    Returns a matrix where p_ij is the p-value for H0: row <= column
    (i.e., row better than column).
    """
    models = pivot.columns
    n_models = len(models)

    # Initialize matrix
    p_matrix = np.ones((n_models, n_models))

    # Store raw p-values for correction
    p_values = []
    indices = []

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i != j:
                # Compute Wilcoxon signed-rank test
                stat, p_two_sided = wilcoxon(pivot[m1], pivot[m2], alternative="greater")

                # Determine direction: row better than column
                diff = pivot[m1] - pivot[m2]
                n_positive = np.sum(diff > 0)

                # One-sided p-value
                if n_positive >= len(diff)/2:
                    p_one_sided = p_two_sided / 2
                else:
                    p_one_sided = 1 - p_two_sided / 2

                p_matrix[i, j] = p_one_sided
                if i < j:
                    p_values.append(p_one_sided)
                    indices.append((i, j))

    # Apply FDR correction if needed
    if correction and p_values:
        _, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        for k, (i, j) in enumerate(indices):
            p_matrix[i, j] = p_corrected[k]
            p_matrix[j, i] = 1 - p_corrected[k]  # opposite direction

    return p_matrix


def load_results():
    # cell 129
    df_list = []
    for (dataset, cohort) in exps:
        for fold_seed in SEEDS:
            for m_type, model in all_models:
                if m_type == "temporal":
                    file = read_ts(dataset, cohort, model, PREFIX, grid="best", seed=fold_seed)
                    if file is None:
                        continue
                    file = file[['cohort', 'fold', 'model_type'] + print_metrics[:-1]].copy()
                    file["model_type"] = file["model_type"].map(model_names)
                    file.rename(columns={"fold": "Fold", "model_type": "Model", "auroc": "AUC-ROC", "auprc": "AUC-PRC", "f1": "f1_score"}, inplace=True)
                else:
                    file = read_ml(dataset, cohort, model, PREFIX, seed=fold_seed)
                    if file is None:
                        continue
                    file = file[['Fold', 'Model', 'AUC-ROC', 'AUC-PRC', 'f1_score', 'cohort']].copy()
                    file["Model"] = file["Model"].map(model_names)
                file["Type"] = m_type
                file["Seed"] = fold_seed
                df_list.append(file)
    return concat(df_list)


def main():
    combined_df = load_results()

    # cell 144
    models_ordered_dict = {}
    all_metric_p_matrices = {}
    for sel_metric in ["AUC-ROC", "AUC-PRC", "f1_score"]:
        models_ordered = combined_df.groupby("Model")[sel_metric].mean().sort_values(ascending=False).index
        models_ordered_dict[sel_metric] = models_ordered
        # comparison per cohort
        pivot = combined_df.pivot(index=["cohort", "Seed", "Fold"], columns="Model", values=sel_metric)[models_ordered]
        p_matrix = compute_one_sided_p_vals(pivot, correction=True)
        all_metric_p_matrices[sel_metric] = p_matrix

    # cell 145
    plt.rcParams.update({
        "font.size": 14,          # base font size
        "axes.labelsize": 14,     # axis labels
        "axes.titlesize": 18,     # subplot titles
        "xtick.labelsize": 15,    # x tick labels
        "ytick.labelsize": 15,    # y tick labels
        "legend.fontsize": 6,    # legend text
        "legend.title_fontsize": 14
    })

    # cell 147
    sns.set_style("white")
    cmap, norm = make_p_val_cmap()
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.4), sharex=False, sharey=False)

    for ax, (sel_metric, matrix) in zip(axes, all_metric_p_matrices.items()):
        models_ordered = models_ordered_dict[sel_metric]
        # Use imshow for consistent square layout and smooth color scaling
        im = ax.imshow(matrix, cmap=cmap, norm=norm, alpha=0.6, aspect='equal')

        # Axis ticks and labels
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_xticklabels(models_ordered, rotation=90, ha='right', fontsize=12)
        ax.set_yticklabels(models_ordered, fontsize=12)
        ax.set_ylabel("", fontsize=13)
        ax.set_xlabel("", fontsize=13)
        title = sel_metric if sel_metric != "f1_score" else "F1 score"
        ax.set_title(title, fontsize=11)

        ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
        ax.grid(which="minor", color="lightgrey", linestyle='-', linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Add horizontal colorbar below each subplot
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.12, pad=0.2)
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        cbar.set_label('adjusted p-value', fontsize=13, labelpad=8)
        cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    save(plt, "wilcoxon_onesided_asimmetric")
    plt.close()
    report_missing()


if __name__ == "__main__":
    main()
