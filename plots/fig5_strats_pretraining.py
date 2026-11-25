"""Fig. 5: STraTS test metrics (mean over 5 folds) without pretraining and with pretraining on the dataset or the cohort,
fine-tuning all layers or the final layers; colours show the difference to no pretraining.
Ported as is from analyse_results.ipynb (cells 70-83, default rcParams as in the original figures).
Both versions of the notebook are produced: pretraining.{svg,pdf,png} (cell 81) and pretraining2.svg (cell 83).

Needs:  bash experiments/exp_strats_pretraining.sh
Usage:  python -m plots.fig5_strats_pretraining
Output: saved_data/figures/pretraining.{svg,pdf,png}, saved_data/figures/pretraining2.svg
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

from plots.load import cohorts, concat, exps, metrics_name_dict, print_metrics, read_ts, report_missing, save

# notebook mode -> (train_mode, prefix) of experiments/exp_strats_pretraining.sh
MODES = {
    "pt_none": ("standard", "pt_none"),
    "pt_all_df_last_layers": ("finetune", "pt_all_freeze"),
    "pt_all_df_all_weights": ("finetune", "pt_all_all_weights"),
    "pt_cohort_last_layers": ("finetune", "pt_cohort_freeze"),
    "pt_cohort_all_weights": ("finetune", "pt_cohort_all_weights"),
}


def main():
    # cell 70
    df_list = []
    for (dataset, cohort) in exps:
        for mode, (train_mode, prefix) in MODES.items():
            file = read_ts(dataset, cohort, "strats", prefix, grid="best_epochs", train_mode=train_mode)
            if file is not None:
                file["mode"] = mode
                df_list.append(file)
    combined_df = concat(df_list)

    # cell 74
    df_plot = (
        combined_df[combined_df["split"] == "test"]
        .groupby(["cohort", "mode"])[print_metrics[:-1]]
        .mean()
        .unstack("mode")
    )
    df_plot = df_plot.reindex(cohorts)
    df_plot.loc["Average"] = df_plot.mean(axis=0)

    # cell 75
    df_plot_ = df_plot.copy()
    for m in print_metrics[:-1]:
        for mode in df_plot[m].columns:
            df_plot_[(m, mode)] = df_plot[(m, mode)] - df_plot[(m, "pt_none")]

    # cell 79
    df_plot_.index = ['Aplasia (MIMIC-IV)', 'Aplasia (UKEr)', 'NF (MIMIC-IV)', 'NF (UKEr)', 'Average']

    # cell 80
    vmin = df_plot_.min().min()
    vmax = df_plot_.max().max()
    extreme = max(np.abs(vmin), np.abs(vmax))
    norm = TwoSlopeNorm(vmin=-extreme, vcenter=0, vmax=extreme)

    # cells 76, 81: version 1
    pt_mode_dict = {'pt_none': 'no pretraining', 'pt_all_df_last_layers': 'dataset - final', 'pt_all_df_all_weights': 'dataset - all',
                    'pt_cohort_last_layers': 'cohort - final', 'pt_cohort_all_weights': 'cohort - all'}

    sns.set_style("white")
    fig, axes = plt.subplots(1, len(print_metrics[:-1]), figsize=(3.5 * len(print_metrics[:-1]), 4), sharey=True, gridspec_kw={'width_ratios': [1, 1, 1.25]})
    if len(print_metrics[:-1]) == 1:
        axes = [axes]

    # Plot each metric in a separate heatmap section
    for i, metric in enumerate(print_metrics[:-1]):

        data = df_plot_[metric]
        data.columns = data.columns.map(pt_mode_dict)
        hm = sns.heatmap(
            data,
            ax=axes[i],
            cmap="coolwarm_r",
            cbar=True if i == len(print_metrics[:-1]) - 1 else False,  # only last heatmap gets colorbar
            square=True,
            norm=norm,
            alpha=0.6,
            annot=df_plot[metric].round(3),  # add values
            fmt=".3f",                        # 3 decimals
            annot_kws={"fontsize": 9.5, "color": "black"},
            cbar_kws={"shrink": 0.8, "label": "difference"}
        )

        axes[i].set_title(metrics_name_dict[metric], fontsize=13)
        axes[i].set_ylabel("")
        axes[i].set_xlabel("")

    for ax in axes:
        plt.setp(ax.get_yticklabels(), ha='right', fontsize=11)

    cbar = hm.collections[0].colorbar  # get the last colorbar
    for spine in cbar.ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("black")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)

    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    plt.suptitle("Pretraning Finetuning all final none all adm cohort adm", fontsize=14)
    save(plt, "pretraining", formats=("svg", "pdf", "png"))
    plt.close()

    # cells 82, 83: version 2
    pt_mode_dict = {'pt_none': 'none', 'pt_all_df_last_layers': 'final\ndataset', 'pt_all_df_all_weights': 'all\ndataset',
                    'pt_cohort_last_layers': 'final\ncohort', 'pt_cohort_all_weights': 'all\ncohort'}

    # the original pretraining2.svg shows the neutropenic fever cohorts first
    nf_first = [2, 3, 0, 1, 4]
    df_plot, df_plot_ = df_plot.iloc[nf_first], df_plot_.iloc[nf_first]

    sns.set_style("white")

    fig, axes = plt.subplots(len(print_metrics[:2]), 1, figsize=(4.5, 4.5 * len(print_metrics[:2])))

    if len(print_metrics[:2]) == 1:
        axes = [axes]

    for i, metric in enumerate(print_metrics[:2]):
        data = df_plot_[metric]
        data.columns = data.columns.map(pt_mode_dict)

        # Plot heatmap without square=True
        ax = axes[i]
        sns.heatmap(
            data,
            ax=ax,
            cmap="coolwarm_r",
            square=True,  # allow stretching
            norm=norm,
            alpha=0.6,
            annot=df_plot[metric].round(3),
            fmt=".3f",
            annot_kws={"fontsize": 10, "color": "black"},
            cbar=False,  # disable automatic colorbar
            linewidths=0.5,
            linecolor="gray"
        )

        ax.set_title(metrics_name_dict[metric], fontsize=11)
        ax.set_ylabel("")
        ax.set_xlabel("pretraining")

        # Add horizontal colorbar for this heatmap
        cbar = ax.figure.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap="coolwarm_r"),
            ax=ax,
            orientation='horizontal',
            fraction=0.1,  # shrink size
            pad=0.2,        # distance from heatmap
            label='difference'
        )

        cbar.set_ticks([-0.06, -0.03, 0.0, 0.03, 0.06])  # optional: specify ticks
        cbar.ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))  # format with 2 decimals

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1)

    plt.tight_layout()
    save(plt, "pretraining2")
    plt.close()
    report_missing()


if __name__ == "__main__":
    main()
