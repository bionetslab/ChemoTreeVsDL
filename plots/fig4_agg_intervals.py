"""Fig. 4: AUC-ROC (mean ± sd over 5 folds) on MIMIC-IV for the aggregation intervals 3/6/12/24h (non-temporal and
regular TS models) and for GRU-D/STraTS with timestamps retained vs dropped. Nested 5x3 CV, VMD features.
Ported as is from analyse_results.ipynb (with the rcParams that were active when it was drawn).

Needs:  bash experiments/exp_agg_intervals.sh   (3, 6, 12h)
        bash experiments/exp_drop_timestamps.sh (timestamps dropped)
        bash experiments/exp_full_run.sh        (24h, timestamps retained)
Usage:  python -m plots.fig4_agg_intervals
Output: saved_data/figures/agg_int_up_sd.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from plots import old_paths
from plots.load import cohort_names, concat, exps, model_names, read_ml, read_ts, report_missing, save

FULL_RUN = "full_run"  # 24h, timestamps retained


def agg_prefix(agg_int):
    return FULL_RUN if agg_int == 24 else f"agg_int_{agg_int}"


def main():
    # regular TS models
    df_list = []
    for model in ["gru", "tcn", "lstm", "sand", "mlp"]:
        for (dataset, cohort) in exps[:2]:
            for mode in [3, 6, 12, 24]:
                file = read_ts(dataset, cohort, model, agg_prefix(mode))
                if file is not None:
                    file["mode"] = mode
                    df_list.append(file)
    combined_df = concat(df_list)
    df1 = combined_df[["cohort", "model_type", "agg_int", "auroc"]]

    # non-temporal models
    tree_models = ["logistic_regression", "random_forest", "xgboost", "gradient_boosting", "catboost"]
    agg_intervals = [3, 6, 12, 24]
    df_list = []
    for (dataset, cohort) in exps[:2]:
        for model in tree_models:
            for agg_int in agg_intervals:
                # the old notebook took the 24h ML results from the aggregation interval run, not the full run
                prefix = "agg_int_24" if old_paths.OLD_RESULTS_HOME and agg_int == 24 else agg_prefix(agg_int)
                file = read_ml(dataset, cohort, model, prefix)
                if file is not None:
                    file["agg_int"] = agg_int
                    df_list.append(file)
    combined_df = concat(df_list)
    df1t = combined_df[["cohort", "Model", "agg_int", "AUC-ROC"]].rename(columns={"Model": "model_type", "AUC-ROC": "auroc"})
    df1 = pd.concat([df1, df1t])

    # irregular TS models with timestamps dropped / retained
    df_list = []
    for (dataset, cohort) in exps[:2]:
        for model in ["grud", "interpnet", "strats"]:
            for mode, prefix in {"dropped_timestamps": "drop_minutes", "with_timestamps": FULL_RUN}.items():
                file = read_ts(dataset, cohort, model, prefix)
                if file is not None:
                    file["mode"] = mode
                    df_list.append(file)
    combined_df = concat(df_list)
    df2 = combined_df[["cohort", "model_type", "mode", "auroc"]].copy()
    df2["mode"] = df2["mode"].map({"with_timestamps": "retained", "dropped_timestamps": "dropped"})
    df = pd.concat([df1.rename(columns={"agg_int": "mode"}), df2])

    cmap = sns.color_palette("Spectral", as_cmap=True)
    flare_cols = [cmap(i / 8) for i in range(8)]
    hue_colors = {"3": flare_cols[3], "6": flare_cols[2], "12": flare_cols[1], "24": flare_cols[0],
                  "retained": flare_cols[6], "dropped": flare_cols[7]}

    sub_cohorts = ["mimic_cohort_NF_30_days", "mimic_cohort_aplasia_45_days"]
    x_order = ["LR", "MLP", "RF", "GB", "XGB", "CatB", "GRU", "LSTM", "TCN", "SAnD", "GRU-D", "STraTS"]

    df_filtered = df[df["cohort"].isin(sub_cohorts)].copy()
    df_filtered["model_type"] = df_filtered["model_type"].map(model_names)
    df_filtered["cohort"] = df_filtered["cohort"].map(cohort_names)
    df_filtered["model_type"] = pd.Categorical(df_filtered["model_type"], categories=x_order, ordered=True)
    df_filtered["mode"] = df_filtered["mode"].astype("str")

    # rcParams active when this figure was drawn
    plt.rcParams.update({
        "font.size": 14,          # base font size
        "axes.labelsize": 14,     # axis labels
        "axes.titlesize": 18,     # subplot titles
        "xtick.labelsize": 14,    # x tick labels
        "ytick.labelsize": 14,    # y tick labels
        "legend.fontsize": 14,    # legend text
        "legend.title_fontsize": 14
    })

    sns.set_style("whitegrid")

    g = sns.catplot(
        data=df_filtered.reset_index(drop=True),
        x="model_type",
        y="auroc",
        hue="mode",
        col="cohort",
        kind="point",
        dodge=0.8,
        linestyle="none",
        errorbar="sd",
        capsize=0.15,
        height=4.5,
        aspect=1.2,
        linewidth=1,
        markersize=6,
        palette=hue_colors,
        sharey=False
    )

    g.set_axis_labels("", "AUC-ROC")
    g.set_titles("{col_name}")

    g._legend.set_bbox_to_anchor((0.1, 0.1))
    g._legend.set_loc('lower left')
    g._legend.set_title("Aggregation\n interval (h)")

    # Hide the default legend
    g._legend.remove()

    # numeric agg_int dots
    num_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=hue_colors[x],
                          markersize=8, label=str(x)) for x in ["3", "6", "12", "24"]]

    # special cases dots
    spec_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=hue_colors[x],
                           markersize=8, label=x) for x in ["retained", "dropped"]]

    # add legends to figure
    leg1 = g.fig.legend(handles=num_handles, title="Aggregation\n interval (h)", loc="lower left", bbox_to_anchor=(0.14, 0.2), fontsize=13)
    leg2 = g.fig.legend(handles=spec_handles, title="Timestamps", loc="lower left", bbox_to_anchor=(0.35, 0.2), fontsize=13)
    leg1.get_title().set_fontsize(14)
    leg2.get_title().set_fontsize(14)

    for ax in g.axes.flatten():
        # Make all spines visible
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.5)

    # Rotate x-axis tick labels
    for ax in g.axes.flatten():
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=14)
        plt.setp(ax.get_yticklabels(), ha='right', fontsize=13)
        ax.set_title(ax.get_title(), fontsize=14)

    for ax in g.axes.flatten():
        # Get the number of x-ticks (models)
        xticks = ax.get_xticks()

        # Draw vertical lines between each model
        for x in xticks[:-1]:  # skip the last tick
            ax.axvline(x=x + 0.34, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    g.set_ylabels("AUC-ROC", fontsize=14)
    # Format y-ticks: 2 decimal points
    for ax in g.axes.flatten():
        ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    plt.tight_layout()
    save(plt, "agg_int_up_sd")
    plt.close()
    report_missing()


if __name__ == "__main__":
    main()
