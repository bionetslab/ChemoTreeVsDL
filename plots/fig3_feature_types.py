"""Fig. 3: AUC-ROC (mean ± sd over 5 folds) of the non-temporal models for the feature types V, M, D, VD, VM, MD, VMD,
nested 5x3 CV, 24h bins. Also the color wheel used as legend.
Ported as is from analyse_results.ipynb (cells 9-14, 18; color wheel cells 12-13).

Needs:  bash experiments/exp_feature_types.sh   (V, M, D, VD, VM, MD)
        bash experiments/exp_full_run.sh        (VMD)
Usage:  python -m plots.fig3_feature_types
Output: saved_data/figures/feat_types_paper_AUC-ROC_sd.svg, saved_data/figures/color_wheel.{svg,pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from plots.load import cohort_names, cohorts, concat, exps, model_names, read_ml, read_ts, report_missing, save

FULL_RUN = "full_run"  # VMD


def feat_prefix(feat_type):
    return FULL_RUN if feat_type == "VMD" else f"feat_type_{feat_type}"


def color_wheel():
    # cell 13
    # Palette and labels
    palette_list = ["#feef00", "#f88f1c", "#ee4234", "#a54398", "#3c5daa", "#01a357"]
    feat_types = ["M", "MD", "D", "VD", "V", "VM"]

    # Central circle info
    center_color = "#484848"
    center_label = "VMD"

    fig, ax = plt.subplots(figsize=(3, 3))

    # Plot the color wheel as a pie chart
    wedges, _ = ax.pie(
        [1]*len(palette_list),  # equal slices
        colors=palette_list,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.5, edgecolor='k', alpha=0.8),  # width<1 for donut shape
    )

    # Add labels for each slice
    for i, w in enumerate(wedges):
        angle = (w.theta2 + w.theta1)/2
        x = 0.75 * np.cos(np.deg2rad(angle))
        y = 0.75 * np.sin(np.deg2rad(angle))
        ax.text(x, y, feat_types[i], ha='center', va='center', fontsize=15, color='black')

    # Draw central grey circle
    circle = plt.Circle((0, 0), 0.35, color=center_color, ec='k', zorder=2, alpha=0.8)
    ax.add_artist(circle)
    ax.text(0, 0, center_label, ha='center', va='center', fontsize=15, color='white', zorder=3)

    ax.set(aspect="equal")
    save(plt, "color_wheel", formats=("svg", "pdf", "png"))
    plt.close()


def main():
    # cell 9: non-temporal models
    tree_models = ["random_forest", "xgboost", "gradient_boosting", "catboost", "logistic_regression"]
    feat_types = ["V", "M", "D", "VMD", "VM", "VD", "MD"]
    df_list = []
    for (dataset, cohort) in exps:
        for model in tree_models:
            for feat_type in feat_types:
                file = read_ml(dataset, cohort, model, feat_prefix(feat_type))
                if file is not None:
                    file["feat_type"] = feat_type
                    df_list.append(file)
    combined_df = concat(df_list)

    # cells 10-11: MLP
    df_list = []
    model = "mlp"
    for (dataset, cohort) in exps:
        for feat_type in ["V", "M", "D", "VMD", "VM", "VD", "MD"]:
            file = read_ts(dataset, cohort, model, feat_prefix(feat_type))
            if file is not None:
                file["cohort"] = cohort
                file["feat_type"] = feat_type
                df_list.append(file)
    combined_df_ = concat(df_list)
    combined_df_ = combined_df_[["cohort", "feat_type", "auroc", "auprc", "f1", "model_type"]].rename(columns={"auroc": "AUC-ROC", "auprc": "AUC-PRC", "f1": "f1_score", "model_type": "Model"})

    # cell 12
    palette_list = ["#feef00", "#f88f1c", "#ee4234", "#a54398", "#3c5daa", "#01a357", "#484848"]
    feat_types = ["M", "MD", "D", "VD", "V", "VM", "VMD"]
    palette_dict = dict(zip(feat_types, palette_list))

    color_wheel()

    # cell 14
    combined_df = pd.concat([combined_df, combined_df_])

    df_filtered = combined_df[combined_df["cohort"].isin(cohorts)].copy()
    df_filtered["cohort"] = df_filtered["cohort"].map(cohort_names)
    df_filtered["Model"] = df_filtered["Model"].map(model_names)

    model_order = ["LR", "MLP", "RF", "GB", "XGB", "CatB"]

    df_filtered["dataset"] = df_filtered["cohort"].apply(lambda x: "MIMIC-IV" if "MIMIC" in x else "UKEr")
    df_filtered["condition"] = df_filtered["cohort"].apply(lambda x: "Aplasia" if "Aplasia" in x else "Neutropenic Fever")
    df_filtered["Model"] = pd.Categorical(df_filtered["Model"], categories=model_order, ordered=True)
    df_filtered["feat_type"] = pd.Categorical(df_filtered["feat_type"], categories=["M", "D", "MD", "VD", "VM", "V", "VMD"], ordered=True)

    # cell 18
    sns.set_style("whitegrid")

    mm = "AUC-ROC"
    g = sns.catplot(
        data=df_filtered.reset_index(drop=True),
        x="Model",            # corresponds to x-axis in your FacetGrid
        y=mm,                 # corresponds to y-axis
        hue="feat_type",      # corresponds to hue in scatterplot
        col="dataset",        # former FacetGrid column
        row="condition",      # former FacetGrid row
        kind="point",         # use points to replicate scatter
        dodge=0.7,            # slightly separate points of different hue
        linestyle="none",     # no line connecting points
        errorbar="sd",        # show standard error
        capsize=0.15,          # small caps on error bars
        height=4.2,
        aspect=0.75,
        linewidth=1,
        markersize=6,
        palette=palette_dict,
        sharey="row",          # optional, keep y independent per row
        legend=None,
    )
    # Adjust spines and axes
    for i, ax in enumerate(g.axes.flatten()):
        # Make all spines visible
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.5)

        ax.set_title(cohort_names[cohorts[i]], fontsize=14)

    for ax in g.axes.flatten():
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=14)
        plt.setp(ax.get_yticklabels(), ha='right', fontsize=13)

    for ax in g.axes.flatten():
        # Get the number of x-ticks (models)
        xticks = ax.get_xticks()

        # Draw vertical lines between each model
        for x in xticks[:-1]:  # skip the last tick
            ax.axvline(x=x + 0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    g.set_ylabels(mm, fontsize=14)
    g.set_xlabels("")

    # Format y-ticks to 2 decimals
    for ax in g.axes.flatten():
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    plt.tight_layout()
    save(plt, f"feat_types_paper_{mm}_sd")
    plt.close()
    report_missing()


if __name__ == "__main__":
    main()
