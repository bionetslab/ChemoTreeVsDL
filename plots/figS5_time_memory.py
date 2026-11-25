"""Fig. S5: number of parameters, peak memory and time for training and inference, summed over the 5 folds,
non-temporal models on CPU and TS models on GPU, best parameters, VMD features, 24h bins.
Ported as is from analyse_results.ipynb (cells 47-62).

Needs:  TS: training.ts.main with --grid best --config_path experiments/configs/ts_config_params_tm.yaml
            --prefix tm_gpu (and tm_cpu); there is no experiment script for this yet.
        ML: <model>_tm.csv files with columns fold, train time, train peak memory, inference time, inference peak memory.
            The code that wrote them in the old pipeline has not been found yet, so ML models are skipped if missing.
Usage:  python -m plots.figS5_time_memory
Output: saved_data/figures/time_memory_log.{svg,pdf}
"""
import colorsys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plots import old_paths
from plots.load import _missing, cohort_names, concat, exps, ml_run_dir, model_names, read_ts, report_missing, save, tree_models

TS_GPU_PREFIX = "tm_gpu"
TS_CPU_PREFIX = "tm_cpu"
ML_TM_PREFIX = "time_memory"


def ml_tm_path(dataset, cohort, model):
    if old_paths.OLD_RESULTS_HOME:
        return old_paths.ml_tm_file(dataset, cohort, model)
    return ml_run_dir(dataset, cohort, model, ML_TM_PREFIX) / f"{model}_tm.csv"


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by blending it with white.
    amount: 0 = original color, 1 = white
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    lightened = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    return lightened


def main():
    # cell 47
    tm_metrics_sub = ['total_parameters', 'total_peak_memory', 'inference_peak_memory', 'epoch_train_time', 'total_train_time', 'total_inference_time']
    tm_metrics_sub_names = ['# parameters', 'training peak memory', 'inference peak memory', 'epoch training time', 'total training time', 'total inference time']
    n_metrics = len(tm_metrics_sub)

    # cells 48, 50: TS models on GPU and CPU
    ts_models = ["gru", "tcn", "lstm", "sand", "grud", "interpnet", "strats", "mlp"]
    combined_df = concat([read_ts(dataset, cohort, model, TS_GPU_PREFIX, grid="best") for model in ts_models for (dataset, cohort) in exps])
    combined_df_gpu = combined_df[combined_df["split"] == "test"][["fold", "model_type", "cohort"] + tm_metrics_sub]
    cpu = [read_ts(dataset, cohort, model, TS_CPU_PREFIX, grid="best") for model in ts_models for (dataset, cohort) in exps]
    combined_df_cpu = (pd.concat([d for d in cpu if d is not None], ignore_index=True)[["fold", "model_type", "cohort"] + tm_metrics_sub]
                       if any(d is not None for d in cpu) else pd.DataFrame(columns=["fold", "model_type", "cohort"] + tm_metrics_sub))

    # cell 53: non-temporal models
    df_list = []
    for model in tree_models:
        for (dataset, cohort) in exps:
            path = ml_tm_path(dataset, cohort, model)
            if not path.exists():
                _missing.append(path)
                continue
            file = pd.read_csv(path)
            file["model_type"] = model
            file["cohort"] = cohort
            df_list.append(file)
    combined_df_trees = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame(columns=range(5))
    if not df_list:
        combined_df_trees["model_type"] = []
        combined_df_trees["cohort"] = []
    combined_df_trees.columns = ['fold', 'total_train_time', 'total_peak_memory', 'total_inference_time', 'inference_peak_memory', 'model_type', 'cohort']

    # cell 56
    combined_df_gpu["model_type"] = combined_df_gpu["model_type"].map(model_names)
    combined_df_gpu["model_type"] = combined_df_gpu["model_type"] + " [GPU]"
    combined_df_trees["model_type"] = combined_df_trees["model_type"].map(model_names)
    combined_df_trees["model_type"] = combined_df_trees["model_type"] + " [CPU]"
    # cells 57-58
    df = pd.concat([combined_df_gpu, combined_df_cpu, combined_df_trees])
    df['inference_peak_memory'] = df['inference_peak_memory'] / 1000
    df['total_peak_memory'] = df['total_peak_memory'] / 1000

    # cell 60
    cohort_colors = plt.get_cmap("Set2").colors
    cohort_colors = [lighten_color(c, amount=0.9) for c in cohort_colors]

    # cell 62
    sns.set_style("whitegrid")

    plt.rcParams.update({
        "font.size": 14,          # base font size
        "axes.labelsize": 14,     # axis labels
        "axes.titlesize": 18,     # subplot titles
        "xtick.labelsize": 14,    # x tick labels
        "ytick.labelsize": 14,    # y tick labels
        "legend.fontsize": 14,    # legend text
        "legend.title_fontsize": 14
    })

    model_order = ['LR [CPU]', 'RF [CPU]', 'GB [CPU]', 'XGB [CPU]', 'CatB [CPU]', 'MLP [GPU]', 'GRU [GPU]', 'LSTM [GPU]', 'TCN [GPU]', 'SAnD [GPU]', 'InterpNet [GPU]', 'GRU-D [GPU]', 'STraTS [GPU]']
    cohorts_order = df['cohort'].unique()  # or define manually
    cohort_color_map = {c: cohort_colors[i % len(cohort_colors)] for i, c in enumerate(cohorts_order)}

    fig, axes = plt.subplots(2, int(n_metrics / 2), figsize=(3.8 * int(n_metrics / 2), 8), sharey=True)
    axes = axes.ravel()

    for ax, metric, metric_name in zip(axes, tm_metrics_sub, tm_metrics_sub_names):
        df_plot = df.dropna(subset=[metric])

        for m in model_order[::-1]:
            for cohort in cohorts_order:
                val = df_plot.loc[(df_plot['model_type'] == m) & (df_plot['cohort'] == cohort), metric]
                val_sum = val.sum() if not val.empty else 0
                if val_sum > 0:
                    # Draw a line from x=0 to the dot
                    ax.plot([0, val_sum], [m, m], color="darkgrey", alpha=0.4, linewidth=4, zorder=1)

                    # Draw the dot
                    ax.scatter(
                        val_sum,        # x position
                        m,              # y position
                        s=100,          # circle size (area)
                        color=cohort_color_map[cohort],
                        edgecolor="#111111",
                        linewidth=1,
                        alpha=1,
                        zorder=2,
                        label=cohort
                    )

        # Axis labels
        if 'time' in metric_name.lower():
            xlabel = f"{metric_name} [s]"
        elif 'memory' in metric_name.lower():
            xlabel = f"{metric_name} [Gbyte]"
        else:
            xlabel = metric_name

        ax.axhline(y=7.5, color="gray", linestyle="--", linewidth=2, alpha=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        ax.set_xscale("log")

    # Adjust spines
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)

    # Shared legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    labels, handles = list(unique.keys()), list(unique.values())
    labels = [cohort_names.get(l, l) for l in labels]
    fig.legend(handles, labels, title="", loc="lower center", ncol=len(labels))

    plt.tight_layout(rect=[0.0, 0.05, 1, 1])
    save(plt, "time_memory_log", formats=("svg", "pdf"))
    plt.close()
    report_missing()


if __name__ == "__main__":
    main()
