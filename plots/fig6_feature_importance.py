"""Fig. 6 and Fig. S2-S4: CatBoost feature importance (PredictionValuesChange) averaged over the 5 outer folds,
VMD features, 24h bins: top 20 features, importance summed per lab test, per time bin and per V/M/D.
Ported as is from analyse_feature_importance.ipynb, one set of plots per cohort.

Needs:  bash experiments/exp_full_run.sh all catboost
        MIMIC_DIR (MIMIC-IV, for d_labitems) and UKER_DIR (UKEr database, for the loinc table) set in the environment,
        and supporting_files/Loinc.csv (from loinc.org, for the LOINC names)
Usage:  python -m plots.fig6_feature_importance [--cohort <cohort>]
Output: saved_data/figures/feature_importance/<cohort>/{feature_importance_plot,clinical_labels_distribution,
        bin_distribution,vmd_type_distribution}_new.pdf
"""
import argparse
import os
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.constants import MIMIC_DIR, SUPPORTING_FILES_DIR, UKER_DIR
from plots import old_paths
from plots.load import FIGURES_DIR, NUM_FOLDS, exps, ml_run_dir

LOINC_PATH = SUPPORTING_FILES_DIR / 'Loinc.csv'
MODEL_TYPE = "catboost"
PREFIX = "full_run"


def load_labitems(dataset):
    if dataset == "MIMIC_IV":
        return pd.read_csv(MIMIC_DIR + "hosp/d_labitems.csv.gz", compression="gzip")

    conn = sqlite3.connect(UKER_DIR)
    df_loinc = pd.read_sql_query("SELECT * FROM loinc;", conn)
    conn.close()

    d_loinc = pd.read_csv(
        LOINC_PATH,
        usecols=["LOINC_NUM", "COMPONENT"],
        dtype=str
    ).rename(columns={"LOINC_NUM": "code", "COMPONENT": "label"})

    return df_loinc.merge(d_loinc[['code', 'label']],
                          on='code',
                          how='left'
                          ).rename(columns={"id": "itemid"})


def extract_itemid_from_feature(feature_name):
    """Extract itemid from feature names like X_51256_14 or D_50954_13"""
    if feature_name in ['age', 'gender']:
        return None
    parts = feature_name.split('_')
    itemid = int(parts[1])
    feat_type = parts[0]
    if feat_type == 'X': feat_type = 'V'
    bin = int(parts[2])

    return itemid, feat_type, bin


def plot_cohort(dataset, cohort, d_labitems):
    output_dir = ml_run_dir(dataset, cohort, MODEL_TYPE, PREFIX)

    all_folds_importance = []
    for fold in range(NUM_FOLDS):
        if old_paths.OLD_RESULTS_HOME:
            feat_importance_path = old_paths.ml_feature_importance_file(dataset, cohort, MODEL_TYPE, fold)
        else:
            feat_importance_path = output_dir / f"fold_{fold}" / "feature_importance.csv"
        if not feat_importance_path.exists():
            print(f"Skipping {cohort}, missing {feat_importance_path}")
            return
        feat_imp = pd.read_csv(feat_importance_path)
        all_folds_importance.append(feat_imp)

    avg_df = all_folds_importance[0][['feature']].copy()
    importance_values = pd.DataFrame()
    for i, df in enumerate(all_folds_importance):
        importance_values[f'fold_{i}'] = df.set_index('feature')['imp']

    avg_df['imp_mean'] = importance_values.mean(axis=1).values
    avg_df['imp_std'] = importance_values.std(axis=1).values

    avg_df = avg_df.sort_values(by='imp_mean', ascending=False)

    avg_df[['itemid', 'V/M/D', 'bin']] = avg_df['feature'].apply(extract_itemid_from_feature).apply(pd.Series)

    if "mimic" in cohort:
        avg_df = avg_df.merge(
            d_labitems[['itemid', 'label', 'fluid', 'category']],
            on='itemid',
            how='left'
        )

    else:
        avg_df = avg_df.merge(
            d_labitems[['itemid', 'code', 'label']],
            on='itemid',
            how='left'
        )

    avg_df['clinical_name'] = avg_df['label'].fillna(avg_df['feature'])

    avg_df.loc[avg_df['clinical_name'].str.lower() == 'monocytes', 'clinical_name'] = 'Monocytes (%)'

    title_font = 32
    label_font = 30
    ticks_font = 28

    save_path = FIGURES_DIR / "feature_importance" / cohort
    os.makedirs(save_path, exist_ok=True)

    avg_df['feature_label'] = avg_df['feature']

    mask = ~avg_df['feature'].isin(['age', 'gender'])
    avg_df.loc[mask, 'feature_label'] = (
        avg_df.loc[mask, 'clinical_name'].astype(str) + ' [' +
        avg_df.loc[mask, 'V/M/D'].astype(str) + '] [bin ' +
        avg_df.loc[mask, 'bin'].astype('Int64').astype(str) + ']'
    )

    fig, ax = plt.subplots(figsize=(16, 12))
    top_features = avg_df.head(20)

    unique_values = top_features['imp_mean'].unique()
    cmap = plt.cm.inferno
    color_positions = np.linspace(0.3, 0.9, len(unique_values))
    value_color_map = dict(zip(unique_values, color_positions))
    colors = [cmap(value_color_map[value]) for value in unique_values]

    bars = ax.barh(
        y=range(len(top_features)),
        width=top_features['imp_mean'],
        color=colors,
        edgecolor='none',
        height=0.7
    )
    ax.set_yticks(range(len(top_features)))
    ax.tick_params(axis='x', labelsize=ticks_font)
    ax.set_yticklabels(top_features['feature_label'], fontsize=ticks_font)

    ax.invert_yaxis()
    ax.set_xlabel('Mean importance across folds', fontsize=label_font, labelpad=30)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{save_path}/feature_importance_plot_new.pdf', dpi=300)
    plt.close()

    top_features = avg_df

    fig1, ax1 = plt.subplots(figsize=(14, 8))

    label_counts = top_features.groupby("clinical_name")["imp_mean"].sum().round(2).sort_values(ascending=False).head(10)
    unique_values = label_counts.unique()
    cmap = plt.cm.coolwarm_r
    color_positions = np.linspace(0, 0.5, len(unique_values))
    value_color_map = dict(zip(unique_values, color_positions))
    colors = [cmap(value_color_map[val]) for val in label_counts.values]

    bars1 = ax1.bar(range(len(label_counts)), label_counts.values, color=colors, width=0.7)
    ax1.set_xticks(range(len(label_counts)))
    ax1.set_yticks([])
    ax1.set_xticklabels(label_counts.index, rotation=45, ha='right', fontsize=ticks_font)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    # Add value labels
    for i, val in enumerate(label_counts.values):
        ax1.text(i, val + 0.5, f'{val}', ha='center', va='bottom', fontsize=ticks_font)
    plt.tight_layout()
    plt.savefig(f'{save_path}/clinical_labels_distribution_new.pdf', dpi=300)
    plt.close()

    fig3, ax3 = plt.subplots(figsize=(14, 8))
    bin_counts = top_features.groupby("bin")["imp_mean"].sum().round(2).sort_values(ascending=False)
    unique_bins = bin_counts.unique()
    cmap = plt.cm.coolwarm_r
    color_positions = np.linspace(0, 0.5, len(unique_bins))
    bin_color_map = dict(zip(unique_bins, color_positions))
    colors = [cmap(bin_color_map[val]) for val in bin_counts.values]

    bars3 = ax3.bar(bin_counts.index, bin_counts.values, color=colors, width=0.9)
    ax3.set_xlabel('Time bin', fontsize=label_font, labelpad=30)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_xticks(bin_counts.index)
    ax3.tick_params(axis='x', labelsize=ticks_font)
    ax3.set_yticks([])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    # Add value labels
    for idx, val in bin_counts.items():
        ax3.text(idx, val + 0.5, f'{val}', ha='center', va='bottom', fontsize=24)
    plt.tight_layout()
    plt.savefig(f'{save_path}/bin_distribution_new.pdf', dpi=300)
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(6, 6))

    vmd_counts = top_features.groupby('V/M/D')['imp_mean'].sum().round(2).sort_values(ascending=False)
    categories = ['V', 'D', 'M']
    cmap = plt.cm.coolwarm_r
    color_positions = [0.5, 0.7, 0.9]
    color_map = {cat: cmap(pos) for cat, pos in zip(categories, color_positions)}
    colors = [color_map[val] for val in vmd_counts.index]

    wedges, texts, autotexts = ax2.pie(
        vmd_counts.values,
        labels=vmd_counts.index,
        autopct=lambda pct: f'{pct:.1f}%',
        pctdistance=1.3,
        labeldistance=1.1,
        colors=colors,
        startangle=90,
        textprops={'fontsize': label_font},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )

    offsets = [
        (-0.4, 0.4),
        (0.45, -0.3),
        (0.6, -0.1),
    ]

    for autotext, (dx, dy) in zip(autotexts, offsets):
        x, y = autotext.get_position()
        autotext.set_position((x + dx, y + dy))

    plt.tight_layout()
    plt.savefig(f'{save_path}/vmd_type_distribution_new.pdf', dpi=300)
    plt.close()
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default=None, help="one cohort (default: all four)")
    args = parser.parse_args()

    labitems = {}
    for dataset, cohort in exps:
        if args.cohort and cohort != args.cohort:
            continue
        if dataset not in labitems:
            labitems[dataset] = load_labitems(dataset)
        plot_cohort(dataset, cohort, labitems[dataset])


if __name__ == "__main__":
    main()
