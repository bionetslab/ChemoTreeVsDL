"""Table 1: test metrics (mean and std over the 5 outer folds), nested 5x3 CV, VMD features, 24h bins.
Ported from analyse_results.ipynb (cell 28: ML models, cell 32: TS models).

Needs:  bash experiments/exp_full_run.sh
Usage:  python -m plots.table1
Output: saved_data/figures/table1_ml.csv, saved_data/figures/table1_ts.csv
"""
import pandas as pd

from plots.load import (FIGURES_DIR, cohorts, concat, exps, models, print_metrics, read_ml, read_ts, report_missing,
                        tree_models)

PREFIX = "full_run"


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ML models (cell 28)
    combined_df = concat([read_ml(dataset, cohort, model, PREFIX) for (dataset, cohort) in exps for model in tree_models])
    table_ml = combined_df.groupby(["cohort", "Model"])[["AUC-ROC", "AUC-PRC", "f1_score"]].agg(
        ["mean", lambda x: x.std(ddof=1)]
    ).rename(columns={"<lambda_0>": "std"}).round(4).reindex(
        pd.MultiIndex.from_product(
            [cohorts, ['logistic_regression', 'random_forest', 'gradient_boosting', 'xgboost', 'catboost']],
            names=["cohort", "model_type"]
        )
    )
    print(table_ml)
    table_ml.to_csv(FIGURES_DIR / "table1_ml.csv")

    # TS models (cell 32)
    combined_df = concat([read_ts(dataset, cohort, model, PREFIX) for model in models for (dataset, cohort) in exps])
    table_ts = (combined_df
     .groupby(["cohort", "model_type"])[print_metrics]
     .agg(["mean", "std"])
     .reindex(pd.MultiIndex.from_product([cohorts, models], names=["cohort", "model_type"]))
    )
    print(table_ts.round(3))
    table_ts.round(3).to_csv(FIGURES_DIR / "table1_ts.csv")

    report_missing()
    print(f"\nSaved {FIGURES_DIR / 'table1_ml.csv'} and {FIGURES_DIR / 'table1_ts.csv'}")


if __name__ == "__main__":
    main()
