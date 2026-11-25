#!/bin/bash
# Full run, ML models: nested 5x3 CV with grid search, VMD features, 24h bins, minority oversampling
# Usage: bash experiments/exp_full_run_ml.sh [cohort|all] [model|all] [prefix]
#   bash experiments/exp_full_run_ml.sh
#   bash experiments/exp_full_run_ml.sh mimic_cohort_NF_30_days catboost
cd "$(dirname "$0")/.."

cohorts="mimic_cohort_aplasia_45_days mimic_cohort_NF_30_days uker_cohort_aplasia_45_days uker_cohort_NF_30_days"
models="logistic_regression random_forest gradient_boosting xgboost catboost"
[ -n "$1" ] && [ "$1" != all ] && cohorts="$1"
[ -n "$2" ] && [ "$2" != all ] && models="$2"
prefix="${3:-full_run}"

for cohort in $cohorts; do
    # MIMIC-IV: 100 most frequent labs; UKEr: all labs
    if [[ "$cohort" == mimic_* ]]; then dataset=MIMIC_IV; feature_selection=1; else dataset=UKEr; feature_selection=0; fi

    python -m training.ml.ml_main \
        --action prepare_ml_matrices \
        --dataset $dataset \
        --cohort $cohort \
        --feature_selection $feature_selection

    for model in $models; do
        python -m training.ml.ml_main \
            --action train \
            --dataset $dataset \
            --cohort $cohort \
            --model_type $model \
            --feature_selection $feature_selection \
            --oversampling minority \
            --grid_search 1 \
            --load_data 1 \
            --prefix $prefix
    done
done
