#!/bin/bash
# Full run: nested 5x3 CV with grid search, VMD features, 24h bins.
#   ML models: minority oversampling, grid from config/model_params.yaml
#   TS models (incl. MLP): grid from config/ts_config_params.yaml
# Usage: bash experiments/exp_full_run.sh [cohort|all] [model|ml|ts|all] [prefix]
#   bash experiments/exp_full_run.sh                      # everything
#   bash experiments/exp_full_run.sh all ml               # only ML models (run in parallel with 'ts')
#   bash experiments/exp_full_run.sh all ts               # only TS models
#   bash experiments/exp_full_run.sh mimic_cohort_NF_30_days catboost
cd "$(dirname "$0")/.."

cohorts="mimic_cohort_aplasia_45_days mimic_cohort_NF_30_days uker_cohort_aplasia_45_days uker_cohort_NF_30_days"
ml_models="logistic_regression random_forest gradient_boosting xgboost catboost"
ts_models="mlp gru lstm tcn sand grud interpnet strats"
[ -n "$1" ] && [ "$1" != all ] && cohorts="$1"
case "${2:-all}" in
    all) ;;
    ml)  ts_models="" ;;
    ts)  ml_models="" ;;
    *)   if [[ " $ml_models " == *" $2 "* ]]; then ml_models="$2"; ts_models=""; else ml_models=""; ts_models="$2"; fi ;;
esac
prefix="${3:-full_run}"

for cohort in $cohorts; do
    # MIMIC-IV: 100 most frequent labs; UKEr: all labs
    if [[ "$cohort" == mimic_* ]]; then
        dataset=MIMIC_IV; feature_selection=1; feature_threshold="--feature_threshold"
    else
        dataset=UKEr; feature_selection=0; feature_threshold=""
    fi

    # ML models: feature matrices, then train
    if [ -n "$ml_models" ]; then
        python -m training.ml.ml_main \
            --action prepare_ml_matrices \
            --dataset $dataset \
            --cohort $cohort \
            --feature_selection $feature_selection

        for model in $ml_models; do
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
    fi

    # TS models
    for model in $ts_models; do
        for fold in 0 1 2 3 4; do
            python -m training.ts.main \
                --dataset $dataset \
                --cohort $cohort \
                --fold $fold \
                --model_type $model \
                $feature_threshold \
                --static_threshold 0 \
                --hid_dim_demo 32 \
                --grid nested \
                --prefix $prefix
        done
    done
done
