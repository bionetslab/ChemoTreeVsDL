#!/bin/bash
# Aggregation intervals (MIMIC-IV only): nested 5x3 CV with grid search on VMD features binned at 3, 6 and 12 hours
# (24h is the full run). Non-temporal models and TS models for regular time series.
#   ML models: minority oversampling, grid from config/model_params.yaml
#   TS models (incl. MLP): grid from config/ts_config_params.yaml
# Results are saved with prefix agg_int_<hours>.
# Usage: bash experiments/exp_agg_intervals.sh [cohort|all] [model|ml|ts|all] [interval|all]
#   bash experiments/exp_agg_intervals.sh                 # everything
#   bash experiments/exp_agg_intervals.sh all ml          # only ML models (run in parallel with 'ts')
#   bash experiments/exp_agg_intervals.sh all ts          # only TS models
#   bash experiments/exp_agg_intervals.sh mimic_cohort_NF_30_days gru 6
cd "$(dirname "$0")/.."

cohorts="mimic_cohort_aplasia_45_days mimic_cohort_NF_30_days"
ml_models="logistic_regression random_forest gradient_boosting xgboost catboost"
ts_models="mlp gru lstm tcn sand"
intervals="3 6 12"
[ -n "$1" ] && [ "$1" != all ] && cohorts="$1"
case "${2:-all}" in
    all) ;;
    ml)  ts_models="" ;;
    ts)  ml_models="" ;;
    *)   if [[ " $ml_models " == *" $2 "* ]]; then ml_models="$2"; ts_models=""; else ml_models=""; ts_models="$2"; fi ;;
esac
[ -n "$3" ] && [ "$3" != all ] && intervals="$3"

for cohort in $cohorts; do
    for interval in $intervals; do
        # ML models: feature matrices are stored once per cohort, so they are rebuilt for each interval
        if [ -n "$ml_models" ]; then
            python -m training.ml.ml_main \
                --action prepare_ml_matrices \
                --dataset MIMIC_IV \
                --cohort $cohort \
                --feature_selection 1 \
                --agg_interval $interval

            for model in $ml_models; do
                python -m training.ml.ml_main \
                    --action train \
                    --dataset MIMIC_IV \
                    --cohort $cohort \
                    --model_type $model \
                    --feature_selection 1 \
                    --agg_interval $interval \
                    --oversampling minority \
                    --grid_search 1 \
                    --load_data 1 \
                    --prefix agg_int_$interval
            done
        fi

        # TS models
        for model in $ts_models; do
            for fold in 0 1 2 3 4; do
                python -m training.ts.main \
                    --dataset MIMIC_IV \
                    --cohort $cohort \
                    --fold $fold \
                    --model_type $model \
                    --feature_threshold \
                    --static_threshold 0 \
                    --hid_dim_demo 32 \
                    --grid nested \
                    --agg_int $interval \
                    --prefix agg_int_$interval
            done
        done
    done
done
