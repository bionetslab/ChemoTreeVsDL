#!/bin/bash
# Dropped timestamps (MIMIC-IV only): TS models for irregular time series trained without exact times,
# i.e. all values of a day set to the same time (--drop_minutes). Nested 5x3 CV with grid search (config/ts_config_params.yaml).
# Timestamps retained is the full run. Results are saved with prefix drop_minutes.
# Usage: bash experiments/exp_drop_timestamps.sh [cohort|all] [model|all]
#   bash experiments/exp_drop_timestamps.sh
#   bash experiments/exp_drop_timestamps.sh mimic_cohort_NF_30_days grud
cd "$(dirname "$0")/.."

cohorts="mimic_cohort_aplasia_45_days mimic_cohort_NF_30_days"
models="grud strats"
[ -n "$1" ] && [ "$1" != all ] && cohorts="$1"
[ -n "$2" ] && [ "$2" != all ] && models="$2"

for cohort in $cohorts; do
    for model in $models; do
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
                --drop_minutes \
                --prefix drop_minutes
        done
    done
done
