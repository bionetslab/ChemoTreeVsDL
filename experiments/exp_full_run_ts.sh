#!/bin/bash
# Full run, TS models (incl. MLP): nested 5x3 CV with grid search (config/ts_config_params.yaml), VMD features, 24h bins
# Usage: bash experiments/exp_full_run_ts.sh [cohort|all] [model|all] [prefix]
#   bash experiments/exp_full_run_ts.sh
#   bash experiments/exp_full_run_ts.sh uker_cohort_aplasia_45_days gru
cd "$(dirname "$0")/.."

cohorts="mimic_cohort_aplasia_45_days mimic_cohort_NF_30_days uker_cohort_aplasia_45_days uker_cohort_NF_30_days"
models="mlp gru lstm tcn sand grud interpnet strats"
[ -n "$1" ] && [ "$1" != all ] && cohorts="$1"
[ -n "$2" ] && [ "$2" != all ] && models="$2"
prefix="${3:-full_run}"

for cohort in $cohorts; do
    # MIMIC-IV: 100 most frequent labs; UKEr: all labs
    if [[ "$cohort" == mimic_* ]]; then dataset=MIMIC_IV; feature_threshold="--feature_threshold"; else dataset=UKEr; feature_threshold=""; fi

    for model in $models; do
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
