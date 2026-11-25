#!/bin/bash
# Multi-seed run: 5 CV split seeds x 5 folds with fixed best parameters per cohort, VMD features, 24h bins.
# Used for the statistical tests between models.
#   ML models: config/ml_config_params_best.yaml, minority oversampling
#   TS models (incl. MLP and STraTS without pretraining): config/ts_config_params_best.yaml
# Usage: bash experiments/exp_multi_seed.sh [cohort|all] [model|ml|ts|all] [prefix]
#   bash experiments/exp_multi_seed.sh                    # everything
#   bash experiments/exp_multi_seed.sh all ml             # only ML models (run in parallel with 'ts')
#   bash experiments/exp_multi_seed.sh all ts             # only TS models
#   bash experiments/exp_multi_seed.sh mimic_cohort_NF_30_days catboost
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
prefix="${3:-multi_seed}"

# 5 CV split seeds drawn pseudo-randomly from 1-100 with Python's random module seeded with 42: 42 82 15 4 95
seeds=$(python -c "import random; random.seed(42); print(42, *[random.randint(1, 100) for _ in range(4)])")

for cohort in $cohorts; do
    # MIMIC-IV: 100 most frequent labs; UKEr: all labs
    if [[ "$cohort" == mimic_* ]]; then
        dataset=MIMIC_IV; feature_selection=1; feature_threshold="--feature_threshold"
    else
        dataset=UKEr; feature_selection=0; feature_threshold=""
    fi

    # CV folds per seed (saved_data/<dataset>/multi_seed_training/folds/<cohort>/seed_<n>/)
    python -m preprocessing.create_cv_folds \
        --dataset $dataset \
        --cohorts $cohort \
        --seeds $seeds

    # ML models: feature matrices per seed, then train
    if [ -n "$ml_models" ]; then
        for seed in $seeds; do
            python -m training.ml.ml_main \
                --action prepare_ml_matrices \
                --dataset $dataset \
                --cohort $cohort \
                --feature_selection $feature_selection \
                --split_seed $seed

            for model in $ml_models; do
                python -m training.ml.ml_main \
                    --action train \
                    --dataset $dataset \
                    --cohort $cohort \
                    --model_type $model \
                    --feature_selection $feature_selection \
                    --oversampling minority \
                    --grid_search 0 \
                    --load_data 1 \
                    --split_seed $seed \
                    --prefix $prefix
            done
        done
    fi

    # TS models
    for model in $ts_models; do
        for seed in $seeds; do
            for fold in 0 1 2 3 4; do
                python -m training.ts.main \
                    --dataset $dataset \
                    --cohort $cohort \
                    --fold $fold \
                    --model_type $model \
                    $feature_threshold \
                    --static_threshold 0 \
                    --hid_dim_demo 32 \
                    --grid best \
                    --config_path config/ts_config_params_best.yaml \
                    --split_seed $seed \
                    --prefix $prefix
            done
        done
    done
done
