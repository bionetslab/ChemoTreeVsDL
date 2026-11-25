#!/bin/bash
# Feature types: nested 5x3 CV with grid search on the feature sets V, M, D, VD, VM, MD (VMD is the full run), 24h bins.
#   ML models: minority oversampling, grid from config/model_params.yaml
#   MLP (TS pipeline): grid from experiments/configs/ts_config_params_mlp_feat_<type>.yaml
# Results are saved with prefix feat_type_<type>.
# Usage: bash experiments/exp_feature_types.sh [cohort|all] [model|ml|ts|all] [feat_type|all]
#   bash experiments/exp_feature_types.sh                 # everything
#   bash experiments/exp_feature_types.sh all ml          # only ML models (run in parallel with 'ts')
#   bash experiments/exp_feature_types.sh all ts          # only MLP
#   bash experiments/exp_feature_types.sh mimic_cohort_NF_30_days catboost VD
cd "$(dirname "$0")/.."

cohorts="mimic_cohort_aplasia_45_days mimic_cohort_NF_30_days uker_cohort_aplasia_45_days uker_cohort_NF_30_days"
ml_models="logistic_regression random_forest gradient_boosting xgboost catboost"
ts_models="mlp"
feat_types="V M D VD VM MD"
[ -n "$1" ] && [ "$1" != all ] && cohorts="$1"
case "${2:-all}" in
    all) ;;
    ml)  ts_models="" ;;
    ts)  ml_models="" ;;
    *)   if [[ " $ml_models " == *" $2 "* ]]; then ml_models="$2"; ts_models=""; else ml_models=""; ts_models="$2"; fi ;;
esac
[ -n "$3" ] && [ "$3" != all ] && feat_types="$3"

for cohort in $cohorts; do
    # MIMIC-IV: 100 most frequent labs; UKEr: all labs
    if [[ "$cohort" == mimic_* ]]; then
        dataset=MIMIC_IV; feature_selection=1; feature_threshold="--feature_threshold"
    else
        dataset=UKEr; feature_selection=0; feature_threshold=""
    fi

    for feat_type in $feat_types; do
        # ML models: feature matrices are stored once per cohort, so they are rebuilt for each feature type
        if [ -n "$ml_models" ]; then
            python -m training.ml.ml_main \
                --action prepare_ml_matrices \
                --dataset $dataset \
                --cohort $cohort \
                --feature_selection $feature_selection \
                --feat_type $feat_type

            for model in $ml_models; do
                python -m training.ml.ml_main \
                    --action train \
                    --dataset $dataset \
                    --cohort $cohort \
                    --model_type $model \
                    --feature_selection $feature_selection \
                    --feat_type $feat_type \
                    --oversampling minority \
                    --grid_search 1 \
                    --load_data 1 \
                    --prefix feat_type_$feat_type
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
                    --config_path experiments/configs/ts_config_params_mlp_feat_$feat_type.yaml \
                    --prefix feat_type_$feat_type
            done
        done
    done
done
