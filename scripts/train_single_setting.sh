#!/bin/bash
# Train one model on one cohort.
# Usage: bash scripts/train_model.sh <cohort> <model> [prefix]
#   bash scripts/train_model.sh mimic_cohort_aplasia_45_days "Random Forest"
#   bash scripts/train_model.sh uker_cohort_NF_30_days gru
# ML models need prepared matrices first: bash scripts/train_ml.sh <mimic|uker> prepare_ml_matrices
cohort="$1"
model="$2"
prefix="${3:-$(date +%d%m%y)}"

if [[ "$cohort" == mimic_* ]]; then dataset="MIMIC_IV"; else dataset="UKEr"; fi

case "$model" in
    gru|tcn|lstm|sand|grud|interpnet|strats|mlp)
        extra=""; [ "$dataset" = "MIMIC_IV" ] && extra="--feature_threshold"
        for fold in 0 1 2 3 4; do
            python -m training.ts.main \
                --dataset "$dataset" \
                --cohort "$cohort" \
                --fold "$fold" \
                --model_type "$model" \
                --grid nested \
                --config_path config/ts_config_params.yaml \
                --static_threshold 0 \
                --hid_dim_demo 32 \
                --prefix "$prefix" \
                $extra
        done ;;
    *)
        feature_selection=0; [ "$dataset" = "MIMIC_IV" ] && feature_selection=1
        python -u -m training.ml.ml_main \
            --action train \
            --dataset "$dataset" \
            --cohort "$cohort" \
            --model_type "$model" \
            --features LAB DEMO \
            --feature_method concatenate \
            --feat_type VMD \
            --agg_interval 24 \
            --feature_selection "$feature_selection" \
            --oversampling minority \
            --grid_search 1 \
            --num_folds 5 \
            --load_data 1 \
            --prefix "$prefix" ;;
esac
