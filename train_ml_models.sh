#!/bin/bash
prefix="${3:-}"
feature_types=("VMD") #("D" "VD" "VM" "MD") #"standard" "V" "M" "D" "VD" "VM" "MD"
models=("CatBoost") #("Random Forest" "Logistic Regression" "Xgboost" "CatBoost" "Gradient Boosting")
agg_intervals=("24") # "12" "6" "3")


# Common function to run the command safely (no eval)
run_command() {
    local dataset="$1"
    local cohort="$2"
    local model_type="$3"
    local feat_type="$4"
    local feature_selection="$5"
    local agg_interval="$6"
    local action="$7"

    echo ">>> Running: $dataset | $cohort | $model_type | $feat_type | $agg_interval | $action"

    python -m ml_model_training.ml_main \
        --dataset "$dataset" \
        --cohort "$cohort" \
        --model_type "$model_type" \
        --features LAB DEMO \
        --feature_method concatenate \
        --oversampling minority \
        --grid_search 1 \
        --num_folds 5 \
        --feature_selection "$feature_selection" \
        --load_data 1 \
        --feat_type "$feat_type" \
        --agg_interval "$agg_interval" \
        --prefix "$prefix" \
        --action "$action"
}

# ---------- MIMIC ----------
run_mimic() {
    action="$1"
    echo "Running MIMIC pipeline: $action"
    
    dataset="MIMIC_IV"
    cohorts=("mimic_cohort_aplasia_45_days" "mimic_cohort_NF_30_days")
    feature_selection="1"

    for cohort in "${cohorts[@]}"; do
        for feat_type in "${feature_types[@]}"; do
            for interval in "${agg_intervals[@]}"; do
                if [ "$action" = "prepare" ]; then
                    echo "=========================================="
                    echo "Preparing: $cohort, $feat_type, $interval"
                    run_command "$dataset" "$cohort" "Random Forest" "$feat_type" "$feature_selection" "$interval" "$action"
                else
                    for model in "${models[@]}"; do
                        echo "=========================================="
                        echo "Training: $model, $cohort, $feat_type, $interval"
                        run_command "$dataset" "$cohort" "$model" "$feat_type" "$feature_selection" "$interval" "$action"
                    done
                fi
            done
        done
    done
}

# ---------- UKER ----------
run_uker() {
    action="$1"
    echo "Running UKEr pipeline: $action"
    
    dataset="UKEr"
    cohorts=("uker_cohort_NF_30_days" "uker_cohort_aplasia_45_days")
    feature_selection="0"

    for cohort in "${cohorts[@]}"; do
        for feat_type in "${feature_types[@]}"; do
            for interval in "${agg_intervals[@]}"; do
                if [ "$action" = "prepare" ]; then
                    echo "=========================================="
                    echo "Preparing: $cohort, $feat_type, $interval"
                    run_command "$dataset" "$cohort" "Random Forest" "$feat_type" "$feature_selection" "$interval" "$action"
                else
                    for model in "${models[@]}"; do
                        echo "=========================================="
                        echo "Training: $model, $cohort, $feat_type, $interval"
                        run_command "$dataset" "$cohort" "$model" "$feat_type" "$feature_selection" "$interval" "$action"
                    done
                fi
            done
        done
    done
}

# ---------- MAIN ----------
if [ "$1" = "mimic" ]; then
    if [ "$2" = "full_run" ]; then
        run_mimic "prepare"
        echo "MIMIC data preparation completed. Starting training..."
        run_mimic "train"
    else
        run_mimic "$2"
    fi
elif [ "$1" = "uker" ]; then
    if [ "$2" = "full_run" ]; then
        run_uker "prepare"
        echo "UKEr data preparation completed. Starting training..."
        run_uker "train"
    else
        run_uker "$2"
    fi
else
    echo "Usage: $0 [mimic|uker] [prepare|train|full_run] [prefix]"
    echo "  mimic prepare: Prepare MIMIC data only"
    echo "  mimic train:   Train MIMIC models only"
    echo "  mimic full_run: Prepare + train MIMIC"
    echo "  uker prepare:  Prepare UKEr data only"
    echo "  uker train:    Train UKEr models only"
    echo "  uker full_run: Prepare + train UKEr"
    echo "  prefix:        Optional prefix for output files (default: empty)"
fi
