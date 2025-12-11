run_commands() {
    template="$1"
    for model in gru tcn lstm sand grud interpnet strats; do
        for fold in 0 1 2 3 4; do    
            eval "$template --dataset MIMIC_IV --cohort mimic_cohort_NF_30_days --fold $fold --model_type $model --feature_threshold"
            eval "$template --dataset UKEr --cohort uker_cohort_NF_30_days --fold $fold --model_type $model"
            eval "$template --dataset MIMIC_IV --cohort mimic_cohort_aplasia_45_days --fold $fold --model_type $model --feature_threshold"
            eval "$template --dataset UKEr --cohort uker_cohort_aplasia_45_days --fold $fold --model_type $model"
        done
    done
}

template="python -m ts_model_training.main"
run_commands "$template --static_threshold 0 --hid_dim_demo 32 --prefix test --grid nested  --config ts_config_params.yaml"

