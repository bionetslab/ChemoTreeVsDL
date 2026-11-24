prefix="${2:-$(date +%d%m%y)}"

run_commands() {
    template="$1"
    dataset="$2"
    feature_threshold="$3"
    shift 3
    for model in gru tcn lstm sand grud interpnet strats mlp; do
        for fold in 0 1 2 3 4; do
            for cohort in "$@"; do
                eval "$template --dataset $dataset --cohort $cohort --fold $fold --model_type $model $feature_threshold"
            done
        done
    done
}

template="python -m training.ts.main --static_threshold 0 --hid_dim_demo 32 --grid nested --prefix $prefix --config config/ts_config_params.yaml"

if [ "$1" = "mimic" ]; then
    run_commands "$template" MIMIC_IV "--feature_threshold" mimic_cohort_NF_30_days mimic_cohort_aplasia_45_days
elif [ "$1" = "uker" ]; then
    run_commands "$template" UKEr "" uker_cohort_NF_30_days uker_cohort_aplasia_45_days
else
    echo "Usage: $0 [mimic|uker] [prefix]"
fi
