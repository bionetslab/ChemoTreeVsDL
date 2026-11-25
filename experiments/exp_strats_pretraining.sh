#!/bin/bash
# STraTS pretraining: best configuration per cohort (config/ts_config_params_best.yaml), 5 folds, epochs via early stopping.
# Pretraining uses STraTS' forecasting task (no labels), either on the training data of each fold ("cohort")
# or once on all admissions of the dataset ("all"), then fine-tuning on the target either of all layers or
# only of the last layers (--freeze).
#
# Steps and result prefixes:
#   none              pt_none                                no pretraining
#   pretrain_cohort   pt_cohort                              pretraining on the cohort (checkpoints)
#   finetune_cohort   pt_cohort_all_weights, pt_cohort_freeze  fine-tuning after pretrain_cohort
#   pretrain_all      pt_all_<nf|aplasia>                    pretraining on all admissions (checkpoints,
#                                                            experiments/configs/ts_config_params_pt_<dataset>_<target>.yaml)
#   finetune_all      pt_all_all_weights, pt_all_freeze      fine-tuning after pretrain_all
# none, pretrain_cohort and pretrain_all are independent and can run in parallel;
# finetune_cohort needs pretrain_cohort, finetune_all needs pretrain_all.
#
# Usage: bash experiments/exp_strats_pretraining.sh [cohort|all] [step|all]
#   bash experiments/exp_strats_pretraining.sh                        # all steps in order
#   bash experiments/exp_strats_pretraining.sh all pretrain_all
#   bash experiments/exp_strats_pretraining.sh uker_cohort_NF_30_days finetune_cohort
cd "$(dirname "$0")/.."

cohorts="mimic_cohort_aplasia_45_days mimic_cohort_NF_30_days uker_cohort_aplasia_45_days uker_cohort_NF_30_days"
steps="pretrain_all none pretrain_cohort finetune_cohort finetune_all"
[ -n "$1" ] && [ "$1" != all ] && cohorts="$1"
[ -n "$2" ] && [ "$2" != all ] && steps="$2"
run_step() { [[ " $steps " == *" $1 "* ]]; }

for cohort in $cohorts; do
    # MIMIC-IV: 100 most frequent labs; UKEr: all labs
    if [[ "$cohort" == mimic_* ]]; then
        dataset=MIMIC_IV; all_cohort=mimic_all; feature_threshold="--feature_threshold"
    else
        dataset=UKEr; all_cohort=uker_all; feature_threshold=""
    fi
    if [[ "$cohort" == *_NF_* ]]; then target=nf; else target=aplasia; fi
    results=saved_data/$dataset/results

    # pretraining on all admissions of the dataset (once per target, fold 0)
    if run_step pretrain_all; then
        python -m training.ts.main \
            --dataset $dataset \
            --cohort $all_cohort \
            --fold 0 \
            --model_type strats \
            $feature_threshold \
            --static_threshold 0 \
            --hid_dim_demo 32 \
            --pretrain \
            --grid best_epochs \
            --config_path experiments/configs/ts_config_params_pt_${all_cohort%_all}_$target.yaml \
            --prefix pt_all_$target
    fi

    for fold in 0 1 2 3 4; do
        # no pretraining
        if run_step none; then
            python -m training.ts.main \
                --dataset $dataset \
                --cohort $cohort \
                --fold $fold \
                --model_type strats \
                $feature_threshold \
                --static_threshold 0 \
                --hid_dim_demo 32 \
                --grid best_epochs \
                --config_path config/ts_config_params_best.yaml \
                --prefix pt_none
        fi

        # pretraining on the cohort (training data of this fold)
        if run_step pretrain_cohort; then
            python -m training.ts.main \
                --dataset $dataset \
                --cohort $cohort \
                --fold $fold \
                --model_type strats \
                $feature_threshold \
                --static_threshold 0 \
                --hid_dim_demo 32 \
                --pretrain \
                --grid best_epochs \
                --config_path config/ts_config_params_best.yaml \
                --prefix pt_cohort
        fi

        # fine-tuning after cohort pretraining: all layers, then last layers only
        if run_step finetune_cohort; then
            for freeze in "" "--freeze"; do
                if [ -z "$freeze" ]; then prefix=pt_cohort_all_weights; else prefix=pt_cohort_freeze; fi
                python -m training.ts.main \
                    --dataset $dataset \
                    --cohort $cohort \
                    --fold $fold \
                    --model_type strats \
                    $feature_threshold \
                    --static_threshold 0 \
                    --hid_dim_demo 32 \
                    --grid best_epochs \
                    --config_path config/ts_config_params_best.yaml \
                    --load_ckpt_path $results/$cohort/time_series/pretrain/strats/pt_cohort/fold_$fold/grid_best_epochs \
                    $freeze \
                    --prefix $prefix
            done
        fi

        # fine-tuning after pretraining on all admissions: all layers, then last layers only
        if run_step finetune_all; then
            for freeze in "" "--freeze"; do
                if [ -z "$freeze" ]; then prefix=pt_all_all_weights; else prefix=pt_all_freeze; fi
                python -m training.ts.main \
                    --dataset $dataset \
                    --cohort $cohort \
                    --fold $fold \
                    --model_type strats \
                    $feature_threshold \
                    --static_threshold 0 \
                    --hid_dim_demo 32 \
                    --grid best_epochs \
                    --config_path config/ts_config_params_best.yaml \
                    --load_ckpt_path $results/$all_cohort/time_series/pretrain/strats/pt_all_$target/fold_0/grid_best_epochs \
                    $freeze \
                    --prefix $prefix
            done
        fi
    done
done
