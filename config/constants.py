from pathlib import Path
import os
import sys
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTING_FILES_DIR = PROJECT_ROOT / 'supporting_files'
SAVED_DATA_DIR = PROJECT_ROOT / 'saved_data'  # saved_data/<dataset>/{cohorts,features,folds,...}
TOP_FEATURES_PATH = SUPPORTING_FILES_DIR / 'mimic_top_features_cancer_chemo.csv'
MIMIC_DIR = os.environ.get('MIMIC_DIR', '/path/to/mimiciv/3.1/')
MIMIC_LABS_DIR = MIMIC_DIR +'hosp/labevents.csv.gz'
UKER_DIR = os.environ.get('UKER_DIR', '/path/to/ped-data.db')



CONFIG_DIR = Path(__file__).resolve().parents[0]
MODEL_PARAM_PATH = CONFIG_DIR / 'model_params.yaml'
BEST_PARAM_PATH = CONFIG_DIR / 'ml_config_params_best.yaml'

def load_model_params():
    with open(MODEL_PARAM_PATH, "r") as f:
        return yaml.safe_load(f)

def load_best_params():
    with open(BEST_PARAM_PATH, "r") as f:
        return yaml.safe_load(f)

MODEL_PARAMS = load_model_params()
BEST_PARAMS = load_best_params()


num_folds = 5
num_inner_folds = 3

RANDOM_SEED = 42

