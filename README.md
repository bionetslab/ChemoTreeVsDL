# ChemoTreeVsDL - Chemotherapy Side Effect Prediction Pipeline

This repository provides a reproducible pipeline to:

- Extract cohorts of patients with cancer diagnoses undergoing chemotherapy from **MIMIC-IV** and the pediatric **MII Core Data Set** from Uniklinikum Erlangen (UKEr).  
- Train **classical machine learning** and **temporal deep learning** models to predict chemotherapy-induced side effects using longitudinal lab data and demographic features.

## Models

**Classical models**  
- Tree-based: Random Forest (RF), Gradient Boosting (GB), Extreme Gradient Boosting (XGB), CatBoost (CatB)  
- Other: Logistic Regression (LR), Multilayer Perceptron (MLP)  

**Temporal models**  
- Regular time series: GRU, LSTM, TCN, SAnD  
- Irregular time series: GRU-D, InterpNet, STraTS  

## Cohorts

- Patients with a cancer diagnosis and at least one chemotherapy procedure.  
- **Aplasia:** defined by transfusion procedures or low absolute neutrophil count.  
- **Neutropenic fever:** concurrent neutropenia and fever diagnoses.  
- **Prediction target:** onset of the condition within the prediction window (45 days after discharge for aplasia, 30 days for neutropenic fever) and before the next chemotherapy cycle.  
- **Observation window:** 14 days prior to discharge.  

## Structure

- **config/** – Configuration files
  - `constants.py` – Shared constants and paths
  - `model_params.yaml` – Grid search parameters for classical models
  - `ml_config_params_best.yaml` – Best parameters/cohort for classical models
  - `ts_config_params.yaml` – Grid search parameters for neural network models
  - `ts_config_params_best.yaml` – Best parameters/cohort for neural network models

- **cohort_and_features/** – Cohort definition notebooks and lab feature extraction scripts
  - `mimic_iv/` – MIMIC-IV cohorts (`cohort_aplasia.ipynb`, `cohort_neutropenic_fever.ipynb`), `select_top_labs.py`, `extract_features.py`
  - `uker/` – UKEr cohorts (`cohort_aplasia.ipynb`, `cohort_neutropenic_fever.ipynb`), `extract_features.py`
- **preprocessing/** – CV fold splitting and feature matrix generation
- **training/ml/** – Training code for classical models
- **training/ts/** – Training code for deep learning models
- **scripts/** – Shell scripts to run the training (`train_ml.sh`, `train_ts.sh`)
- **notebooks/** – Interactive preprocessing and training notebook
- **supporting_files/** – Code lists used for cohort definition
- **saved_data/** – Generated outputs per dataset: `saved_data/<MIMIC_IV|UKEr>/{cohorts,features,folds,...}` (created at runtime)


## Usage

All commands are run from the repository root.

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate chemotreevsdl
```

### 2. Set the data paths (e.g. in your `~/.bashrc`)

```bash
export MIMIC_DIR=/path/to/mimiciv/3.1/
export UKER_DIR=/path/to/ped-data.db
```

### 3. Build the cohorts

Run the cohort notebooks (from their own directory). Each one writes the cohort to `saved_data/<dataset>/cohorts/`.

| Dataset | Notebooks | Cohorts created |
|---|---|---|
| MIMIC-IV | `cohort_and_features/mimic_iv/cohort_aplasia.ipynb`, `cohort_and_features/mimic_iv/cohort_neutropenic_fever.ipynb` | `mimic_cohort_aplasia_45_days`, `mimic_cohort_NF_30_days` |
| UKEr | `cohort_and_features/uker/cohort_aplasia.ipynb`, `cohort_and_features/uker/cohort_neutropenic_fever.ipynb` | `uker_cohort_aplasia_45_days`, `uker_cohort_NF_30_days` |

### 4. Extract lab features

**MIMIC-IV**

```bash
# select the 100 most frequent labs (used for feature selection in MIMIC-IV)
python -m cohort_and_features.mimic_iv.select_top_labs

python -m cohort_and_features.mimic_iv.extract_features --cohort mimic_cohort_aplasia_45_days
python -m cohort_and_features.mimic_iv.extract_features --cohort mimic_cohort_NF_30_days
```

**UKEr**

```bash
python -m cohort_and_features.uker.extract_features --cohort auker_cohort_aplasia_45_days
python -m cohort_and_features.uker.extract_features --cohort uker_cohort_NF_30_days
```

The observation window can be changed with `--days_before_discharge` (default: 14).

### 5. Split cross-validation folds

```bash
# MIMIC-IV
python -m preprocessing.create_cv_folds --dataset MIMIC_IV --cohorts mimic_cohort_aplasia_45_days mimic_cohort_NF_30_days

# UKEr
python -m preprocessing.create_cv_folds --dataset UKEr --cohorts uker_cohort_aplasia_45_days uker_cohort_NF_30_days
```

### 6. Train models

#### Classical models

Classical models are trained in two steps: first build the feature matrices once per cohort, then train any number of models on them.

```bash
bash scripts/train_ml.sh <mimic|uker> prepare_ml_matrices [cohort|all]
bash scripts/train_ml.sh <mimic|uker> train   [cohort|all] [model|all] [prefix]
```

- `mimic` – MIMIC-IV cohorts, `uker` – Uniklinikum Erlangen cohorts
- `prepare_ml_matrices` – builds the per-fold feature matrices in `saved_data/<dataset>/ml_feature_matrices/<cohort>/` and records the settings and input files used in `config.json`
- `train` – trains the models on the prepared matrices. It stops with a message if the matrices are missing, were prepared with other settings, or if the folds or lab files changed since `prepare_ml_matrices`; run `prepare_ml_matrices` again in that case.
- `cohort` (optional) – e.g. `mimic_cohort_aplasia_45_days` (default: all cohorts of the dataset)
- `model` (optional) – one of `"Random Forest"`, `"Logistic Regression"`, `Xgboost`, `CatBoost`, `"Gradient Boosting"` (default: all)
- `prefix` (optional) – prefix for result folders (default: today's date, `ddmmyy`)

Examples:

```bash
bash scripts/train_ml.sh mimic prepare_ml_matrices
bash scripts/train_ml.sh mimic train
bash scripts/train_ml.sh uker prepare_ml_matrices uker_cohort_NF_30_days
bash scripts/train_ml.sh uker train uker_cohort_NF_30_days "Random Forest"
```

Several `train` jobs can run in parallel on the same prepared matrices. Do not run `prepare_ml_matrices` for a cohort while `train` jobs on that cohort are still running.

#### Temporal deep models

```bash
bash scripts/train_ts.sh <mimic|uker> [cohort|all] [model|all] [prefix]
```

- `model` (optional) – one of `gru`, `tcn`, `lstm`, `sand`, `grud`, `interpnet`, `strats`, `mlp` (default: all)

No separate prepare step is needed: the time-series input is built automatically on first use.

Examples:

```bash
bash scripts/train_ts.sh mimic
bash scripts/train_ts.sh uker uker_cohort_NF_30_days gru
```

Each run covers the 5 folds with nested grid search.
