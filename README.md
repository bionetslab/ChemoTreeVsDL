# Chemotherapy Side Effect Prediction Pipeline

This repository provides a reproducible pipeline to:

- Extract cohorts from **MIMIC-IV** and a **Code dataset from a MII node** (Erlangen University Hospital, UKEr) for patients with cancer diagnoses undergoing chemotherapy.  
- Train **classical machine learning** and **temporal deep learning** models to predict chemotherapy-induced side effects using longitudinal lab data and demographic features.

## Models

**Classical models**  
- Tree-based: Random Forest (RF), Gradient Boosting (GB), Extreme Gradient Boosting (XGB), CatBoost (CatB)  
- Other: Logistic Regression (LR), Multilayer Perceptron (MLP)  

**Temporal models**  
- Regular time series: GRU, LSTM, Temporal Convolutional Network (TCN), SAnD  
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

- **ml_model_training/** – Training scripts for classical models
- **ts_model_training/** – Training scripts for deep learning models


## Usage

1. Create the environment:

```bash
conda env create -f environment.yml
conda activate flabnet_ml_pipeline_env
```
### 2. Train models

#### Classical models
Run the training script with:

```bash
bash train_ml_models.sh [mimic|uker] [prepare|train|full_run]
```

- mimic – for MIMIC-IV cohorts
- uker – for Erlangen University Hospital cohorts

Workflow options:
- prepare – prepare inputs
- train – train models on prepared inputs
- full_run – prepare inputs and train models

Example:

```bash
bash train_ml_models.sh mimic full_run
```

#### Temporal deep models
To run temporal deep models on already prepared inputs:

```bash
bash train_ts_models.sh
```
