import argparse
import pandas as pd
import utils.lab_utils
from config.constants import MIMIC_DIR, PROJECT_ROOT

COHORTS = {"NF": "mimic_cohort_NF_30_days", "aplasia": "mimic_cohort_aplasia_45_days"}
SAVED_DATA_DIR = PROJECT_ROOT / "saved_data" / "MIMIC_IV"
FEATURES_DIR = SAVED_DATA_DIR / "features"


def extract_cohort_labs(cohort, target_cohort):
    lab = utils.lab_utils.extract_cohort_labs(MIMIC_DIR, cohort, "charttime", dtypes=None, usecols=None)
    lab = utils.lab_utils.drop_wrong_uom(lab, 0.95)
    lab = lab[["subject_id", "hadm_id", "charttime", "itemid", "valuenum"]]
    lab.to_csv(FEATURES_DIR / f"{target_cohort}_labs.csv.gz", index=False)
    return lab


def extract_admission_labs(cohort, lab, target_cohort, days_before_discharge):
    # remove H-I-L
    lab = lab[~lab["itemid"].isin([50934, 50947, 51678])]
    lab_results = cohort.apply(lambda x: utils.lab_utils.extract_admission_labs(x, days_before_discharge, lab), axis=1)
    admission_labs = pd.concat(lab_results.tolist(), ignore_index=True)
    admission_labs.columns = ["subject_id", "hadm_id", "date", "itemid", "value", "admittime", "dischtime", "lab_time_from_disch"]
    admission_labs.to_csv(FEATURES_DIR / f"{target_cohort}_admissions_labs_{days_before_discharge}_days.csv.gz", index=False)
    return admission_labs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True, choices=COHORTS.keys())
    parser.add_argument("--days_before_discharge", type=int, default=14)
    args = parser.parse_args()

    target_cohort = COHORTS[args.cohort]
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(SAVED_DATA_DIR / "cohorts" / f"{target_cohort}.csv.gz", parse_dates=["admittime", "dischtime"])

    lab = extract_cohort_labs(cohort, target_cohort)
    extract_admission_labs(cohort, lab, target_cohort, args.days_before_discharge)


if __name__ == "__main__":
    main()
