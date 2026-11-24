import argparse
import sqlite3
import pandas as pd
from config.constants import UKER_DIR, PROJECT_ROOT

COHORTS = {"NF": "uker_cohort_NF_30_days", "aplasia": "uker_cohort_aplasia_45_days"}
SAVED_DATA_DIR = PROJECT_ROOT / "saved_data" / "UKEr"
FEATURES_DIR = SAVED_DATA_DIR / "features"


def read_SQL(data_path, query, parameters):
    with sqlite3.connect(data_path) as conn:
        return pd.read_sql(query, conn, params=parameters)


def extract_admission_labs(cohort, target_cohort, days_before_discharge):
    fid_list = cohort.hadm_id.unique().tolist()
    placeholders = ", ".join(["?"] * len(fid_list))
    query = f"""
    SELECT l.pid, f.fid, l.loinc, l.'alter', l.wert, f.aufnahme_alter, f.entlassung_alter
    FROM lab l
    JOIN fall f ON l.pid = f.pid
    JOIN pat p ON l.pid = p.pid
    WHERE l.'alter' BETWEEN (f.entlassung_alter - ?) AND f.entlassung_alter
    AND f.fid IN ({placeholders})
    ORDER BY f.pid, l.'alter'
    """
    admission_labs = read_SQL(UKER_DIR, query, [days_before_discharge] + fid_list)
    admission_labs.columns = ["subject_id", "hadm_id", "itemid", "date", "value", "admittime", "dischtime"]
    admission_labs.to_csv(FEATURES_DIR / f"{target_cohort}_admissions_labs_{days_before_discharge}_days.csv.gz", index=False)
    print(f"Labs saved to {FEATURES_DIR}")
    return admission_labs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True, choices=COHORTS.values())
    parser.add_argument("--days_before_discharge", type=int, default=14)
    args = parser.parse_args()

    target_cohort = args.cohort
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(SAVED_DATA_DIR / "cohorts" / f"{target_cohort}.csv.gz")
    print("UKEr Dataset")
    print(f"Extracting labs in the observation window of {args.days_before_discharge} days before discharge for {args.cohort} cohort. ")
    extract_admission_labs(cohort, target_cohort, args.days_before_discharge)


if __name__ == "__main__":
    main()
