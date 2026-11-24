import pandas as pd
import utils.cohort_utils
import utils.lab_utils
from config.constants import MIMIC_DIR, TOP_FEATURES_PATH, SAVED_DATA_DIR


def select_most_frequent_labs(full_labs, itemids_to_remove, num_features=100):
    full_labs = full_labs[~full_labs['itemid'].isin(itemids_to_remove)]
    hadm_freq = (full_labs[["itemid","hadm_id"]]
        .drop_duplicates()
        .groupby(["itemid"])["hadm_id"]
        .size()
        .sort_values(ascending=False)
        .reset_index(name="hadm_freq")
    )
    hadm_freq["hadm_freq"] = hadm_freq["hadm_freq"] / full_labs["hadm_id"].nunique()
    print('-' * 93)
    print(f"{num_features} most frequent features of cancer chemo cohort selected and saved to {TOP_FEATURES_PATH}")
    return hadm_freq.head(num_features)['itemid'].tolist()


def main():
    data_dir = SAVED_DATA_DIR / 'MIMIC_IV'
    cohort_path = data_dir / 'cohorts' / 'mimic_cancer_chemo_cohort.csv.gz'
    labs_path = data_dir / 'features' / 'mimic_cancer_chemo_cohort_labs.csv.gz'

    if labs_path.exists():
        print(f"cancer chemo labs loaded from {labs_path}")
        labs = pd.read_csv(labs_path)
    else:
        if cohort_path.exists():
            print(f"cancer chemo cohort loaded from {cohort_path}")
            cancer_chemo_cohort = pd.read_csv(cohort_path)
        else:     
            print("Extracting Cancer Chemo Cohort...")
            print('-' * 93)
            cancer_cohort = utils.cohort_utils.extract_disease_cohort(
                MIMIC_DIR,
                subject_col='subject_id',
                visit_col='hadm_id',
                admit_col='admittime',
                disch_col='dischtime',
                disease_label='C',
            )
            cancer_chemo_cohort = utils.cohort_utils.extract_chemo_cohort(cancer_cohort, MIMIC_DIR)
            
        print('-' * 93)
        print("Extracting Labs...")
        
        labs = utils.lab_utils.extract_cohort_labs(MIMIC_DIR, cancer_chemo_cohort, 'charttime', dtypes=None, usecols=None)
        labs = utils.lab_utils.drop_wrong_uom(labs, 0.95)

            
    selected_features = select_most_frequent_labs(labs, [50934, 50947, 51678])
    pd.DataFrame({'itemid': selected_features}).to_csv(TOP_FEATURES_PATH, index=False)


if __name__ == "__main__":
    main()
