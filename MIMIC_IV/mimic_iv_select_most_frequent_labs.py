import pickle
import utils.cohort_utils
import utils.lab_utils
from config.constants import MIMIC_DIR, SUPPORTING_FILES_DIR


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
    return hadm_freq.head(num_features)['itemid'].tolist()


def main():
    cancer_cohort = utils.cohort_utils.extract_disease_cohort(
        MIMIC_DIR,
        subject_col='subject_id',
        visit_col='hadm_id',
        admit_col='admittime',
        disch_col='dischtime',
        disease_label='C',
    )
    cancer_chemo_cohort = utils.cohort_utils.extract_chemo_cohort(cancer_cohort, MIMIC_DIR)
    labs = utils.lab_utils.extract_cohort_labs(MIMIC_DIR, cancer_chemo_cohort, 'charttime', dtypes=None, usecols=None)
    labs = utils.lab_utils.drop_wrong_uom(labs, 0.95)
    selected_features = select_most_frequent_labs(labs, [50934, 50947, 51678])
    with open(SUPPORTING_FILES_DIR / 'mimic_top_features_cancer_chemo.pkl', 'wb') as f:
        pickle.dump(selected_features, f)


if __name__ == "__main__":
    main()
