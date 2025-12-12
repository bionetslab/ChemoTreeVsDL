import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta

def get_lab_items_labels(module_path: str) -> pd.DataFrame:
    """Reads in d_labitems table"""

    return pd.read_csv(
        module_path + "/hosp/d_labitems.csv.gz", compression="gzip", header=0
    )
    
def extract_cohort_labs(mimic4_path: str, cohort: pd.DataFrame, time_col:str, dtypes: dict, usecols: list) -> pd.DataFrame:
    """Function for getting hosp observations pertaining to a pickled cohort. Function is structured to save memory when reading and transforming data."""
    
    usecols = ['itemid','subject_id','hadm_id','charttime','valuenum','valueuom']
    dtypes = {
        'itemid':'int64',
        'subject_id':'int64',
        # 'hadm_id':'int64',            # hadm_id type not defined because it contains NaN values
        # 'charttime':'datetime64[ns]', # used as an argument in 'parse_cols' in pd.read_csv
        'value':'object',
        'valuenum':'float64',
        'valueuom':'object',
        'flag':'object'
    }


    lab_df_cohort=pd.DataFrame()
    cohort['dischtime']=pd.to_datetime(cohort['dischtime'])


    chunksize = 10000000
    for chunk in tqdm(pd.read_csv(mimic4_path + "hosp/labevents.csv.gz", compression='gzip', usecols=usecols, dtype=dtypes, parse_dates=[time_col],chunksize=chunksize)):

        chunk=chunk.dropna(subset=['valuenum'])
        chunk['valueuom']=chunk['valueuom'].fillna(0)
        chunk=chunk[chunk['subject_id'].isin(cohort['subject_id'].unique())]
        chunk['charttime']=pd.to_datetime(chunk['charttime'])
        chunk['hadm_id']=chunk['hadm_id'].fillna(0)
        chunk=chunk.dropna()

        if lab_df_cohort.empty:
            lab_df_cohort=chunk
        else:
            lab_df_cohort = pd.concat([lab_df_cohort, chunk], ignore_index=True) #return all the lab results for the subjects
        
    print("# Itemid: ", lab_df_cohort.itemid.nunique())

    return lab_df_cohort   



def drop_wrong_uom(data, cut_off):
#     count=0
    grouped = data.groupby(['itemid'])['valueuom']
    for id_number, uom in grouped:
        value_counts = uom.value_counts()
        num_observations = len(uom)
        if(value_counts.size >1):
#             count+=1
            most_frequent_measurement = value_counts.index[0]
            frequency = value_counts[0]
#             print(id_number,value_counts.size,frequency/num_observations)
            if(frequency/num_observations > cut_off):
                values = uom
                index_to_drop = values[values != most_frequent_measurement].index
                data.drop(index_to_drop, axis=0, inplace=True)
    data = data.reset_index(drop=True)
#     print(count)
    return data


def find_itemid (mimic4_path,label)-> list:
    lab_item_labels =  get_lab_items_labels(mimic4_path)
    lab_item_labels["label"] = lab_item_labels["label"].str.lower()
    ids = lab_item_labels[lab_item_labels["label"] == label]['itemid']
    return ids.tolist()

def extract_admission_labs(x, days,lab):
    admit_lab = lab[
            (lab["subject_id"] == x.subject_id) & 
            (lab["charttime"] <= (x.dischtime))&
            (lab["charttime"] >= (x.dischtime - timedelta(days=days)))
        ].sort_values("charttime")  
    
    admit_lab["hadm_id"] = x.hadm_id     #assign admission's hadm_id   to the lab results 
    admit_lab["admittime"] = x.admittime #assign admission's admittime to lab results 
    admit_lab["dischtime"] = x.dischtime #assign admission's dischtime to lab results    
    admit_lab['lab_time_from_disch'] = admit_lab['charttime'] - admit_lab['dischtime'] 

    return admit_lab