import pandas as pd
import os
from tqdm import tqdm
#-----------------------
# Load Diagnosis Table
def get_diagnosis(module_path: str) -> pd.DataFrame:
    """Reads in diagnosis_icd table"""

    return pd.read_csv(
        module_path + "/hosp/diagnoses_icd.csv.gz", compression="gzip", header=0
    )
#-----------------------
# Load Procedures Table  
def get_procedures(module_path: str) -> pd.DataFrame:
    """Reads in procedures table"""
        
    return pd.read_csv(
        module_path + "/hosp/procedures_icd.csv.gz", compression="gzip", header=0
    )
    
#-----------------------
# Load Procedures definition Table  
def get_procedures_definition(module_path: str) -> pd.DataFrame:
    """Reads in procedures table"""
        
    return pd.read_csv(
        module_path + "/hosp/d_icd_procedures.csv.gz", compression="gzip", header=0
    )
    
#-----------------------
# Load Diagnoses definition Table  
def get_diagnoses_definition(module_path: str) -> pd.DataFrame:
    """Reads in procedures table"""
        
    return pd.read_csv(
        module_path + "/hosp/d_icd_diagnoses.csv.gz", compression="gzip", header=0
    )
    
#-----------------------

# Extract patients for a specific ICD code in diag
     
def extract_diag_pts(module_path: str, ICD10_code: str ) -> tuple:

    diag = get_diagnosis(module_path)
    pt_ids = pd.DataFrame(diag.loc[diag.icd_code.str.startswith(ICD10_code, na=False)])

    return pt_ids
#-------------------------
# Extract patients for a specific ICD code in proc

def extract_procedure_pts(module_path: str, ICD10_code: str ) -> tuple:

    ICD10_code = "|".join(ICD10_code)
    procedure = get_procedures(module_path)

    pt_ids = pd.DataFrame(
            procedure.loc[procedure.icd_code.str.contains(ICD10_code)])
    return pt_ids


#-------------------------
# Extract admissions 

def extract_disease_cohort(mimic4_path:str, subject_col:str, visit_col:str, admit_col:str, disch_col:str, disease_label:str):
    

    visit = pd.read_csv(mimic4_path + "hosp/admissions.csv.gz", compression='gzip', header=0, index_col=None, parse_dates=[admit_col, disch_col])
    visit['los']=visit[disch_col]-visit[admit_col]
    print('# MIMIC-IV admissions :',visit['hadm_id'].nunique())
    print('# MIMIC-IV patients:',visit['subject_id'].nunique())
    print('---------------------------------------------------------------------------------------------')
    visit[admit_col] = pd.to_datetime(visit[admit_col])
    visit[disch_col] = pd.to_datetime(visit[disch_col])   
    visit['los']=pd.to_timedelta(visit[disch_col]-visit[admit_col],unit='h')
    visit['los']=visit['los'].astype(str)
    visit[['days', 'dummy','hours']] = visit['los'].str.split(' ', expand=True)
    visit['los']=pd.to_numeric(visit['days'])
    visit=visit.drop(columns=['days', 'dummy','hours'])
    

    # remove hospitalizations with a death; impossible for readmission for such visits
    '''visit = visit.loc[visit.hospital_expire_flag == 0]
    print('# MIMIC-IV admissions where patient is alive:',visit['hadm_id'].nunique())'''

    
    if len(disease_label):  # extracting disease patients
        print(f"patients for ICD code: --> '{disease_label}'")
        disease_pts = extract_diag_pts(mimic4_path,disease_label)
        visit=visit[visit['subject_id'].isin(disease_pts['subject_id'])]
        print(f'# patients: ' , visit['subject_id'].nunique())
        print(f'# all admissions for this group of patients: ' , visit['hadm_id'].nunique())

            
    else: 
        print('No disease specified')


    pts = pd.read_csv(
            mimic4_path + "hosp/patients.csv.gz", compression='gzip', header=0, index_col = None, usecols=[subject_col, 'anchor_year', 'anchor_age', 'anchor_year_group', 'dod','gender']
        )
    #print('number of all patients: ', pts.shape)
    pts['dod'] = pd.to_datetime(pts['dod'])   
    pts['yob']= pts['anchor_year'] - pts['anchor_age']  # get yob to ensure a given visit is from an adult
    pts['min_valid_year'] = pts['anchor_year'] + (2019 - pts['anchor_year_group'].str.slice(start=-4).astype(int))
    
    # Define anchor_year corresponding to the anchor_year_group 2017-2019. This is later used to prevent consideration
    # of visits with prediction windows outside the dataset's time range (2008-2019)
    #[[group_col, visit_col, admit_col, disch_col]]


    visit_pts = visit[[subject_col, visit_col, admit_col, disch_col,'los','hospital_expire_flag']].merge(
            pts[[subject_col, 'anchor_year', 'anchor_age', 'yob', 'min_valid_year', 'dod','gender']], how='inner', left_on=subject_col, right_on=subject_col
        )
    # only take adult patients
    # visit_pts['Age']=visit_pts[admit_col].dt.year - visit_pts['yob']
    # visit_pts = visit_pts.loc[visit_pts['Age'] >= 18]
    
    visit_pts['age']=visit_pts['anchor_age']
    visit_pts = visit_pts.loc[visit_pts['age'] >= 18]
    print('# all  patients older that 18: ', visit_pts['subject_id'].nunique())
    print('---------------------------------------------------------------------------------------------')
    
    ##Add Demo data
    eth = pd.read_csv(mimic4_path + "hosp/admissions.csv.gz", compression='gzip', header=0, usecols=['hadm_id', 'insurance','race'], index_col=None)
    visit_pts= visit_pts.merge(eth, how='inner', left_on='hadm_id', right_on='hadm_id')
    

    return visit_pts.dropna(subset=['min_valid_year'])[[subject_col, visit_col, admit_col, disch_col,'los', 'dod','hospital_expire_flag','age','gender','race', 'insurance']]


def extract_chemo_cohort(df:pd.DataFrame, mimic4_path:str):

    proc_code = pd.read_csv(os.path.join(mimic4_path, 'hosp/chemo_procedures.csv'),header=0, chunksize=None, delimiter=';') #input from Jakob
    
    icd_code = proc_code['icd_code'].tolist()
    icd_code=icd_code[:47] # Last row is for Z51 ommited
    
    chemo_ids= extract_procedure_pts(mimic4_path,icd_code) #all admissions for all patients for the icd_code
    df['chemo'] = df['hadm_id'].isin(chemo_ids['hadm_id']).astype(int)

    icd_code = 'Z5111'
    chemo_ids= extract_diag_pts(mimic4_path, icd_code) #all admissions for all patients for the icd_code
    df['chemo'] = df['chemo'] |df['hadm_id'].isin(chemo_ids['hadm_id']).astype(int)
    #df['chemo'] = df['hadm_id'].isin(chemo_ids['hadm_id']).astype(int)
  
    cancer_chemo_cohort_temp =df[df['chemo'] == 1]    
    #only chemo admissions
    print('#cc patients', cancer_chemo_cohort_temp['subject_id'].nunique())
    print('#cc chemo admissions',cancer_chemo_cohort_temp['hadm_id'].nunique())
    

    cancer_chemo_cohort  = df[df['subject_id'].isin(set(df[df['chemo'] == 1]['subject_id']))] 
    # all admissions of patients who had chemo at least once.
    print('#cc all admissions',cancer_chemo_cohort['hadm_id'].nunique())

    return cancer_chemo_cohort




