import pandas as pd
#Import dataset handlers
import treatment_categories, medical_or_surgical
import diagnostic_categories, ps_handler, reason_for_admission, mech_ventilation
import export_to_excel
import preprocess_numerical
import warnings 

# Load the dataset & Filter Pandas warning
data = pd.read_excel(r"data\Credit_ML_dataset.xlsx", engine="openpyxl")
warnings.filterwarnings("ignore", category=FutureWarning)
# Strip leading/trailing spaces and newline characters from column names
data.columns = data.columns.str.strip().str.replace("\n", "")
# SEX column preprocessing
data = preprocess_numerical.preprocess_sex(data, "Sex")
# BMI column preprocessing  
data = preprocess_numerical.preprocess_mean(data, "BMI")
#ECOG PS at referral to Oncology imputation
data = ps_handler.impute_ecog_ps(data)
#ECOG PS at referral at admission imputation
data = ps_handler.impute_ps_on_admission(data)
#Diagnostic Categories one hot encoding
data = diagnostic_categories.encode_diagnostic_categories(data, "Diagnosis categories")
#Most Recent Oncological Treatment one hot encoding
data = treatment_categories.encode_treatment_categories(data, "Most recent oncological treatment")
#Treatment to Admission Time handling
data = preprocess_numerical.missing_binary(data, "Anticancer Therapy with 6 weeks")
#Admission Reason
data = reason_for_admission.encode_reason_for_admission(data, "Reason for admission to hospital")
#Surgical or Medical Cause of admission
data = medical_or_surgical.encode_and_impute(data, "Surgical or medical")
#Impute NEWS score prior to admission
data = preprocess_numerical.preprocess_mode(data, "Final NEWS 2 score Before Critical Care admission")
#Impute temperature columns with Mode
data = preprocess_numerical.preprocess_mode(data, ['Highest Temp in preceding 8 hours', 'Lowest Temp in preceding 8 hours'])
#Impute MAP with Mean
data = preprocess_numerical.preprocess_mean(data, ['MAP', 'Final HR before Critical Care admission'])
#YES / No  Imputations
data = preprocess_numerical.preprocess_yes_no(data, ['Cardiac arrest_1','Cardiac arrest_2','Direct admission from theatre?','Features of sepsis?', 'Haemodialysis /CRRT', 'AKI y/n', 'Acute renal failure_2','Survival 6 months post crit care'])
#Impute CC Missing observations with Mean
data = preprocess_numerical.preprocess_mean(data, ['First GCS on Critical Care admission', 'Final RR before Critical Care admission','Lowest temp', 'Highest HR', 'Lowest HR','Highest RR', 'Lowest RR', 'Lowest GCS'])   
#Impute Further CC Missing observations with Mean
data = preprocess_numerical.preprocess_mean(data, 'Urine output ml per day')   
#Mechanical Ventilation Imputation
data = mech_ventilation.one_hot_encode(data,'Mechanical ventilation (incl CPAP)')
#Impute CC Bloods with Mean
blood_data = ['Hb_1', 'Haematocrit_1', 'WBC_1', 'Neutrophils_1', 'Platelets_1', 'Na_1', 'K_1', 'Urea_1', 
              'Creatinine_umolperL_1', 'Bilirubin_1', 'Albumin_1', 'Hb_2', 'Haematocrit_2', 'WBC_2', 'Platelets_2', 
              'Na_2', 'K_2', 'Urea_2', 'Creatinine_mgperdL_2', 'Creatinine_umolperL_2', 'Bilirubin_2', 'First pH on Admission to Critical Care', 
              'FiO2_1', 'PaCo2 kPa_1', 'PaO2 kPa_1', 'Aa gradient_1', 'PaO2 mmHg_1', 'PaO2_FiO2_1', 'BiCarb_1', 'Lactate_1', 'FiO2_2', 'pH_1', 
              'FiO2_3', 'PaCo2 kPa_2', 'PaO2 kPa_2', 'Aa gradient_2', 'PaO2 mmHg_2', 'PaO2_FiO2_2', 'Worst PaO2:FiO2 ratio', 'BiCarb_2', 'Lactate_2', 
              'FiO2_4']    

data = preprocess_numerical.preprocess_mean(data, blood_data)       
#EXPORT TO EXCEL
export_to_excel.export_to_excel(data, r"data\Credit_ML_dataset_cleaned.xlsx")
#FINIT