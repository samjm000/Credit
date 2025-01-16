import pandas as pd
#Import dataset handlers
import encoder_treatment_categories, medical_or_surgical
import encoder_diagnostic_categories, encoder_ps, encoder_reason_for_admission, encoder_mech_ventilation
import export_to_excel
import pre_process_numerical
import warnings 

# Load the dataset & Filter Pandas warning
data = pd.read_excel(r"data\Credit_ML_dataset.xlsx", engine="openpyxl")
warnings.filterwarnings("ignore", category=FutureWarning)
# Strip leading/trailing spaces and newline characters from column names
data.columns = data.columns.str.strip().str.replace("\n", "")
# SEX column preprocessing
data = pre_process_numerical.preprocess_sex(data, "Sex")
# BMI column preprocessing  
data = pre_process_numerical.preprocess_mean(data, "BMI")
#ECOG PS at referral to Oncology imputation
data = encoder_ps.impute_ecog_ps(data)
#ECOG PS at referral at admission imputation
data = encoder_ps.impute_ps_on_admission(data)
#Diagnostic Categories one hot encoding
data = encoder_diagnostic_categories.encode_diagnostic_categories(data, "Diagnosis categories")
#Most Recent Oncological Treatment one hot encoding
data = encoder_treatment_categories.encode_treatment_categories(data, "Most recent oncological treatment")
#Treatment to Admission Time handling
data = pre_process_numerical.missing_binary(data, "Anticancer Therapy with 6 weeks")
#Admission Reason
data = encoder_reason_for_admission.encode_reason_for_admission(data, "Reason for admission to hospital")
#Surgical or Medical Cause of admission
data = medical_or_surgical.encode_and_impute(data, "Surgical or medical")
#Impute NEWS score prior to admission
data = pre_process_numerical.preprocess_mode(data, "Final NEWS 2 score Before Critical Care admission")
#Impute temperature columns with Mode
data = pre_process_numerical.preprocess_mode(data, ['Highest Temp in preceding 8 hours', 'Lowest Temp in preceding 8 hours'])
#Impute MAP with Mean
data = pre_process_numerical.preprocess_mean(data, ['MAP', 'Final HR before Critical Care admission'])
#YES / No  Imputations
data = pre_process_numerical.preprocess_yes_no(data, ['Cardiac arrest_1','Cardiac arrest_2','Direct admission from theatre?','Features of sepsis?', 'Haemodialysis /CRRT', 'AKI y/n', 'Acute renal failure_2','Survival 6 months post crit care'])
#Impute CC Missing observations with Mean
data = pre_process_numerical.preprocess_mean(data, ['First GCS on Critical Care admission', 'Final RR before Critical Care admission','Lowest temp', 'Highest HR', 'Lowest HR','Highest RR', 'Lowest RR', 'Lowest GCS'])   
#Impute Further CC Missing observations with Mean
data = pre_process_numerical.preprocess_mean(data, 'Urine output ml per day')   
#Mechanical Ventilation Imputation
data = encoder_mech_ventilation.one_hot_encode(data,'Mechanical ventilation (incl CPAP)')
#Impute CC Bloods with Mean
blood_data = ['Hb_1', 'Haematocrit_1', 'WBC_1', 'Neutrophils_1', 'Platelets_1', 'Na_1', 'K_1', 'Urea_1', 
              'Creatinine_umolperL_1', 'Bilirubin_1', 'Albumin_1','Albumin_2', 'Hb_2', 'Haematocrit_2', 'WBC_2', 'Platelets_2', 
              'Na_2', 'K_2', 'Urea_2', 'Creatinine_mgperdL_2', 'Creatinine_umolperL_2', 'Bilirubin_2', 'First pH on Admission to Critical Care', 
              'FiO2_1', 'PaCo2 kPa_1', 'PaO2 kPa_1', 'Aa gradient_1', 'PaO2 mmHg_1', 'PaO2_FiO2_1', 'BiCarb_1', 'Lactate_1', 'FiO2_2', 'pH_1', 
              'FiO2_3', 'PaCo2 kPa_2', 'PaO2 kPa_2', 'Aa gradient_2', 'PaO2 mmHg_2', 'PaO2_FiO2_2', 'Worst PaO2:FiO2 ratio', 'BiCarb_2', 'Lactate_2', 
              'FiO2_4']    

data = pre_process_numerical.preprocess_mean(data, blood_data)       
data = pre_process_numerical.replace_negatives_with_average(data, "Number of Days in Critical Care")
#EXPORT TO EXCEL
export_to_excel.export_to_excel(data, r"data\Credit_ML_dataset_cleaned.xlsx")
#FINIT