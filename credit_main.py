import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

#Import dataset handlers
import treatment_categories, medical_or_surgical
import diagnostic_categories, ps_handler, reason_for_admission
import export_to_excel
import preprocess_numerical


# DATASET CLEANING

# Load the dataset 
data = pd.read_excel(r"data\Credit_ML_dataset.xlsx", engine="openpyxl")
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
data = preprocess_numerical.preprocess_mean(data, "Time between last treatment and admission")
#Admission Reason
data = reason_for_admission.encode_reason_for_admission(data, "Reason for admission to hospital")
#Surgical or Medical Cause of admission
data = medical_or_surgical.encode_and_impute(data, "Surgical or medical")
#Impute NEWS score prior to admission
data = preprocess_numerical.preprocess_mode(data, "Final NEWS 2 score Before Critical Care admission")
#Impute temperature columns with Mode
data = preprocess_numerical.preprocess_mode(data, ['Highest Temp in preceding 8 hours', 'Lowest Temp in preceding 8 hours'])
#YES / NO imputations
data = preprocess_numerical.preprocess_yes_no(data, ['Cardiac arrest_1','Cardiac arrest_2','Direct admission from theatre?','Features of sepsis? ', 'Haemodialysis /CRRT', 'AKI y/n', 'Acute renal failure_2','Survival 6 months post crit care'])


#Test
export_to_excel.export_to_excel(data, 'data\Credit_ML_dataset_cleaned.xlsx')

#Impute heart rate, MAP, resp rate and cardiac arrest columns with Mode
# data = preprocess_numerical.preprocess_mode(data, ['MAP', 'Final HR before Critical Care admission','Cardiac arrest_1','Final RR before Critical Care admission', 
# 'Final HR before Critical Care admission', 'Lowest Temp in preceding 8 hours'])
# #imputation Theatre admission , Sepsis and cardiac arrest, Mechanical Ventilation, Haemodialisys 
# data = preprocess_numerical.preprocess_mode(data, ['Cardiac arrest_1','Cardiac arrest_2','Direct admission from theatre?', 'Features of sepsis?', 'Haemodialysis /CRRT', 'Mechanical ventilation (incl CPAP)'])
# #List of fields that are likely to have missing data without significant clinical importance
# mode_preprocess_list = [
#     'First GCS on Critical Care admission', 'Lowest temp', 'Highest HR', 'Lowest HR', 'Cardiac arrest_2', 
#     'Highest RR', 'Lowest RR', 'Lowest GCS', 'Urine output ml per day', 'Pressors y/n', 
#     'Mechanical ventilation (incl CPAP)', 'resp support y/n', 'Haemodialysis /CRRT', 'Hb_1', 
#     'Haematocrit_1', 'WBC_1', 'Neutrophils_1', 'Platelets_1', 'Na_1', 'K_1', 'Urea_1', 
#     'Creatinine_umolperL_1', 'AKI y/n', 'Bilirubin_1', 'Albumin_1', 'Hb_2', 'Haematocrit_2', 
#     'WBC_2', 'Platelets_2', 'Na_2', 'K_2', 'Urea_2', 'Creatinine_mgperdL_2', 'Creatinine_umolperL_2', 
#     'Acute renal failure_2', 'Albumin_2', 'Bilirubin_2', 'First pH on Admission to Critical Care', 
#     'FiO2_1', 'PaCo2 kPa_1', 'PaO2 kPa_1', 'Aa gradient_1', 'PaO2 mmHg_1', 'PaO2_FiO2_1', 
#     'BiCarb_1', 'Lactate_1', 'FiO2_2', 'pH_1', 'FiO2_3', 'PaCo2 kPa_2', 'PaO2 kPa_2', 
#     'Aa gradient_2', 'PaO2 mmHg_2', 'PaO2_FiO2_2', 'Worst PaO2:FiO2 ratio', 'BiCarb_2', 
#     'Lactate_2', 'FiO2_4', 'Number of Days in Critical Care'
# ]
# #bloods missing values imputation
# data = preprocess_numerical.preprocess_mode(data,mode_preprocess_list)
# 

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Initialize the Random Forest model
# rf_model = RandomForestClassifier(random_state=42)

# # Train the model
# rf_model.fit(X_train, y_train)

# # Make predictions
# y_pred = rf_model.predict(X_test)
# y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# # Calculate ROC AUC Score
# roc_auc = roc_auc_score(y_test, y_pred_proba)
# print(f"ROC AUC Score: {roc_auc:.4f}")

# # Print Classification Report
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# # Print Confusion Matrix
# print("Confusion Matrix:")
# conf_matrix = confusion_matrix(y_test, y_pred)
# print(conf_matrix)

# # Define parameter grid
# param_grid = {'n_estimators': [100, 200, 300],'max_depth': [10, 20, 30],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4]}

# # # Initialize Grid Search
# grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='roc_auc')

# # # Fit Grid Search to training data
# grid_search.fit(X_train, y_train)

# # Best parameters and score
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_

# print(f"Best Parameters: {best_params}")
# print(f"Best ROC AUC Score: {best_score:.4f}")
