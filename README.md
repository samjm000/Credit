# Methodology for Validating the CREDIT Score with Machine Learning

## Overview
This repository contains the implementation of a machine learning methodology for validating the CREDIT Score. The CREDIT Score is designed to assist in risk stratification and decision-making for patients with incurable locally advanced or metastatic solid cancer admitted to ICU/critical care. This project leverages machine learning techniques to enhance the predictive capabilities of the CREDIT Score.

## Table of Contents
1. Overview
2. Data Preparation
3. Preprocessing
4. Machine Learning Implementation
5. Model Evaluation
6. Validation
7. Model Interpretation and Implementation
8. How to Use
9. Acknowledgments

## Overview
This project applies a Random Forest classifier to validate the CREDIT Score by modeling the relationships between patient characteristics and clinical outcomes. The primary outcomes assessed include:
- **Survival six months after ICU/critical care admission**
- **Return to anticancer therapy (ACT)**
- **ECOG PS greater than 2**

The methodology includes robust data preparation, preprocessing, model training, and validation to ensure accuracy and reliability.

## Data Preparation
### Dataset
The dataset comprises comprehensive patient data from the initial CREDIT Score study, including:
- **Demographic details**
- **Treatment history**
- **Physiological and biochemical characteristics**

The dataset focuses on patients admitted to ICU/critical care between January 2018 and December 2019. Missing data was handled using structured imputation techniques to maintain dataset integrity.

### Missing Data Handling
**Imputation Techniques:**
- **Categorical Data:** Mode imputation.
- **Numerical Data:** Mean imputation or Last Observation Carried Forward (LOCF).

## Preprocessing
### Adjustments and Encoding
- **Categorical Variables:** Converted using one-hot encoding to avoid bias.
- **Specific Handling:**
  - **BMI:** Missing values imputed using the mean.
  - **Histological Cancer Diagnosis:** Transformed into "Diagnosis categories" using one-hot encoding.
  - **Final NEWS 2 Score:** Missing values filled with the mode.
  - **ECOG PS:** Values on admission and referral imputed using the mode or set to 3 where appropriate.

### Fields Removed
Fields with excessive missing data, redundancy, or irrelevance were excluded. These include:
- APACHE, SAPS, and SOFA scores
- Admission and other date fields
- Specific treatment details with sparse data
- Fields with inconsistent or erroneous values

## Machine Learning Implementation
The **Random Forest classifier** was used to validate the CREDIT Score. Key steps include:
1. Splitting the dataset into **training** and **testing** sets.
2. Scaling numerical features using **StandardScaler**.
3. Performing **GridSearchCV** to optimize hyperparameters.
4. Training the model on the **training set**.
5. Evaluating performance using metrics such as **accuracy, ROC AUC, and feature importance**.
6. Implementing **parallel processing** where applicable to improve efficiency.

## Model Evaluation
### Performance Metrics
The model was evaluated using:
- **AUROC (Area Under the Receiver Operating Characteristic Curve)**
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

### ROC Curves
ROC curves were used to assess the model's discriminatory power.

### Comparison
The model’s performance was compared to traditional risk assessment scores like **APACHE, SOFA, and SAPS2**.

## Validation
### Internal Validation
- Conducted using the **testing cohort**.
- Ensured consistency in model performance.

### External Validation
- Recommended for future work using **independent datasets** from diverse clinical settings.

## Model Interpretation and Implementation
### Interpretation
- **Feature importance analysis** identified the most significant predictors.
- **SHAP (SHapley Additive exPlanations)** was used for explainability.
- Clinical relevance was assessed to validate the utility of the CREDIT Score.

### Implementation
- Developed **guidelines** for clinical application.
- Recommended **training resources** for clinicians.

## How to Use
1. **Clone the repository:**
   ```bash
   git clone https://github.com/samjm000/Credit.git
   ```
2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the preprocessing script to prepare the dataset:**
   ```bash
   python preprocess.py
   ```
4. **Train and evaluate the model:**
   ```bash
   python train_model.py
   ```

## Acknowledgments
This work builds on the **CREDIT Score study** and leverages patient data with appropriate ethical approvals. The methodology was designed to support clinical decision-making for advanced cancer patients.

