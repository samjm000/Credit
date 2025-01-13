import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("Credit.csv", encoding="ISO-8859-1")

# Strip leading/trailing spaces and newline characters from column names
data.columns = data.columns.str.strip().str.replace("\n", "")

# Verify columns
print(data.columns)

# Define the exact outcome column names after cleaning
outcome_column1 = "Survival 6 months post crit care"
outcome_column2 = "ECOG PS: 0=<2; 1=>3"
outcome_column3 = "Oncology treatment, 0=no, 1=yes"

# Ensure binary conversion for the outcome columns if they exist
if outcome_column1 in data.columns:
    data[outcome_column1] = data[outcome_column1].map({"Yes": 1, "No": 0})
    print(f"Column '{outcome_column1}' exists and is mapped to binary")

if outcome_column2 in data.columns:
    data[outcome_column2] = data[outcome_column2].map({"Yes": 1, "No": 0})
    print(f"Column '{outcome_column2}' exists and is mapped to binary")

# outcome 3 column is alreadt binary :D)

# Fill missing values for numeric columns
numeric_columns = data.select_dtypes(include="number").columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

# Fill missing values for non-numeric columns one by one
non_numeric_columns = data.select_dtypes(exclude="number").columns
for column in non_numeric_columns:
    if not data[column].mode().empty:
        mode_value = data[column].mode().iloc[0]
    else:
        mode_value = "Unknown"
    data[column] = data[column].fillna(mode_value)

# Encode categorical variables
data = pd.get_dummies(data)

# Define outcome variables
outcome1 = data[outcome_column1]
outcome2 = data[outcome_column2]
outcome3 = data[outcome_column3]

# Define predictor variables
predictors = data.drop(columns=[outcome_column1, outcome_column2, outcome_column3])

# Identify columns with missing values
missing_values = predictors.isnull().sum()
print(missing_values[missing_values > 0])

# Drop columns with all NaNs
predictors = predictors.dropna(axis=1, how="all")

# Impute missing values in predictor variables
imputer = SimpleImputer(strategy="median")
predictors = pd.DataFrame(imputer.fit_transform(predictors), columns=predictors.columns)


# Feature selection and modeling for each outcome
def feature_selection_and_modeling(X, y):
    model = LogisticRegression(max_iter=1000)
    rfe = RFE(model, n_features_to_select=10)
    X_rfe = rfe.fit_transform(X, y)
    model.fit(X_rfe, y)
    return model, rfe, X_rfe


# Train and evaluate for each outcome
for outcome, name in zip(
    [outcome1, outcome2, outcome3], [outcome_column1, outcome_column2, outcome_column3]
):
    print(f"Training for {name}...")
    X_train, X_test, y_train, y_test = train_test_split(
        predictors, outcome, test_size=0.2, random_state=42
    )
    model, rfe, X_rfe = feature_selection_and_modeling(X_train, y_train)
    y_pred = model.predict(rfe.transform(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {name}: {accuracy}")
    # Calculate ROC AUC if needed
    y_pred_proba = model.predict_proba(rfe.transform(X_test))[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC for {name}: {roc_auc}")
    # Feature importance
    print(f"Top features for {name}:")
    for feature in predictors.columns[rfe.support_]:
        print(feature)
