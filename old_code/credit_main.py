import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("Credit.csv", encoding="ISO-8859-1")

# Strip leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Verify the columns and their exact names with indices
for idx, col in enumerate(data.columns):
    print(f"Index {idx}: Column {col}")

# Ensure exact column names for outcome variables
outcome_column1 = "Survival 6 months post crit care"
outcome_column2 = "ECOG PS: 0=<2; 1=>3"

# Check if the outcome columns exist
if outcome_column1 in data.columns:
    data[outcome_column1] = data[outcome_column1].map({"Yes": 1, "No": 0})
    outcome1 = data[outcome_column1]
    print(f"Column '{outcome_column1}' exists and is assigned to outcome1")

if outcome_column2 in data.columns:
    data[outcome_column2] = data[outcome_column2].map({"Yes": 1, "No": 0})
    outcome2 = data[outcome_column2]
    print(f"Column '{outcome_column2}' exists and is assigned to outcome2")

# Fill missing values for numeric columns
numeric_columns = data.select_dtypes(include="number").columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

# Fill missing values for non-numeric columns one by one
non_numeric_columns = data.select_dtypes(exclude="number").columns
for column in non_numeric_columns:
    if not data[column].mode().empty:
        mode_value = (
            data[column].mode().iloc[0]
        )  # use .iloc to safely access mode value
    else:
        mode_value = "Unknown"
    data[column] = data[column].fillna(mode_value)  # Avoid chaining method

# Encode categorical variables
data = pd.get_dummies(data)

# Define predictors (excluding outcome columns)
predictors = data.drop(
    columns=[col for col in [outcome_column1, outcome_column2] if col in data.columns]
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    predictors, outcome1, test_size=0.2, random_state=42
)

# Initialize the model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Predict probabilities on the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {roc_auc}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc})")
plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()
