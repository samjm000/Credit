import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz
import shap



# Load the cleaned dataset
data = pd.read_excel(r"data\Credit_ML_dataset_cleaned.xlsx", engine="openpyxl")

# Define the exact outcome column names
outcome_column1 = "Survival 6 months post crit care"
outcome_column2 = "ECOG PS: 1=<2; 0=>3"
outcome_column3 = "Oncology treatment, 0=no, 1=yes"

# Separate the outcomes
outcome1 = data[outcome_column1]
outcome2 = data[outcome_column2]
outcome3 = data[outcome_column3]

# Define predictor variables (exclude the outcome columns)
predictors = data.drop(columns=[outcome_column1, outcome_column2, outcome_column3])

# Handle non-numeric columns separately
numeric_predictors = predictors.select_dtypes(include="number")
non_numeric_predictors = predictors.select_dtypes(exclude="number")

# Check if numeric_predictors is empty
if numeric_predictors.empty:
    raise ValueError("The numeric_predictors DataFrame is empty. Please check your data loading and preprocessing steps.")

# Combine back the numeric and non-numeric predictors
predictors = pd.concat(
    [
        numeric_predictors.reset_index(drop=True),
        non_numeric_predictors.reset_index(drop=True),
    ],
    axis=1,
)

# Scale the numeric columns only
scaler = StandardScaler()
numeric_predictors_scaled = pd.DataFrame(
    scaler.fit_transform(numeric_predictors), columns=numeric_predictors.columns
)

# Re-assemble predictors with scaled numeric values and non-numeric values
predictors_scaled = pd.concat(
    [
        numeric_predictors_scaled.reset_index(drop=True),
        non_numeric_predictors.reset_index(drop=True),
    ],
    axis=1,
)

# Align predictors with each outcome index
predictors_scaled1 = predictors_scaled.loc[outcome1.index].reset_index(drop=True)
predictors_scaled2 = predictors_scaled.loc[outcome2.index].reset_index(drop=True)
predictors_scaled3 = predictors_scaled.loc[outcome3.index].reset_index(drop=True)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Function to perform grid search and return the best parameters
def perform_grid_search(X, y):
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    return grid_search.best_params_, grid_search.best_score_

# Perform grid search for each outcome
best_params_outcome1, best_score_outcome1 = perform_grid_search(predictors_scaled1, outcome1)
best_params_outcome2, best_score_outcome2 = perform_grid_search(predictors_scaled2, outcome2)
best_params_outcome3, best_score_outcome3 = perform_grid_search(predictors_scaled3, outcome3)

# Print the best parameters and scores
print("Best parameters for outcome1:", best_params_outcome1)
print("Best accuracy score for outcome1:", best_score_outcome1)
print("Best parameters for outcome2:", best_params_outcome2)
print("Best accuracy score for outcome2:", best_score_outcome2)
print("Best parameters for outcome3:", best_params_outcome3)
print("Best accuracy score for outcome3:", best_score_outcome3)


# Print the shapes of predictors and outcomes
print(f"Predictors shape after cleaning: {predictors_scaled.shape}")
print(
    f"Outcome1 shape: {outcome1.shape}, Outcome2 shape: {outcome2.shape}, Outcome3 shape: {outcome3.shape}"
)

def train_and_evaluate_rf(X, y, outcome_name, best_params):
    print(f"Training model for {outcome_name} with best parameters...")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize the Random Forest model with the best parameters
    model = RandomForestClassifier(**best_params, random_state=42)

    # Train the model
    rf_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # SHAP EXPLAINABILITY 
    explainer = shap.TreeExplainer(model)  # Use TreeExplainer for RandomForest
    shap_values = explainer.shap_values(X_train)  # Compute SHAP values

    # SHAP Summary Plot - Shows top predictors
    shap.summary_plot(shap_values[1], X_train)  # [1] for class 1 in classification

    # SHAP Waterfall Plot - Explains a single prediction
    sample_index = 0  # Choose an index from X_test
    shap.waterfall_plot(shap.Explanation(values=shap_values[1][sample_index], 
                                         base_values=explainer.expected_value[1], 
                                         data=X_train.iloc[sample_index]))

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Accuracy for {outcome_name}: {accuracy:.4f}")
    print(f"ROC AUC for {outcome_name}: {roc_auc:.4f}")

    # Print Classification Report and Confusion Matrix
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance from the model
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = feature_importances.nlargest(10)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    top_features.plot(kind="barh", color="skyblue", edgecolor="black")
    plt.title(f"Top Features for {outcome_name}", fontsize=16)
    plt.xlabel("Feature Importance", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    # # Feature importance
    # feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    # feature_importances.nlargest(10).plot(kind="barh")
    # plt.title(f"Top Features for {outcome_name}")
    # plt.show()

        # Feature importance plot with enhanced aesthetics
    feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    top_features = feature_importances.nlargest(10)
    
    plt.figure(figsize=(10, 6))
    top_features.sort_values().plot(kind="barh", color="skyblue", edgecolor="black")
    plt.title(f"Top Features for {outcome_name}", fontsize=16)
    plt.xlabel("Importance", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()  # Adjust layout to avoid cropping
    plt.show()


    return rf_model, X_train, y_train  # Return model and training data for further use


# Train and evaluate with the best parameters for each outcome
model1, X_train1, y_train1 = train_and_evaluate_rf(predictors_scaled1, outcome1, outcome_column1, best_params_outcome1)
model2, X_train2, y_train2 = train_and_evaluate_rf(predictors_scaled2, outcome2, outcome_column2, best_params_outcome2)
model3, X_train3, y_train3 = train_and_evaluate_rf(predictors_scaled3, outcome3, outcome_column3, best_params_outcome3)


def visualize_single_tree(model, X, outcome_name, tree_index=0):
    """Extracts and visualizes a single decision tree from the Random Forest model."""
    tree = model.estimators_[tree_index]  # Get one tree from the forest

    dot_data = export_graphviz(
        tree, 
        out_file=None, 
        feature_names=X.columns, 
        class_names=["No", "Yes"],  # Modify based on your class labels
        filled=True, rounded=True, special_characters=True
    )

    graph = graphviz.Source(dot_data)
    graph.render(f"decision_tree_{outcome_name}", format="png", cleanup=True)  # Saves as PNG
    return graph






# Grid Search for Hyperparameter Tuning

# Define parameter grid for Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Function to perform Grid Search
def grid_search_tuning(X_train, y_train, outcome_name):
    print(f"Grid search for {outcome_name}...")

    # Initialize Grid Search
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='roc_auc')

    # Fit Grid Search to training data
    grid_search.fit(X_train, y_train)

    # Best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best Parameters for {outcome_name}: {best_params}")
    print(f"Best ROC AUC Score for {outcome_name}: {best_score:.4f}")

# Perform Grid Search for each outcome
grid_search_tuning(X_train1, y_train1, outcome_column1)
grid_search_tuning(X_train2, y_train2, outcome_column2)
grid_search_tuning(X_train3, y_train3, outcome_column3)
