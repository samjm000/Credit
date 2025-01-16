from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


#Test
#export_to_excel.export_to_excel(data, r"data\Credit_ML_dataset_cleaned.xlsx")

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
