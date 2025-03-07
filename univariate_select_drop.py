import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import StandardScaler

# Load the cleaned dataset
data = pd.read_excel(r"data\Credit_ML_dataset_cleaned.xlsx", engine="openpyxl")

# Define predictors and target
X = data.drop(columns=['Survival 6 months post crit care', 'ECOG PS: 0=<2; 1=>3', 'Oncology treatment, 0=no, 1=yes'])
y = data['Survival 6 months post crit care']  # or choose another outcome variable

# Drop rows with missing values
X_dropped = X.dropna()

# Ensure corresponding y is also dropped
y_dropped = y[X_dropped.index]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dropped)

# Apply Univariate Feature Selection (ANOVA F-value for classification)
selector = SelectKBest(score_func=f_classif, k='all')  # Select all features to view their scores
X_new = selector.fit_transform(X_scaled, y_dropped)

# Get the scores for each feature
scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': selector.scores_
})

# Sort by the highest score
scores = scores.sort_values(by='Score', ascending=False)

# Display the top features
print(scores.head(10))  # Display top 10 features

# Select top k features
top_k_selector = SelectKBest(score_func=f_classif, k=5)
X_top_k = top_k_selector.fit_transform(X_scaled, y_dropped)

# Get the selected feature names
selected_features = X.columns[top_k_selector.get_support()]
print(f"Selected features: {selected_features}")
