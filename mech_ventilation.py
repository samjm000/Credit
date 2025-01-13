import pandas as pd

# List of unique mechanical ventilation categories
ventilation = [
    'Invasive', 'NIV', 'NO'
]

# Create a DataFrame
df = pd.DataFrame(ventilation, columns=['Mechanical ventilation (incl CPAP)'])

# Normalize case to handle inconsistencies
df['Mechanical ventilation (incl CPAP)'] = df['Mechanical ventilation (incl CPAP)'].str.lower()

# Handle missing values and normalize 'no' entries
df['Mechanical ventilation (incl CPAP)'].replace({'no': 'no'}, inplace=True)

# Apply One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['Mechanical ventilation (incl CPAP)'])

print(df_encoded.head())
