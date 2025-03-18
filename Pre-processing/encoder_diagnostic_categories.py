import pandas as pd

def encode_diagnostic_categories(df, column_name):
    """
    This function handles diagnostic categories, replacing specified combined categories with 'Multiple'
    and applies one-hot encoding to the specified column.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    column_name (str): The name of the column containing diagnostic categories.

    Returns:
    pandas.DataFrame: The DataFrame with one-hot encoded diagnostic categories.
    """
    # List of combined categories to replace
    combined_categories = ['Breast and Lower GI', 'Melanoma and Urology', 'Lung and breast']
    
    # Replace combined categories with 'Multiple'
    df[column_name] = df[column_name].replace(combined_categories, 'Multiple')
    
    # Apply One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=[column_name])
    
    return df_encoded

# Example usage
if __name__ == "__main__":
    # List of diagnosis categories
    diagnosis_categories = [
        'Urology', 'Lower GI', 'H&N', 'Breast', 'Lung', 'Brain', 'CUP', 
        'Upper GI', 'Gynae', 'Sarcoma', 'Endocrine', 'Melanoma', 
        'Breast', 'Germ Cell', 'Lung', 'Squamous cell carcinoma',
        'Breast and Lower GI', 'Melanoma and Urology', 'Lung and breast'
    ]

    # Create a DataFrame
    df = pd.DataFrame(diagnosis_categories, columns=['Diagnosis Category'])
    
    # Apply the encode_diagnostic_categories function to the dataset
    df_encoded = encode_diagnostic_categories(df, 'Diagnosis Category')
    
    print(df_encoded.head())
