import pandas as pd

# Define function to handle and encode treatment categories
def encode_treatment_categories(df, column_name):
    """
    This function handles treatment categories, replacing '0' or blank values with 'Unknown'
    and applies one-hot encoding to the specified column.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    column_name (str): The name of the column containing treatment categories.

    Returns:
    pandas.DataFrame: The DataFrame with one-hot encoded treatment categories.
    """
    # Handle missing values by replacing '0' and None with 'Unknown'
    df[column_name].replace(['0', None], 'Unknown', inplace=True)
    
    # Apply One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=[column_name])
    
    return df_encoded

# Example usage
if __name__ == "__main__":
    # List of most recent oncological treatments
    treatments = [
        'Oral targeted therapy', 'Non-oral cytotoxic chemotherapy', 'Non-oral targeted therapy',
        'Immunotherapy', 'Radioisotopes', 'Oral cytotoxic chemotherapy', 'Radiotherapy',
        'Other', 'Immunotherapy', 'Radiotherapy', 'Chemoradiotherapy', 'Surgery', None
    ]

    # Create a DataFrame
    df = pd.DataFrame(treatments, columns=['Most recent oncological treatment'])
    
    # Apply the encode_treatment_categories function to the dataset
    df_encoded = encode_treatment_categories(df, 'Most recent oncological treatment')
    
    print(df_encoded.head(15))  # Adjust to show more rows if needed
