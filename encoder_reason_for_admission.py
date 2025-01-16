import pandas as pd

def encode_reason_for_admission(df, column_name):
    """
    This function applies one-hot encoding to the 'Reason for admission to hospital' column.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    column_name (str): The name of the column containing reasons for admission.

    Returns:
    pandas.DataFrame: The DataFrame with one-hot encoded reasons for admission.
    """
    # Apply One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=[column_name])
    
    return df_encoded

# Example usage
if __name__ == "__main__":
    # Sample dataset
    data = {
        'Reason for admission to hospital': ['Disease related', 'Other', 'Treatment related', 'Disease related', 'Other']
    }
    df = pd.DataFrame(data)
    
    # Apply the encode_reason_for_admission function to the dataset
    df_encoded = encode_reason_for_admission(df, 'Reason for admission to hospital')
    
    print(df_encoded.head())
