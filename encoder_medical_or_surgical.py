import pandas as pd

def encode_and_impute(df, column_name):
    """
    This function handles a specified column by imputing missing values with the mode
    and then applying one-hot encoding to the column.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    column_name (str): The name of the column to handle.

    Returns:
    pandas.DataFrame: The DataFrame with the imputed and one-hot encoded column.
    """
    # Calculate the mode of the column
    mode_value = df[column_name].mode()[0]
    
    # Impute missing values with the mode
    df[column_name].fillna(mode_value, inplace=True)
    
    # Apply One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=[column_name])
    
    return df_encoded

# Example usage
if __name__ == "__main__":
    # Sample dataset
    data = {
        'Surgical or Medical': ['Surgical', 'Medical', None, 'Surgical', 'Medical', 'Surgical', None]
    }
    df = pd.DataFrame(data)
    
    # Apply the encode_and_impute function to the dataset
    df_encoded = encode_and_impute(df, 'Surgical or Medical')
    
    print(df_encoded.head())
