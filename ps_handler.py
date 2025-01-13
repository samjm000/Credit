import pandas as pd

def impute_ecog_ps(df):
    """
    This function imputes missing values in the 'ECOG PS at referral to Oncology' column
    with the mode (most frequent value) of the column.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    pandas.DataFrame: The DataFrame with imputed missing values.
    """
    # Calculate the mode of the column
    mode_value = df['ECOG PS at referral to Oncology'].mode()[0]
    
    # Impute missing values with the mode
    df['ECOG PS at referral to Oncology'].fillna(mode_value, inplace=True)
    
    return df

def impute_ps_on_admission(df):
    """
    This function sets missing values in the 'ECOG PS on admission to hosptial' column to a PS of 3.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    pandas.DataFrame: The DataFrame with imputed PS on admission values.
    """
    # Set missing values to 3
    df['ECOG PS on admission to hosptial'].fillna(3, inplace=True)
    
    return df

# Example usage
if __name__ == "__main__":
    # Sample dataset
    data = {
        'ECOG PS at referral to Oncology': [2, 3, 1, None, 2, 1, None, 3, 2],
        'PS on admission': [1, None, 2, 3, None, 1, 4, None, 0]
    }
    df = pd.DataFrame(data)
    
    # Apply the impute_ecog_ps function to the dataset
    df = impute_ecog_ps(df)
    
    # Apply the impute_ps_on_admission function to the dataset
    df = impute_ps_on_admission(df)
    
    print(df)
