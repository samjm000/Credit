import pandas as pd

def treatment_and_admission_timeframe_imputation(df, column_name):
    """
    This function handles the 'Time between last treatment and admission' column by 
    replacing missing values and negative numbers with 'Unknown'.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    column_name (str): The name of the column to handle.

    Returns:
    pandas.DataFrame: The DataFrame with the handled column.
    """
    # Replace missing values and negative numbers with 'Unknown'
    df[column_name] = df[column_name].apply(lambda x: 'Unknown' if pd.isnull(x) or x < 0 else x)
    
    return df

# Example usage
if __name__ == "__main__":
    # Sample dataset
    data = {
        'ECOG PS at referral to Oncology': [2, 3, 1, None, 2, 1, None, 3, 2],
        'PS on admission': [1, None, 2, 3, None, 1, 4, None, 0],
        'Time between last treatment and admission': [4, 14, None, -2, 7, None, 10, -1, 3]
    }
    df = pd.DataFrame(data)
    
    # Apply the treatment_and_admission_timeframe_imputation function to the dataset
    df = treatment_and_admission_timeframe_imputation(df, 'Time between last treatment and admission')
    
    print(df)
