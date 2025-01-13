import pandas as pd

def impute_mode(df, columns):
    """
    This function imputes missing values in specified temperature columns with their respective mode values.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    columns (list): A list of column names to impute.

    Returns:
    pandas.DataFrame: The DataFrame with imputed missing values.
    """
    for column in columns:
        # Calculate the mode of the column
        mode_value = df[column].mode()[0]
        
        # Impute missing values with the mode
        df[column].fillna(mode_value, inplace=True)
    
    return df

# Example usage
if __name__ == "__main__":
    # Sample dataset
    data = {
        'Highest Temp in preceding 8 hours': [37.5, 38.2, None, 37.8, 38.1, None, 37.9],
        'Lowest Temp in preceding 8 hours': [36.4, None, 36.1, 36.5, None, 36.2, 36.3]
    }
    df = pd.DataFrame(data)
    
    # Columns to impute
    columns_to_impute = ['Highest Temp in preceding 8 hours', 'Lowest Temp in preceding 8 hours']
    
    # Apply the impute_temperature_modes function to the dataset
    df = impute_temperature_modes(df, columns_to_impute)
    
    print(df)
