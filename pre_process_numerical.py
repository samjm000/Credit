import pandas as pd

def preprocess_sex(df, columns):
    """
    Converts 'M' to 1 and 'F' to 0, and fills missing values with 0 in the specified columns of the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    columns (str or list): A column name or list of column names to preprocess.

    Returns:
    pandas.DataFrame: The modified DataFrame with converted and filled values in the specified columns.
    """
    if isinstance(columns, str):
        columns = [columns]
    
    for column in columns:
        if column in df.columns:
            if not df[column].empty:
                # Convert 'M' to 1 and 'F' to 0
                df[column] = df[column].map({'M': 1, 'F': 0})
                # Fill missing values with 0# Fill missing values with 0 
                df[column] = df[column].fillna(0)
                
    return df


def preprocess_yes_no(df, columns):
    """
    Converts 'yes' to True and 'no' to False, and fills missing values with False in the specified columns of the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    columns (str or list): A column name or list of column names to preprocess.

    Returns:
    pandas.DataFrame: The modified DataFrame with converted and filled values in the specified columns.
    """
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if column in df.columns and not df[column].empty:
            # Ensure the column is treated as string before applying string methods
            df[column] = df[column].astype(str).str.strip().str.lower()
            
            # Map 'yes' to True, 'no' to False, leave other values untouched
            df[column] = df[column].map({'yes': True, 'no': False}, na_action='ignore')
            
            # Fill missing values with False
            df[column] = df[column].fillna(False)
    return df



def missing_binary(df, columns):
    """
    If within 1 or 0 is missing, fills missing values with 0 in the specified columns of the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    columns (str or list): A column name or list of column names to preprocess.

    Returns:
    pandas.DataFrame: The modified DataFrame with converted and filled values in the specified columns.
    """
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if column in df.columns and not df[column].empty:
            if not df[column].empty:    # Fill missing values with 0                
                df[column] = df[column].fillna(0)
    return df

def preprocess_mode(df, columns):
    """
    Fills missing values with the mode in the specified columns of the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    columns (list): A list of column names to preprocess.

    Returns:
    pandas.DataFrame: The modified DataFrame with missing values filled.
    """
    if isinstance(columns, str):
        columns = [columns]
    
    for column in columns:
        if column in df.columns and not df[column].empty:
            # Fill missing values with the mode (whilst checking there is a mode)
            mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else None
            if mode_value is not None:
                df[column] = df[column].fillna(mode_value)
                
                
    return df
import pandas as pd

def preprocess_mean(df, columns):
    """
    Fills missing values with the mean in the specified numerical columns of the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    columns (str or list): A column name or list of column names to preprocess.

    Returns:
    pandas.DataFrame: The modified DataFrame with missing values filled with the mean.
    """
    if isinstance(columns, str):
        columns = [columns]
    
    for column in columns:
        if column in df.columns:
            # Ensure all values are numeric, convert non-numeric to NaN 
            df[column] = pd.to_numeric(df[column], errors='coerce')
            # Calculate the mean of the column, ignoring NaNs
            mean_value = df[column].mean()
            
            # Fill missing values with the mean
            df[column] = round(df[column].fillna(mean_value))
    
    return df


def replace_negatives_with_average(df, column):
    """
    Replaces negative values in the specified column with the average of the non-negative values.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    column (str): The name of the column to preprocess.

    Returns:
    pandas.DataFrame: The modified DataFrame with negative values replaced by the average of non-negative values.
    """
    if column in df.columns:
        # Calculate the average of non-negative values
        non_negative_mean = df[df[column] >= 0][column].mean()

        # Replace negative values with the calculated average
        df[column] = round(df[column].apply(lambda x: non_negative_mean if x < 0 else x))
    
    return df

if __name__ == "__main__":
    # Example usage
    print("test binary")

    test_cases = {
        "patient_id" : [123,456,789,198,876,654,432,345,745,999],
        "binary_sequence" : [1,0,0,1,1,1,None,1,None,0],
        "Sex" : ["M","F","F","F","M","M","M","M","F","F"],
        "Urine output ml per day" : [832, 3686, None, 3995, 2200, 2080, None, 3000, 1500, 4000]
        }

    test_dataframe = pd.DataFrame(test_cases)
    print(f"Before: {test_dataframe}")
    test_dataframe = missing_binary(test_dataframe, "binary_sequence")
    test_dataframe = preprocess_sex(test_dataframe, "Sex")
    test_dataframe = preprocess_mean(test_dataframe, "Urine output ml per day")
    print(f"After: {test_dataframe}")

    