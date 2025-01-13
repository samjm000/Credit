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
                # Fill missing values with 0
                df[column].fillna(0, inplace=True)
    return df

def preprocess_yes_no(df, columns):
    """
    Converts 'yes' to 1 and 'no' to 0, and fills missing values with 0 in the specified columns of the DataFrame.

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
                # Ensure the column is treated as string before applying string methods
                df[column] = df[column].astype(str).str.strip().str.lower()
                
                # Map 'yes' to 1, 'no' to 0, leave other values untouched
                df[column] = df[column].map({'yes': 1, 'no': 0}, na_action='ignore')
                
                # Fill missing values with 0
                df[column].fillna(0, inplace=True)
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
        if column in df.columns:
            if not df[column].empty:
                # Fill missing values with the mode
    for column in columns:
        if column in df.columns:
            if not df[column].empty:
                # Fill missing values with the mode
                mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else None
                if mode_value is not None:
                    df[column].fillna(mode_value, inplace=True)
    return df

def preprocess_mean(df, columns):
    """
    Fills missing values with the mean in the specified numerical columns of the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    columns (list): A list of numerical column names to preprocess.

    Returns:
    pandas.DataFrame: The modified DataFrame with missing values filled with the mean.
    """
    for column in columns:
        if column in df.columns:
            # Fill missing values with the mean
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)
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
        df[column] = df[column].apply(lambda x: non_negative_mean if x < 0 else x)
    
    return df

if __name__ == "__main__":
    # Example usage
    print("test")
  