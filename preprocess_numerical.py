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
                # Convert 'yes' to 1 and 'no' to 0
                df[column] = df[column].map({'yes': 1, 'no': 0})
                # Fill missing values with 0
                df[column].fillna(0, inplace=True)
    return df

def preprocess_mode(df, columns):
    """
    Fills missing values with the mode in the specified columns of the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    columns (str or list): A column name or list of column names to preprocess.

    Returns:
    pandas.DataFrame: The modified DataFrame with missing values filled.
    """
    if isinstance(columns, str):
        columns = [columns]
    
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
    columns (str or list): A column name or list of numerical column names to preprocess.

    Returns:
    pandas.DataFrame: The modified DataFrame with missing values filled with the mean.
    """
    if isinstance(columns, str):
        columns = [columns]
    
    for column in columns:
        if column in df.columns:
            # Fill missing values with the mean
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)
    return df

if __name__ == "__main__":
    # Example usage
    data = {
        'Sex': ['M', 'F', None, 'F', 'M'],
        'TimeBetweenLastTreatmentAndAdmission': [5, 10, None, 8, 12],
        'AnotherNumericFeature': [1.5, 2.0, 2.5, None, 3.0],
        'Feature1': ['yes', 'no', None, 'no'],
        'Feature2': ['no', 'yes', 'no', None]
    }
    df = pd.DataFrame(data)

    # Preprocess the 'Sex' column
    df = preprocess_sex(df, 'Sex')

    # Preprocess the specified numerical columns with mean imputation
    numeric_columns = ['TimeBetweenLastTreatmentAndAdmission', 'AnotherNumericFeature']
    df = preprocess_mean(df, numeric_columns)

    # Preprocess the specified yes/no columns
    yes_no_columns = ['Feature1', 'Feature2']
    df = preprocess_yes_no(df, yes_no_columns)

    # Preprocess the specified categorical columns with mode imputation (not yes/no)
    mode_columns = []  # Add any additional columns needing mode imputation
    df = preprocess_mode(df, mode_columns)

    print(df)
