import pandas as pd

def convert_yes_no_fill_mode(df, columns):
    """
    Converts yes/no values to 1/0 in the specified columns of the DataFrame
    and fills missing values with the mode of the column.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    columns (list): A list of column names to convert.

    Returns:
    pandas.DataFrame: The modified DataFrame with yes/no converted to 1/0 and missing values filled.
    """
    for column in columns:
        if column in df.columns:
            # Fill missing values with the mode
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)
            # Convert yes/no to 1/0
            df[column] = df[column].map({'yes': 1, 'no': 0})
    return df

# Example usage
data = {'Feature1': ['yes', 'no', None, 'no'], 'Feature2': ['no', 'yes', 'no', None]}
df = pd.DataFrame(data)

# Convert yes/no to 1/0 in both 'Feature1' and 'Feature2' and fill missing values
df = convert_yes_no_fill_mode(df, ['Feature1', 'Feature2'])

print(df)
