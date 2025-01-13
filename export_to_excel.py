import pandas as pd

def export_to_excel(df, filename):
    """
    Exports the DataFrame to an Excel file.

    Parameters:
    df (pandas.DataFrame): The DataFrame to export.
    filename (str): The name of the Excel file to create.

    Returns:
    None
    """
    df.to_excel(filename, index=False)

if __name__ == "__main__":
    # Example usage
    data = {'Feature1': [1, 0, 1, 0], 'Feature2': [0, 1, 0, 1]}
    df = pd.DataFrame(data)

    # Export the DataFrame to an Excel file named 'bloods_data.xlsx'
    export_to_excel(df, 'bloods_data.xlsx')

    print("Data exported to bloods_data.xlsx")
