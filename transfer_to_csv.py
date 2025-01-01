import pandas as pd


def transfer_to_csv(file_name):
    # Load the Excel file
    df = pd.read_excel("./data/" + file_name)

    # Rename columns to 'text' and 'label'
    df.columns = ['source', 'target']

    # Save the DataFrame to a CSV file
    output_file_name = "./data/" + file_name.split(".")[0] + ".csv"
    df.to_csv(output_file_name, encoding="utf-8", index=False)
