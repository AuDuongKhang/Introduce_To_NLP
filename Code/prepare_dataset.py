from transfer_to_csv import transfer_to_csv

import pandas as pd

from sklearn.model_selection import train_test_split
from datasets import Dataset


output_file_path = './data/dataset.csv'
file_paths = ['tqdn1_ch_vn.csv', 'tqdn2_ch_vn.csv', 'tqdn3_ch_vn.csv']
chinese_file = 'corpus.zh'
vietnamese_file = 'corpus.vi'


def save_dataset(output_file_path, file_paths, chinese_file, vietnamese_file):
    # Initialize an empty DataFrame to consolidate all data
    consolidated_data = pd.DataFrame(columns=['source', 'target'])

    # Process the CSV files
    for file_path in file_paths:
        data = pd.read_csv("./data/" + file_path)

        cleaned_data = data.dropna(subset=['source', 'target'])

        # Append to the consolidated DataFrame
        consolidated_data = pd.concat(
            [consolidated_data, cleaned_data], ignore_index=True)

    # Process the text files
    with open("./data/" + chinese_file, 'r', encoding='utf-8') as zh_file:
        chinese_lines = zh_file.readlines()

    with open("./data/" + vietnamese_file, 'r', encoding='utf-8') as vi_file:
        vietnamese_lines = vi_file.readlines()

    # Ensure both files have the same number of lines
    if len(chinese_lines) != len(vietnamese_lines):
        raise ValueError("The files do not have the same number of lines!")

    # Create a DataFrame with Chinese in 'target' and Vietnamese in 'label'
    text_file_data = pd.DataFrame({
        'source': [line.strip() for line in chinese_lines],
        'target': [line.strip() for line in vietnamese_lines]
    })

    # Append text file data to the consolidated DataFrame
    consolidated_data = pd.concat(
        [consolidated_data, text_file_data], ignore_index=True)

    consolidated_data = consolidated_data.dropna(subset=['source', 'target'])
    consolidated_data = consolidated_data[consolidated_data['source'].str.strip(
    ) != ""]
    consolidated_data = consolidated_data[consolidated_data['target'].str.strip(
    ) != ""]

    # Save the consolidated DataFrame to a CSV file
    consolidated_data.to_csv(output_file_path, index=False, encoding='utf-8')

    print(
        f"The combined dataset has been saved to {output_file_path} as a CSV file.")


def split_dataset():
    data = pd.read_csv(output_file_path)
    train_data, temp_data = train_test_split(
        data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42)

    train_data.to_csv("./data/train_data.csv", index=False)
    val_data.to_csv("./data/val_data.csv", index=False)
    test_data.to_csv("./data/test_data.csv", index=False)

    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)
    test_dataset = Dataset.from_pandas(test_data)

    return train_dataset, val_dataset, test_dataset


def main():
    transfer_to_csv("tqdn1_ch_vn.xlsx")
    transfer_to_csv("tqdn2_ch_vn.xlsx")
    transfer_to_csv("tqdn3_ch_vn.xlsx")
    save_dataset(output_file_path, file_paths, chinese_file, vietnamese_file)
    split_dataset()


if __name__ == "__main__":
    main()
