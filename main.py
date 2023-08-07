import numpy as np
import pandas as pd

def get_data_groups(inputFile: str):
    output_dfs = {}

    input_df = pd.read_csv(inputFile, encoding='cp1252', sep='\t', header=0)

    # Drop rows with all empty cells
    input_df.dropna(axis=0, how='all', inplace=True)

    # base level categories
    base_categories = ['Supervisor for Reporting', 'Department/Org Level 1', 'Division/Org Level 2', 'Strategic Unit/Org Level 3']

    # need to handle multiple ehtnicities being selected
    specific_categories = ['Q1:Gender Identity - Selected Choice', 'Q3:Ethnicity/Race (Check all that apply) - Selected Choice']

    for base_category in base_categories:
        unique_base_entries = input_df[base_category].drop_duplicates()
        unique_base_entries.dropna(axis=0, how='all', inplace=True)

        for base_entry in unique_base_entries:
            base_entry_df = input_df[input_df[base_category] == base_entry]

            if base_entry_df.shape[0] < 5:
                continue

            output_dfs[base_entry] = base_entry_df

            for specific_category in specific_categories:
                unique_specific_entries = base_entry_df[specific_category].drop_duplicates()
                unique_specific_entries.dropna(axis=0, how='all', inplace=True)

                if unique_specific_entries.shape[0] < 2:
                    continue

                current_dfs = {}
                generate = True

                for specific_entry in unique_specific_entries:
                    new_df = base_entry_df[base_entry_df[specific_category] == specific_entry]

                    if new_df.shape[0] < 5:
                        generate = False
                        break

                    current_dfs[f"{base_entry}+{specific_entry}"] = new_df
                
                if generate:
                    output_dfs.update(current_dfs)

    return output_dfs

if __name__ == '__main__':
    dfs_to_process = get_data_groups("./data/sample_data.txt")
    for k, v in dfs_to_process.items():
        print(k)
