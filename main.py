import numpy as np
import pandas as pd
import time

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
        unique_base_entries = dict(tuple(input_df.groupby(base_category)))

        for base_entry, base_entry_df in unique_base_entries.items():

            if base_entry_df.shape[0] < 5:
                continue

            output_dfs[base_entry] = base_entry_df

            for specific_category in specific_categories:
                # multi select
                if specific_category == 'Q3:Ethnicity/Race (Check all that apply) - Selected Choice':
                    values = np.unique(base_entry_df[specific_category].apply(lambda x: x.split(',')).sum())
                    unique_specific_entries = {val:base_entry_df.loc[base_entry_df[specific_category].str.contains(val)] for val in values}
                else:
                    unique_specific_entries = dict(tuple(base_entry_df.groupby(specific_category)))

                current_dfs = {}
                generate = True

                for specific_entry, new_df in unique_specific_entries.items():
                    if new_df.shape[0] < 5:
                        generate = False
                        break

                    current_dfs[f"{base_entry}+{specific_entry}"] = new_df
                
                if generate:
                    output_dfs.update(current_dfs)

    return output_dfs

if __name__ == '__main__':
    start = time.time()
    dfs_to_process = get_data_groups("./data/sample_data.txt")
    print(time.time() - start)
    for k, v in dfs_to_process.items():
        print(k)
