import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import time
from textwrap import wrap

def get_data_groups(inputFile: str) -> dict[str, pd.DataFrame]:
    output_dfs = {}

    input_df = pd.read_csv(inputFile, encoding='cp1252', sep='\t', header=0)

    # Drop rows with all empty cells
    input_df.dropna(axis=0, how='all', inplace=True)

    # create base
    if input_df.shape[0] < 5:
        return {}
    output_dfs['All'] = input_df

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

def get_bar_cats(df: pd.DataFrame) -> list[tuple[str, str]]:
    return [
        (cat.split(":")[0], cat) for cat in df if cat.startswith("Q") and cat[1].isdigit() and "TEXT" not in cat
    ]

def plot_df(df: pd.DataFrame, cats: list[tuple[str, str]], name: str):
    fig, axes = plt.subplots(len(cats), 1, figsize=(8.5, 6 * len(cats)))
    plt.subplots_adjust(hspace=0.4)
    i = 0
    for (cat_name, cat) in cats:
        try:
            print(cat)
            if not "Check all that apply" in cat:
                a = df[cat].value_counts().rename(cat_name).to_frame().transpose()
                a.plot.barh(stacked=True, ax=axes[i])
            else:
                values_df = df[cat].apply(lambda x: x.split(',') if not pd.isna(x) else ["No Answer"])
                values = np.unique(values_df.sum())
                a = pd.DataFrame({val:np.sum([val in x for x in values_df]) for val in values}, index=[cat_name])
                a.plot.barh(ax=axes[i])

            

            # cut off lables on the legend
            max_legend_label_length = 30
            handles, labels = axes[i].get_legend_handles_labels()
            shortened_labels = [label[:max_legend_label_length] + '...' if len(label) > max_legend_label_length else label for label in labels]
            axes[i].legend(handles, shortened_labels)
            axes[i].set_title("\n".join(wrap(cat, 60)), wrap=True)

            i += 1
        except TypeError as e:
            print(e)
            pass

    plt.savefig(f"out/{name}.pdf", format="pdf", bbox_inches="tight")

if __name__ == '__main__':
    start = time.time()
    dfs_to_process = get_data_groups("./data/sample_data.txt")
    print(time.time() - start)
    bar_cats = get_bar_cats(dfs_to_process['All'])
    for k, v in dfs_to_process.items():
        print(k)
        plot_df(v, bar_cats, k)
        print(time.time() - start)
