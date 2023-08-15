import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import time
from textwrap import wrap
import traceback
from itertools import chain

def adjust_df(df: pd.DataFrame):
    val_orders = [
        [r'(?i)^(Strongly agree)', r'(?i)^((?:Somewhat )*agree)', r'(?i)^(Neither agree nor disagree)', r'(?i)^((?:Somewhat )*disagree)', r'(?i)^(Strongly disagree)'],
        [r'(?i)^(Very likely)', r'(?i)^(Likely)', r'(?i)^(Somewhat likely)',r'(?i)^(Somewhat unlikely)', r'(?i)^(Unlikely)', r'(?i)^(Very unlikely)'],
        [r'(?i)^(Very satisfied)', r'(?i)^(Satisfied)', r'(?i)^(Neutral)', r'(?i)^(Neither dissatisfied nor satisfied)', r'(?i)^(Dissatisfied)', r'(?i)^(Very dissatisfied)', r'(?i)^(Does not apply to me)'],
        [r'(?i)^(Extremely confident)', r'(?i)^(Very confident)', r'(?i)^(Moderately confident)',r'(?i)^(Slightly confident)', r'(?i)^(ot at all confident)'],
        [r'(?i)^(Extremely important)', r'(?i)^(Somewhat important)', r'(?i)^(Moderately important)', r'(?i)^(A little important)', r'(?i)^(Not at all important)']
    ]

    original_vals = list(chain.from_iterable(val_orders))
    new_vals = list(chain.from_iterable([[f"{val}/\\1" for val in range(len(order))] for order in val_orders]))

    df.replace(original_vals, new_vals, inplace=True, regex=True)

def get_data_groups(inputFile: str) -> dict[str, pd.DataFrame]:
    output_dfs = {}

    input_df = pd.read_csv(inputFile, encoding='cp1252', sep='\t', header=0)

    # Drop rows with all empty cells
    input_df.dropna(axis=0, how='all', inplace=True)

    # Replacement (useful for sorting)
    adjust_df(input_df)

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

def get_text_cats(df: pd.DataFrame) -> list[str]:
    return [
        cat for cat in df if "TEXT" in cat or "Q31" in cat
    ]

def plot_df(df: pd.DataFrame, bar_cats: list[tuple[str, str]], text_cats: list[str], name: str):
    with PdfPages(f"out/{name}.pdf") as pdf:
        i = 0

        # BAR CHARTS
        # first page
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 11))

        for (cat_name, cat) in bar_cats:
            try:
                ## exclude Q31 (more of a freetext)
                if cat_name == "Q31":
                    continue

                print(cat)
                if not "Check all that apply" in cat:
                    a = df[cat].value_counts().sort_index().rename(cat_name).to_frame().transpose()
                    a.plot.barh(stacked=True, ax=axes[i])
                    total = df[cat].dropna().shape[0]
                else:
                    values_df = df[cat].apply(lambda x: x.split(',') if not pd.isna(x) else ["No Answer"])
                    values = np.unique(values_df.sum())
                    values.sort()
                    a = pd.DataFrame({val:np.sum([val in x for x in values_df]) for val in values}, index=[cat_name])
                    a.plot.barh(ax=axes[i])
                    total = df[cat].shape[0]

                # cut off lables on the legend
                max_legend_label_length = 30
                handles, labels = axes[i].get_legend_handles_labels()
                shortened_labels = [(label[:max_legend_label_length] + '...' if len(label) > max_legend_label_length else label) + ' ({:} / {:.1%})'.format(a[label][cat_name], a[label][cat_name]/total) for label in labels]

                axes[i].legend(handles, shortened_labels, bbox_to_anchor=(1.0, -0.25), ncol=2)
                axes[i].set_title("\n".join(wrap(cat + f" [Responses: {total}]", 60)), wrap=True)

                i += 1

                # subsequent pages
                if i % 3 == 0:
                    fig.tight_layout()
                    pdf.savefig()
                    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11))
                    i = 0

            except TypeError as e:
                print(traceback.format_exc())
                print(e)
        
        if i % 3 != 0:
            pdf.savefig()

        # TEXT RESPONSES
        for cat in text_cats:
            fig, axes = plt.subplots(1, 1, figsize=(8.5, 11))
            axes.axis('off')
            values_df = df[cat].value_counts().to_frame().transpose()
            if values_df.empty:
                continue

            axes.set_title(cat, wrap=True)
            axes.text(0, 0, "\n\n".join(["\n".join(wrap(i, 60))[:180] + ("..." if len(i) > 180 else "") + f" ({values_df[i][cat]})" for i in values_df][:10]) + ("\n\n..." if values_df.shape[1] > 10 else ""), fontsize=8)

            pdf.savefig()



if __name__ == '__main__':
    start = time.time()
    dfs_to_process = get_data_groups("./data/sample_data.txt")
    print(time.time() - start)
    bar_cats = get_bar_cats(dfs_to_process['All'])
    text_cats = get_text_cats(dfs_to_process['All'])
    for name, df in dfs_to_process.items():
        print(name)
        plot_df(df, bar_cats, text_cats, name)
        print(time.time() - start)
        break