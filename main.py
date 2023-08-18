import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import time
from textwrap import wrap
import traceback
from itertools import chain

# generates a negative or positive score based on an index and length
# Example with length=5 (odd): 0 -> -2, 1 -> -1, 2 -> 0, 3 -> 1, 4 -> 2
# Example with length=4 (even): 0 -> -2, 1 -> -1, 2 -> 1, 3 -> 2
def get_score(index: int, length: int):
    if length % 2 == 0:
        if index < length/2:
            return index - length/2
        else:
            return index - length/2 + 1
    else:
        return index - (length-1)/2

# replaces some terms to ensure they are alphabetically ordered for bar chart generation
def order_values(df: pd.DataFrame):
    # special case for numeric ordering (put 2/3 digit numbers after 1 digit numbers)
    df.replace([r'^(\d\d\d)'], [r'B/\1'], inplace=True, regex=True)
    df.replace([r'^(\d\d)'], [r'A/\1'], inplace=True, regex=True)

    # specifies how groups of values should be ordered in bar graphs
    ordered_groups = [
        [r'(?i)^(Strongly agree)', r'(?i)^((?:Somewhat )*agree)', r'(?i)^(Neither agree nor disagree)', r'(?i)^((?:Somewhat )*disagree)', r'(?i)^(Strongly disagree)'],
        [r'(?i)^(Very likely)', r'(?i)^(Likely)', r'(?i)^(Somewhat likely)',r'(?i)^(Somewhat unlikely)', r'(?i)^(Unlikely)', r'(?i)^(Very unlikely)'],
        [r'(?i)^(Very satisfied)', r'(?i)^(Satisfied)', r'(?i)^(Neutral|Neither dissatisfied nor satisfied)', r'(?i)^(Dissatisfied)', r'(?i)^(Very dissatisfied)', r'(?i)^(Does not apply to me)'],
        [r'(?i)^(Extremely confident)', r'(?i)^(Very confident)', r'(?i)^(Moderately confident)',r'(?i)^(Slightly confident)', r'(?i)^(Not at all confident)'],
        [r'(?i)^(Extremely important)', r'(?i)^(Somewhat important)', r'(?i)^(Moderately important)', r'(?i)^(A little important)', r'(?i)^(Not at all important)'],
        [r'(?i)^(Very prepared)', r'(?i)^(Somewhat prepared)', r'(?i)^(Moderately prepared)', r'(?i)^(Slightly prepared)'],
        [r'(?i)^(Yes, I have all the.*)', r'(?i)^(Somewhat)\s*$', r'(?i)^(Moderately)\s*$', r'(?i)^(A little)\s*$', r'(?i)^(Not at all)\s*$']
    ]

    print([len(v) for v in ordered_groups])

    original_vals = list(chain.from_iterable(ordered_groups))

    # These new values are just the old values prefaced with an index (to change the alpha ordering)
    new_vals = list(chain.from_iterable([[f"{index}/\\1" for index in range(len(ordered_group))] for ordered_group in ordered_groups]))

    df.replace(original_vals, new_vals, inplace=True, regex=True)

    # prefer not to say answers should be treated as no answer (NaN)
    df.replace([r'(?i).*prefer not to say.*'], [np.NaN], inplace=True, regex=True)


    ## generate numbers
    numbers = [
        [get_score(index, len(ordered_group)) for index in range(len(ordered_group))] for ordered_group in ordered_groups
    ]
    print(numbers)

# returns a dict of dataframes for which a report should be produced, indexed by the name of that group of data
def get_data_groups(inputFile: str) -> dict[str, pd.DataFrame]:
    output_dfs = {}

    if inputFile.endswith(".xlsx"):
        input_df = pd.read_excel(inputFile)
    else:
        # gets the first sheet by default
        input_df = pd.read_csv(inputFile, encoding='cp1252', sep='\t', header=0)

    # Drop rows with all empty cells
    input_df.dropna(axis=0, how='all', inplace=True)

    # Replacement (useful for sorting)
    order_values(input_df)

    # base level categories
    base_categories = ['All', 'Supervisor for Reporting', 'Department/Org Level 1', 'Division/Org Level 2', 'Strategic Unit/Org Level 3']

    # need to handle multiple ehtnicities being selected
    specific_categories = ['Q1:Gender Identity - Selected Choice', 'Q3:Ethnicity/Race (Check all that apply) - Selected Choice']


    for base_category in base_categories:
        # "All" only has one value -> we want to create reports for each specific category though
        if base_category == 'All':
            unique_base_entries = {'All': input_df}
        else:
            unique_base_entries = dict(tuple(input_df.groupby(base_category)))

        # loops through values for this base category
        for base_entry, base_entry_df in unique_base_entries.items():
            # checks if size is too small for anonymity
            if base_entry_df.shape[0] < 5:
                continue

            output_dfs[base_entry] = base_entry_df

            for specific_category in specific_categories:
                # multi select
                if specific_category == 'Q3:Ethnicity/Race (Check all that apply) - Selected Choice':
                    values = np.unique(base_entry_df[specific_category].dropna().apply(lambda x: x.split(',')).sum())
                    unique_specific_entries = {val:base_entry_df.loc[base_entry_df[specific_category].str.contains(val, na=False)] for val in values}
                # not multi select
                else:
                    unique_specific_entries = dict(tuple(base_entry_df.groupby(specific_category)))

                current_dfs = {}
                generate = True

                # loops through values for this specific category
                for specific_entry, new_df in unique_specific_entries.items():
                    # checks if size is too small for anonymity
                    if new_df.shape[0] < 5:
                        if base_category == 'All':
                            # for 'All' we do not need all values to be over 5, only the current value
                            continue
                        else:
                            generate = False
                            break

                    current_dfs[f"{base_entry}+{specific_entry}"] = new_df
                
                if generate:
                    output_dfs.update(current_dfs)

    return output_dfs

# gets a list of categories which should generate a bar chart, with shortened names
def get_bar_cats(df: pd.DataFrame) -> list[tuple[str, str]]:
    return [
        (cat.split(":")[0], cat) for cat in df if cat.startswith("Q") and cat[1].isdigit() and "TEXT" not in cat
    ]

# gets a list of categories which should generate a list of text responses
def get_text_cats(df: pd.DataFrame) -> list[str]:
    return [
        cat for cat in df if "TEXT" in cat or "Q31" in cat
    ]

def plot_bar_charts(df: pd.DataFrame, bar_cats: list[tuple[str, str]], pdf: PdfPages):
    # first page
    i = 0
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11))

    for (cat_name, cat) in bar_cats:
        try:
            # exclude Q31 (more of a freetext)
            if cat_name == "Q31":
                continue

            print(cat)

            # one choice vs multi select
            if not "Check all that apply" in cat:
                # goal is to create a dataframe with values and frequency, we want this sorted by alpha order (sort_index)
                plottable_df = df[cat].value_counts().sort_index().rename(cat_name).to_frame().transpose()
                
                plottable_df.plot.barh(stacked=True, ax=axes[i])
                total = df[cat].dropna().shape[0]
            else:
                # multi select values are split by ","
                values_df = df[cat].apply(lambda x: x.split(',') if not pd.isna(x) else ["No Answer"])
                values = np.unique(values_df.sum())
                # we want to sort the values by alpha order
                values.sort()

                plottable_df = pd.DataFrame({val:np.sum([val in x for x in values_df]) for val in values}, index=[cat_name])

                plottable_df.plot.barh(ax=axes[i])
                total = df[cat].shape[0]

            # cut off lables on the legend
            max_legend_label_length = 30
            handles, labels = axes[i].get_legend_handles_labels()
            shortened_labels = [(label[:max_legend_label_length] + '...' if len(label) > max_legend_label_length else label) + ' ({:} / {:.1%})'.format(plottable_df[label][cat_name], plottable_df[label][cat_name]/total) for label in labels]

            axes[i].legend(handles, shortened_labels, bbox_to_anchor=(1.0, -0.25), ncol=2)
            axes[i].set_title("\n".join(wrap(cat + f" [Responses: {total}]", 60)), wrap=True)

            i += 1

            # save old page, create a new page
            if i % 3 == 0:
                fig.tight_layout()
                pdf.savefig()
                plt.close(fig)
                fig, axes = plt.subplots(3, 1, figsize=(8.5, 11))
                i = 0

        # occurs when no data to plot
        except TypeError as e:
            print(traceback.format_exc())
            print(e)
    
    # last page (if necessary)
    if i % 3 != 0:
        fig.tight_layout()
        pdf.savefig()
        plt.close(fig)

def plot_text_cats(df: pd.DataFrame, text_cats: list[str], pdf: PdfPages):
    for cat in text_cats:
        fig, axes = plt.subplots(1, 1, figsize=(8.5, 11))
        axes.axis('off')

        answers_df = df[cat].value_counts().to_frame().transpose()
        if answers_df.empty:
            continue

        # to each answer => wrap at 60 characters, cutoff at 180 characters, add count at the end [cut off to 10 answers]
        shortened_answer_list = [
            "\n".join(wrap(i, 60))[:180] + ("..." if len(i) > 180 else "") + f" ({answers_df[i]['count']})" for i in answers_df
        ][:10]

        axes.set_title(cat, wrap=True)
        axes.text(
            0, 
            0, 
            "\n\n".join(shortened_answer_list) + ("\n\n..." if answers_df.shape[1] > 10 else ""), 
            fontsize=8
        )

        pdf.savefig()
        plt.close(fig)

def generate_pdf(df: pd.DataFrame, bar_cats: list[tuple[str, str]], text_cats: list[str], name: str):
    with PdfPages(f"out/{name}.pdf") as pdf:
        plot_bar_charts(df, bar_cats, pdf)
        plot_text_cats(df, text_cats, pdf)


if __name__ == '__main__':
    start = time.time()
    dfs_to_process = get_data_groups("./data/sample_survey_data_20230817.xlsx")
    print(time.time() - start)
    bar_cats = get_bar_cats(dfs_to_process['All'])
    text_cats = get_text_cats(dfs_to_process['All'])
    for name, df in dfs_to_process.items():
        print(name)
        generate_pdf(df, bar_cats, text_cats, name)

        # make sure everything is cleared from last plot
        plt.close('all')

        print(time.time() - start)
