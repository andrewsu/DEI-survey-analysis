import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import pandas as pd
import sys
import time
from textwrap import wrap
from itertools import chain
import warnings

from logger import Logger

## global variables used in program
# used for logging
logger = Logger()

# dict used in next function for comparing to previous scores ("dfname-cat")
prev_scores = {}
# dict to store arrays of parent groupings for a given grouping
parents = {}
# dict to score max absolute scores for each category(question)
max_scores = {}

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
# creates "score" columns for values that needs a score
# sets a map from category -> max score (usually 2.0 or 3.0)
def order_and_score_values(df: pd.DataFrame, bar_cats: list[tuple[str, str]]) -> dict[str, int]:
    # special case for numeric ordering (put 2/3 digit numbers after 1 digit numbers)
    df.replace([r'^(\d\d\d)'], [r'B/\1'], inplace=True, regex=True)
    df.replace([r'^(\d\d)'], [r'A/\1'], inplace=True, regex=True)

    # specifies how groups of values should be ordered in bar graphs
    ordered_groups = [
        [r'(?i)^(Strongly agree)', r'(?i)^((?:Somewhat )*agree)', r'(?i)^(Neither agree nor disagree)', r'(?i)^((?:Somewhat )*disagree)', r'(?i)^(Strongly disagree)'],
        [r'(?i)^(Very likely)', r'(?i)^(Likely)', r'(?i)^(Somewhat likely)',r'(?i)^(Somewhat unlikely)', r'(?i)^(Unlikely)', r'(?i)^(Very unlikely)'],
        [r'(?i)^(Very satisfied)', r'(?i)^(Satisfied)', r'(?i)^(Neutral|Neither dissatisfied nor satisfied)', r'(?i)^(Dissatisfied)', r'(?i)^(Very dissatisfied)'],
        [r'(?i)^(Extremely confident)', r'(?i)^(Very confident)', r'(?i)^(Moderately confident)',r'(?i)^(Slightly confident)', r'(?i)^(Not at all confident)'],
        [r'(?i)^(Extremely important)', r'(?i)^(Moderately important)', r'(?i)^(Somewhat important)', r'(?i)^(A little important)', r'(?i)^(Not at all important)'],
        [r'(?i)^(Very prepared)', r'(?i)^(Moderately prepared)', r'(?i)^(Somewhat prepared)', r'(?i)^(Slightly prepared)'],
        [r'(?i)^(Yes, I have all the.*)', r'(?i)^(Moderately)\s*$', r'(?i)^(Somewhat)\s*$', r'(?i)^(A little)\s*$', r'(?i)^(Not at all)\s*$']
    ]

    original_vals = list(chain.from_iterable(ordered_groups))

    # These new values are just the old values prefaced with an index (to change the alpha ordering)
    new_vals = list(chain.from_iterable([[f"{index}/\\1" for index in range(len(ordered_group))] for ordered_group in ordered_groups]))

    ## generate scores
    scores = [
        [get_score(index, len(ordered_group)) for index in range(len(ordered_group))] for ordered_group in ordered_groups
    ]

    flat_scores = list(chain.from_iterable(scores))
    
    for (_, cat) in bar_cats:
        # convert values to scores
        df[f"scores-{cat}"] = df[cat].replace(original_vals, flat_scores, regex=True).apply(pd.to_numeric,errors='coerce')

        # store the maximum abs score
        max_scores[cat] = max(abs(df[f"scores-{cat}"].min()), df[f"scores-{cat}"].max())

    # fix alpha ordering of order groups
    df.replace(original_vals, new_vals, inplace=True, regex=True)

    # prefer not to say AND does not apply to me answers should be treated as no answer (NaN)
    df.replace([r'(?i).*prefer not to say.*', r'(?i).*does not apply to me.*'], [np.NaN, np.NaN], inplace=True, regex=True)

# gets the dataframe from a file
def get_dataframe(inputFile: str):

    if inputFile.endswith(".xlsx"):
        input_df = pd.read_excel(inputFile)
    else:
        # gets the first sheet by default
        input_df = pd.read_csv(inputFile, encoding='cp1252', sep='\t', header=0)

    # Drop rows with all empty cells
    input_df.dropna(axis=0, how='all', inplace=True)

    return input_df

# returns a dict of dataframes for which a report should be produced, indexed by the name of that group of data
def get_data_groups(input_df: pd.DataFrame, bar_cats: list[tuple[str, str]]) -> dict[str, pd.DataFrame]:
    output_dfs = {}

    # Replacement (useful for sorting)/Value scoring
    # gets max scores for each category (to put in context)
    order_and_score_values(input_df, bar_cats)

    # base level categories
    base_categories = ['All', 'Supervisor for Reporting', 'Strategic Unit/Org Level 3', 'Division/Org Level 2', 'Department/Org Level 1']

    # need to handle multiple ehtnicities being selected
    specific_categories = ['Q1:Gender Identity - Selected Choice', 'Q3:Ethnicity/Race (Check all that apply) - Selected Choice']

    # specifies parent relationships
    parent_categories = {
         'Department/Org Level 1': ['Division/Org Level 2', 'Strategic Unit/Org Level 3'],
         'Division/Org Level 2': ['Strategic Unit/Org Level 3']
    }

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

            # add parents
            if base_category in parent_categories:
                for parent_category in parent_categories[base_category]:
                    if base_entry_df[parent_category].value_counts().shape[0] == 1:
                        if not base_entry in parents:
                            parents[base_entry] = []
                        parents[base_entry].append(base_entry_df[parent_category].value_counts().index[0])

            output_dfs[base_entry] = base_entry_df

            for specific_category in specific_categories:
                # multi select
                if specific_category == 'Q3:Ethnicity/Race (Check all that apply) - Selected Choice':
                    split_df = base_entry_df[specific_category].apply(lambda x: x.split(',') if not pd.isna(x) else [])
                    values = np.unique(split_df.sum())
                    unique_specific_entries = {val:base_entry_df.loc[split_df.apply(lambda x: val in x)] for val in values}
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

# plots bar charts from the dataframe based on the categories passed in
def plot_bar_charts(df: pd.DataFrame, bar_cats: list[tuple[str, str]], pdf: PdfPages, name: str):
    # first page
    i = 0
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11))

    for (cat_name, cat) in bar_cats:
        try:
            # exclude Q31 (more of a freetext)
            if cat_name == "Q31":
                continue

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

            # get the score (scores are negative better, so flip)
            score = -df[f"scores-{cat}"].mean()
            if not np.isnan(score):
                prev_scores[f"{name}-{cat}"] = score

            # add score comparisons
            score_comps = []
            split_name = name.split('+')

            # parent depts score comps
            if split_name[0] in parents:
                for parent in parents[split_name[0]]:
                    if len(split_name) == 1 and f"{parent}-{cat}" in prev_scores:
                        score_comps.append((parent, prev_scores[f"{parent}-{cat}"]))
                    
                    # parent gender/ethnicty comp
                    if len(split_name) > 1 and f"{parent}+{split_name[1]}-{cat}" in prev_scores:
                        score_comps.append((f"{parent}/{split_name[1]}", prev_scores[f"{parent}+{split_name[1]}-{cat}"]))

            # institute score comp
            if len(split_name) == 1 and name != "All" and f"All-{cat}" in prev_scores:
                score_comps.append(("Institute", prev_scores[f"All-{cat}"]))
            
            # gender/ethnicty score comp
            if len(split_name) > 1 and split_name[0] != "All" and f"All+{split_name[1]}-{cat}" in prev_scores:
                score_comps.append((split_name[1], prev_scores[f"All+{split_name[1]}-{cat}"]))

            if len(score_comps) > 0:
                score_str = "\n".join([f"{comp[0]} Score: {comp[1]:.2}" for comp in score_comps])
            else:
                score_str = ""

            axes[i].legend(handles, shortened_labels, bbox_to_anchor=(1.0, -0.25), ncol=2)
            axes[i].set_title("\n".join(wrap(cat + f" [Responses: {total}]", 50)), wrap=True, ha="left", x=-0)

            # put score on fig, add bkg with bbox=dict(facecolor='red', alpha=0.5)
            axes[i].text(1, 1, f"Report Score (-{max_scores[cat]} to {max_scores[cat]}): {score:.2}\n{score_str}", verticalalignment='bottom', horizontalalignment='right', transform=axes[i].transAxes)

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
            if str(e) == "no numeric data to plot":
                modified_cat = cat.replace('\r\n', ' ').replace('\n', ' ')
                logger.log_data(f"The question had no responses (for this report): \"{modified_cat}\"")
            else:
                raise e
    
    # last page (if necessary)
    if i % 3 != 0:
        fig.tight_layout()
        pdf.savefig()
    
    plt.close(fig)

# "plots" freetext response from dataframe and categories given (just puts each question on a page in the the pdf)
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
            .95, 
            "\n\n".join(shortened_answer_list) + ("\n\n..." if answers_df.shape[1] > 10 else ""), 
            fontsize=8,
            transform=axes.transAxes,
            verticalalignment='top'
        )

        pdf.savefig()
        plt.close(fig)

# generates a pdf report for the dataframe and the appropriate categories
def generate_pdf(df: pd.DataFrame, bar_cats: list[tuple[str, str]], text_cats: list[str], name: str):
    with PdfPages(f"out/{name}.pdf") as pdf:
        plot_bar_charts(df, bar_cats, pdf, name)
        plot_text_cats(df, text_cats, pdf)

# entrypoint
if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
    else:
        input_filename = "./data/sample_survey_data_20230817.xlsx"

    if not os.path.exists(input_filename):
        print(f"Error: The file '{input_filename}' does not exist. Specify a valid input file.")
        sys.exit(1)

    # "PerformanceWarning" raised when adding "score" columns in order_and_score_values
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    main_df = get_dataframe(input_filename)
    bar_cats = get_bar_cats(main_df)
    text_cats = get_text_cats(main_df)

    dfs_to_process = get_data_groups(main_df, bar_cats)

    completed = 0
    total = len(dfs_to_process.items())

    full_start = time.time()

    for name, df in dfs_to_process.items():
        sys.stdout.write(f"\rCompleted {completed}/{total} Reports. Working on {name} report.".ljust(100))
        logger.log_data(f"[{name}]")
        start = time.time()

        generate_pdf(df, bar_cats, text_cats, name)

        # make sure everything is cleared from last plot
        plt.close('all')

        logger.log_data(f"Completed in {time.time() - start} seconds")
        logger.log_data("")
        logger.write_log()

        completed += 1

    logger.log_data(f"\n\nTotal time = {time.time() - full_start} seconds")
    logger.write_log()
