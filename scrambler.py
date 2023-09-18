import pandas as pd
import sys
import os

# scramble columns after/indluding 36 (0 indexed)
column_seperate_point = 36

def scramble(input_filename: str, output_filename: str):
    input_df = pd.read_excel(input_filename)

    for i in range(column_seperate_point, input_df.shape[1]):
        col = input_df.columns[i]
        input_df[col] = input_df[col].sample(frac=1).values

    input_df.to_excel(output_filename, sheet_name="Data", index=False)

# entrypoint
if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
    else:
        input_filename = "./data/sample_survey_data_20230904b.xlsx"

    if not os.path.exists(input_filename):
        print(f"Error: The file '{input_filename}' does not exist. Specify a valid input file.")
        sys.exit(1)

    output_filename = input_filename.replace(".xlsx", "-scrambled.xlsx")

    scramble(input_filename, output_filename)