import pandas as pd
import random
import re

# Load the first CSV file
# df1 = pd.read_csv("final_generation_1_to_5_456.csv")

# # Load the second CSV file
# df2 = pd.read_csv("final_generation_5_to_10_456.csv")

# # Merge the two dataframes row-wise
# merged_df = pd.concat([df1, df2], ignore_index=True)

# # Save the merged dataframe to a new CSV file
# merged_df.to_csv("merged_file.csv", index=False)


# Read the CSV file into a DataFrame
df1 = pd.read_csv("direction2/new_constraints.csv")
# Define the list of numbers of constraints
list_num_constraints = [3, 7, 11, 15, 19]

# Create an empty list to store rows
new_rows = []

# Iterate over each row in the DataFrame
for index, row in df1.iterrows():
    # Iterate over each number of constraints
    for numberOfConstraints in list_num_constraints:
        # Split the input string into a list of constraints based on numbering
        constraints_list = re.split(r'\d+\.\s*', row['Constraints'])
        constraints_list = [constraint.strip() for constraint in constraints_list if constraint.strip()]

        # Choose a random subset of constraints
        selected_constraints = constraints_list[:numberOfConstraints]
        final_constraints = "\n".join([f"{i + 1}. {constraint}" for i, constraint in enumerate(selected_constraints)])

        # Create a new row with updated values
        new_row = row.copy()
        new_row['Number_of_Constraints'] = numberOfConstraints
        new_row['SelectedConstraints'] = final_constraints
        new_rows.append(new_row)
# print(new_rows[0].to_frame().T)
# Concatenate the original DataFrame with the new rows
# df_result = pd.concat(new_rows, ignore_index=True)
df = pd.concat([s.to_frame().T for s in new_rows], ignore_index=True)
# print(df.head())
# Save the modified DataFrame to a new CSV file
df.to_csv("direction2/selected_constraints.csv", index=False)
