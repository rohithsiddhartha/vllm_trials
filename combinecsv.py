import pandas as pd
import random
import re
import os

# # Load the first CSV file
# df1 = pd.read_csv("final_generation_1_to_5_456.csv")

# # Load the second CSV file
# df2 = pd.read_csv("final_generation_5_to_10_456.csv")

# # Merge the two dataframes row-wise
# merged_df = pd.concat([df1, df2], ignore_index=True)

# # Save the merged dataframe to a new CSV file
# merged_df.to_csv("merged_file.csv", index=False)


# Read the CSV file into a DataFrame
df = pd.read_csv("/home/rbheemreddy_umass_edu/vllm_trials/Expansion/direction2/d2.csv")
print(df.info())
# Define the list of numbers of constraints
list_num_constraints = [7, 15, 23, 31, 39]

# Create an empty list to store rows
new_rows = []

class RegexSplitError(Exception):
    pass
    


# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Split the input string into a list of constraints based on numbering
    # print(row['Constraints'], type(row['Constraints']))
    constraints_list = re.split(r'\d+\.\s*', row['Constraints'])
    # constraints_list = re.split(r'\n(\d+\.\s*)', row['Constraints'], re.DOTALL)
    # print(constraints_list)
    constraints_list = [constraint.strip() for constraint in constraints_list if constraint.strip()]
    if len(constraints_list)!=40:
        error_message = [str(idx) + " " + i + '\n' for idx, i in enumerate(constraints_list)]
        # print(error_message)
        raise RegexSplitError("Error in regex splitting: \n \n {}".format(''.join(error_message)))
    # Iterate over each number of constraints
    for numberOfConstraints in list_num_constraints:
        
        # Choose a random subset of constraints
        selected_constraints = constraints_list[:numberOfConstraints]
        final_constraints = "\n".join([f"{i + 1}. {constraint}" for i, constraint in enumerate(selected_constraints)])
        
        # Create a new row with updated values
        new_row = row.copy()
        new_row['Number_of_Constraints'] = numberOfConstraints
        new_row['SelectedConstraints'] = final_constraints
        new_row['Direction'] = 'd2'
        new_rows.append(new_row)

# print(new_rows[0].to_frame().T)
# Concatenate the original DataFrame with the new rows
# df_result = pd.concat(new_rows, ignore_index=True)
df = pd.concat([s.to_frame().T for s in new_rows], ignore_index=True)
print(df.info())
# Save the modified DataFrame to a new CSV file
df.to_csv("Expansion/direction2/d2_with_constraints.csv", index=False)

