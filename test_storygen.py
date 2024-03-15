from collections import defaultdict
import pandas as pd
import random
from vllm import LLM, SamplingParams
import torch
import re

# Model Definition
model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
max_tokens = 4096
sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.8, top_p=0.95)

# Create an LLM.

def addNewStory(df, list_num_constraints, llm=None):
    """Takes one instruction as input -> generates story based on the input -> proceed further with tuning the story based on the constraints selected"""
    # Initialize an empty DataFrame to store the results
    single_instruction_df = pd.DataFrame(columns=['Instruction', 'Category', 'Constraints', 'BaseStory','FinalConstraints', 'Number_of_Constraints', 'Final_Prompt', 'FinalGeneratedStory'])
    
    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        # First prompt
        prompt1 = f"Instruction: {row['Instruction']}. Generate the story within 500 words."
        output1 = llm.generate([prompt1], sampling_params)
        for output in output1:
            generated_story = output.outputs[0].text


        for numberOfConstraints in list_num_constraints:
            final_constraints = ""
            # break

            # Split the input string into a list of constraints based on numbering
            constraints_list = re.split(r'\d+\.\s*', row['Constraints'])
            constraints_list = [constraint.strip() for constraint in constraints_list if constraint.strip()]

            # Choose a random subset of constraints
            selected_constraints = random.sample(constraints_list, numberOfConstraints)
            final_constraints = "\n".join([f"{i + 1}. {constraint}" for i, constraint in enumerate(selected_constraints)])

            prompt2_start = f"Now modify the existing story to accommodate the following constraints: {final_constraints} into the LLM generated story and come up with a new story within 500 words: "
            final_prompt = f"""User: "  {row['Instruction']}" \n LLM generated story: " {generated_story} " \n User Instruction: " {prompt2_start} """

            # Second prompt
            output2 = llm.generate([final_prompt], sampling_params)
            for output in output2:
                final_generated_story = output.outputs[0].text

            # Add the data to the result DataFrame

            single_instruction_df.loc[len(single_instruction_df)] = {
                'Instruction': row['Instruction'],
                'Category': row['Category'],
                'Constraints': row['Constraints'],
                'BaseStory': generated_story,
                'FinalConstraints': final_constraints,
                'Number_of_Constraints': numberOfConstraints,
                'Final_Prompt': final_prompt,
                'FinalGeneratedStory': final_generated_story
            }


        # result_df = result_xf.append(data_entry, ignore_index=True)

    return single_instruction_df

# Read the CSV file into DataFrame
filename = "new_constraints.csv"
auto_gen_eval = pd.read_csv(filename)

# Add new columns to store outputs
auto_gen_eval['BaseStory'] = ''
auto_gen_eval['FinalConstraints'] = ''
auto_gen_eval["Number_of_Constraints"] = ''
auto_gen_eval['Final_Prompt'] = ''
auto_gen_eval['FinalGeneratedStory'] = ''

# Initialize an empty DataFrame to store the results

# List of constraints to try
# list_num_constraints = [1, 3, 5, 7, 11, 13, 17, 20]
list_num_constraints = [1, 3, 5]

# # Get the rows from the 8th index with only the specified columns
# subset_df = auto_gen_eval.iloc[8:9, :][['Instruction', 'Category', 'Constraints']]
# print(subset_df.head())

# llm = LLM(model=model_name, dtype=torch.float16)
# # combined_df = pd.DataFcombined_dframe(columns=['Instruction', 'Category', 'Constraints', 'FinalConstraints', 'Final_Prompt', 'FinalGeneratedStory'])

# combined_df = addNewStory(subset_df, list_num_constraints, llm)

# # # Save the combined DataFrame to a single CSV file
# combined_df.to_csv("new_combined_stories.csv", index=False)


# Initialize an empty list to store all generated DataFrames
all_dfs = []
count=1
llm = LLM(model=model_name, dtype=torch.float16)  # Adjust model_name as needed


# Assuming auto_gen_eval is your DataFrame containing multiple rows of data
for index, row in auto_gen_eval.iterrows():
    # Extract relevant columns from the current row
    print(f"Generating base story for row: {index}")
    subset_df = auto_gen_eval.iloc[index:index+1, :][['Instruction', 'Category', 'Constraints']]

    # Initialize your LLM model

    # Call the addNewStory function to generate new stories for the current row
    combined_df = addNewStory(subset_df, list_num_constraints, llm)

    # Append the generated DataFrame to the list
    all_dfs.append(combined_df)
    count+=1
    if count==3:
        break

# Concatenate all DataFrames in the list into a single DataFrame
total_stories_df = pd.concat(all_dfs, ignore_index=True)

# Save the combined DataFrame to a single CSV file
total_stories_df.to_csv("total_stories.csv", index=False)
