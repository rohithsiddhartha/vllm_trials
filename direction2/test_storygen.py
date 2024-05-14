from collections import defaultdict
import pandas as pd
import random
from vllm import LLM, SamplingParams
import torch
import re



max_tokens = 4096
sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.8, top_p=0.95)

# Create an LLM.

def addNewStory(df, list_num_constraints, llm=None):
    """Takes one instruction as input -> generates story based on the input -> proceed further with tuning the story based on the constraints selected"""
    # Initialize an empty DataFrame to store the results
    single_instruction_df = pd.DataFrame(columns=['Instruction', 'Category', 'Constraints', 'BaseStory', 'SelectedConstraints', 'Number_of_Constraints', 'Final_Prompt', 'FinalGeneratedStory'])
    

    prev_instruction = None  # Initialize previous instruction as None

    for index, row in df.iterrows():
        # # Check if the current instruction is different from the previous one
        # if row['Instruction'] != prev_instruction:
        #     # First prompt
        #     print("Processing row number:", index)
        #     prompt1 = f"Instruction: {row['Instruction']}. Generate the story within 500 words."
        #     output1 = llm.generate([prompt1], sampling_params)
        #     for output in output1:
        #         generated_story = output.outputs[0].text
        
        # prev_instruction = row['Instruction']
        print("Processing row number:", index)
        

        prompt2_start = f"Now modify the existing story to accommodate the following constraints: {row['SelectedConstraints']} into the LLM generated story and come up with a new story in 500 words: "
        final_prompt = f"""User: "  {row['Instruction']}" \n BaseStory: " {row["BaseStory"]} " \n User Instruction: " {prompt2_start} """


        output2 = llm.generate([final_prompt], sampling_params)
        for output in output2:
            final_generated_story = output.outputs[0].text

        # Add the data to the result DataFrame

        single_instruction_df.loc[len(single_instruction_df)] = {
            'Instruction': row['Instruction'],
            'Category': row['Category'],
            'Constraints': row['Constraints'],
            'BaseStory': row['BaseStory'],
            'SelectedConstraints': row['SelectedConstraints'],
            'Number_of_Constraints': row['Number_of_Constraints'],
            'Final_Prompt': final_prompt,
            'FinalGeneratedStory': final_generated_story
        }
        
    return single_instruction_df

# Read the CSV file into DataFrame


# Model Definition
model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
# model_name = 'google/gemma-7b-it'
# model_name = 'meta-llama/Llama-2-7b-chat-hf'

# -hf models only work as they have config.json
# model_name = 'meta-llama/Llama-2-7b'
# model_name = 'meta-llama/Llama-2-7b-hf'
# model_name = 'meta-llama/Llama-2-7b-chat'
# model_name = 'meta-llama/Llama-2-7b-chat-hf'

if "gemma" in model_name:
    base_path = 'gemma'
if "Llama" in model_name:
    base_path = 'llama'
if "Mistral" in model_name:
    base_path = 'mistral'

llm = LLM(model=model_name, dtype=torch.float16)  # Adjust model_name as needed


def generalcall():
    filename = "/home/rbheemreddy_umass_edu/vllm_trials/direction2/d2_gpt_selected_constraints.csv"
    auto_gen_eval = pd.read_csv(filename)

    # Add new columns to store outputs
    # auto_gen_eval['BaseStory'] = ''
    auto_gen_eval['Final_Prompt'] = ''
    auto_gen_eval['FinalGeneratedStory'] = ''


    # List of constraints to try
    list_num_constraints = [3, 7, 11, 15, 19]



    # Initialize an empty list to store all generated DataFrames
    all_dfs = []
    count=0


    combined_df = addNewStory(auto_gen_eval, list_num_constraints, llm)

    # Append the generated DataFrame to the list
    all_dfs.append(combined_df)

    # Concatenate all DataFrames in the list into a single DataFrame
    total_stories_df = pd.concat(all_dfs, ignore_index=True)
    print(total_stories_df.head())
    # Save the combined DataFrame to a single CSV file

    if "direction3" in filename:
        d = "d3"
    elif "direction2" in filename:
        d = 'd2'
    total_stories_df.to_csv(f"{base_path}/{d}_GPTBaseStory_{base_path}_{d}.csv", index=False)
