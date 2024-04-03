from collections import defaultdict
import pandas as pd
import random
from vllm import LLM, SamplingParams
import torch
import re

# Model Definition
# model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
model_name = 'google/gemma-7b-it'
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

max_tokens = 4096
sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.8, top_p=0.95)

# Create an LLM.

def addNewStory(df, list_num_constraints, model_name='mistralai/Mistral-7B-Instruct-v0.1'):
    """Takes one instruction as input -> generates story based on the input -> proceed further with tuning the story based on the constraints selected"""
    # Initialize an empty DataFrame to store the results
    llm = LLM(model=model_name, dtype=torch.float16)  # Adjust model_name as needed
    
    single_instruction_df = pd.DataFrame(columns=['Keywords', 'BaseStory', 'SelectedKeywords', 'Number_of_Keywords', 'Final_Prompt', 'FinalGeneratedStory'])
    

    prev_instruction = None  # Initialize previous instruction as None

    for index, row in df.iterrows():
        # Check if the current instruction is different from the previous one
        if row['Keywords'] != prev_instruction:
            # First prompt
            print("Processing row number:", index)
            # prompt1 = f"Instruction: {row['Instruction']}. Generate the story within 500 words."
            prompt1 = f"Instruction: Act as an experienced writer, write a story of 500 words about daily life."
            output1 = llm.generate([prompt1], sampling_params)
            for output in output1:
                generated_story = output.outputs[0].text
        
        prev_instruction = row['Keywords']
        keywords_str = row['Keywords']
        keywords_list = keywords_str.split(', ')

        for number in list_num_constraints:
            keywords = keywords_list[:number]
            prompt2_start = f"Now modify the existing story to accommodate the following keywords: {keywords} into the Base Story and come up with a new story of around 500 words: "
            final_prompt = f"""User: "  {prompt1}" \n  Base Story: " {generated_story} " \n User Instruction: " {prompt2_start} """


            output2 = llm.generate([final_prompt], sampling_params)
            for output in output2:
                final_generated_story = output.outputs[0].text

            # Add the data to the result DataFrame

            single_instruction_df.loc[len(single_instruction_df)] = {
                'Keywords': row['Keywords'],
                'BaseStory': generated_story,
                'SelectedKeywords': keywords,
                'Number_of_Keywords': number,
                'Final_Prompt': final_prompt,
                'FinalGeneratedStory': final_generated_story
            }
        # break
    return single_instruction_df

# Read the CSV file into DataFrame
filename = "direction1/selected_keywords.csv"
auto_gen_eval = pd.read_csv(filename)

# Add new columns to store outputs
auto_gen_eval['BaseStory'] = ''
auto_gen_eval['Final_Prompt'] = ''
auto_gen_eval['FinalGeneratedStory'] = ''


# List of constraints to try
list_num_constraints = [5, 10, 15, 20, 25]


# Initialize an empty list to store all generated DataFrames
all_dfs = []
count=0


combined_df = addNewStory(auto_gen_eval, list_num_constraints, model_name)

# Append the generated DataFrame to the list
all_dfs.append(combined_df)

# Concatenate all DataFrames in the list into a single DataFrame
total_stories_df = pd.concat(all_dfs, ignore_index=True)
print(total_stories_df.head())
# Save the combined DataFrame to a single CSV file


if "direction1" in filename:
    d = "d1"
if "direction3" in filename:
    d = "d3"
elif "direction2" in filename:
    d = 'd2'
total_stories_df.to_csv(f"{base_path}/{base_path}_{d}_full_csv_same_base.csv", index=False)
