from collections import defaultdict
import pandas as pd
import random
from vllm import LLM, SamplingParams
import torch

# Model Definition
model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
max_tokens = 8192
sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model=model_name, dtype=torch.float16) 

def addNewStory(df, llm, numberOfConstraints):
    x = 0
    for index, row in df.iterrows():
        # First prompt
        print("Row:", row)
        print("Row[Instruction]:", row['Instruction'])
        print("Row[Constraints]:", row['Constraints'])
        print("Row[Category]:", row['Category'])
        output1 = llm.generate([row['Instruction']], sampling_params)
        final_constraints = ""

        for output in output1:
            generated_text = output.outputs[0].text

        # final prompt formation
        for i in range(numberOfConstraints):
            final_constraints += random.choice(str(row['Constraints']))
        
        print("Final Constraints selected: ", final_constraints)
        prompt2_start = f"Now modify the existing story to accommodate the following constraints: {final_constraints} into the LLM generated story and come up with a new story: "
        final_prompt = f"""User: "  {row['Instruction']}" \n LLM generated story: " {generated_text} " \n User Instruction: " {prompt2_start} """

        # second prompt
        output2 = llm.generate([final_prompt], sampling_params)
        for output in output2:
            final_generated_story = output.outputs[0].text

        # Add the data to the DataFrame
        df.at[index, 'Final_Prompt'] = final_prompt
        df.at[index, 'FinalGeneratedStory'] = final_generated_story
        # break
        x+=1
        if x==4:
            break

    # Save the modified DataFrame to the same CSV file
    df.to_csv("/home/rbheemreddy_umass_edu/vllm_trial1/listnew.csv", index=False)

# Read the CSV file into DataFrame
filename = "constraints.csv"
auto_gen_eval = pd.read_csv(filename)

# Add new columns to store outputs
# auto_gen_eval['Final_Prompt'] = ''
# auto_gen_eval['FinalGeneratedStory'] = ''

# Example usage
addNewStory(auto_gen_eval, llm, 3)
