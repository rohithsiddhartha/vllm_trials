from collections import defaultdict
import pandas as pd
import numpy as np
import random
from vllm import LLM, SamplingParams
import torch
import subprocess as sp

filename = "constraints.csv"
auto_gen_eval = pd.read_csv("/home/rbheemreddy_umass_edu/vllm_trial1/constraints.csv")

instruction = auto_gen_eval.Instruction[0]
category = auto_gen_eval.Category[0]
constraints = auto_gen_eval.Constraints[0]
numberOfConstraints = 3


# Model Definition
model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
max_tokens = 8192
sampling_params = SamplingParams(max_tokens= max_tokens, temperature=0.8, top_p=0.95)
# Create an LLM.
llm = LLM(model=model_name, dtype=torch.float16) #, gpu_memory_utilization=0.99)

def addNewStory(df, llm, numberOfConstraints):
    # First prompt
    output1 = llm.generate(instruction, sampling_params)
    final_constraints = ""

    for output in output1:
        # prompt = output.prompt
        generated_text = output.outputs[0].text
    # final prompt formation
    for i in range(numberOfConstraints):
        final_constraints += random.choice(constraints)
    prompt2_start = f"Now modify the existing story to accommodate the following constraints: {final_constraints} into the LLM generated story and come up with a new story: "
    final_prompt = [f"""User: "  {instruction}" \n LLM generated story: " {generated_text} " \n User Instruction: " {prompt2_start} """]
    # second prompt
    output2 = llm.generate(final_prompt, sampling_params)
    for output in output2:
        second_prompt = output.prompt
        final_generated_story = output.outputs[0].text

    df['Final_Prompt'][0] = final_prompt
    df['FinalGeneratedStory'][0] = final_generated_story
    # Save the DataFrame to a CSV file
    # df.loc[len(df.index)] = [instruction, constraints, category, ]
    df.to_csv("/home/rbheemreddy_umass_edu/vllm_trial1/new.csv", index=False)

for i in range(5):
    addNewStory(auto_gen_eval, llm, i)
    break
