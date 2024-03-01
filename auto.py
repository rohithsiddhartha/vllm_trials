from collections import defaultdict
import pandas as pd
import numpy as np
import random
from vllm import LLM, SamplingParams
import torch
import subprocess as sp
# Read CSV
filename = "constraints.csv"
auto_gen_eval = pd.read_csv("/home/rbheemreddy_umass_edu/vllm_trial1/constraints.csv")
# Contraint list
Constraints = [
"The protagonist must visit an island.\n",
"The protagonist must find a crown.\n",
"The protagonist must discover a mirror.\n",
 "The protagonist must witness a circus.\n",
"The protagonist must explore a garden.\n",
"The protagonist must visit a library.\n",
 "The protagonist must visit a castle.\n",
 "The protagonist must solve a mystery.\n",
"The protagonist must listen to music.\n",
"The protagonist must paint a picture.\n",
 "The protagonist must attend a festival.\n",
 "The protagonist must climb a mountain."]
# Prompts
prompt1 = ["Act as an experienced writer, write a story about daily life in 500 words."]
final_constraints=""
final_constraints = ""
numberOfConstraints = 3
# Model Definition
model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
max_tokens = 8192
sampling_params = SamplingParams(max_tokens= max_tokens, temperature=0.8, top_p=0.95)
# Create an LLM.
llm = LLM(model=model_name, dtype=torch.float16) #, gpu_memory_utilization=0.99)

def addNewStory(df, llm, numberOfConstraints):
    # First prompt
    output1 = llm.generate(prompt1, sampling_params)
    final_constraints = ""

    for output in output1:
        # prompt = output.prompt
        generated_text = output.outputs[0].text
    # final prompt formation
    for i in range(numberOfConstraints):
        final_constraints += random.choice(Constraints)
    prompt2_start = f"Now modify the existing story to accommodate the following constraints: {final_constraints} into the LLM generated story and come up with a new story: "
    final_prompt = [f"""User: "  {prompt1[0]}" \n LLM generated story: " {generated_text} " \n User Instruction: " {prompt2_start} """]
    # second prompt
    output2 = llm.generate(final_prompt, sampling_params)
    for output in output2:
        prompt = output.prompt
        final_generated_story = output.outputs[0].text
    df.loc[len(df.index)] = [numberOfConstraints, final_prompt, final_generated_story, "", "", "", "", "", "", ""]
    df.to_csv("/home/rbheemreddy_umass_edu/vllm_trial1/output.csv")


for i in range(5):
    addNewStory(auto_gen_eval, llm, i)
    break