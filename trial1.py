from vllm import LLM, SamplingParams
import torch
import subprocess as sp
import os
import json


# This code will return free GPU memory in MegaBytes for each GPU:

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

# Sample prompts.
prompts = [
#     """Keywords:
# Spaceship, Island, Crown, Dragon, Mirror, Circus, Garden, Library, Time-travel, Castle, Detective, Music, Painting, Festival, Mountain.
# Generate constraints based on these keywords""" 

"""Prompt: Act as an experienced writer. Write a long story with the following constraints:

1. The story must take place on a spaceship.\n
2. The protagonist must visit an island.\n
3. The protagonist must find a crown.\n
4. The protagonist must encounter a dragon.\n
5. The protagonist must discover a mirror.\n
6. The protagonist must witness a circus.\n
7. The protagonist must explore a garden.\n
8. The protagonist must visit a library.\n
9. The protagonist must have the ability to travel through time.\n
10. The protagonist must visit a castle.\n
11. The protagonist must solve a mystery.\n
12. The protagonist must listen to music.\n
13. The protagonist must paint a picture.\n
14. The protagonist must attend a festival.\n
15. The protagonist must climb a mountain.'
"""
]

model_name = 'mistralai/Mistral-7B-Instruct-v0.1'

# Create a sampling params object.
# https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
# check this py file for all the arguments
max_tokens = 4096
sampling_params = SamplingParams(max_tokens= max_tokens, temperature=0.8, top_p=0.95)

# Create an LLM.
# llm = LLM(model=model_name, dtype=torch.float16, tensor_parallel_size=1) #, gpu_memory_utilization=0.99)
llm = LLM(model=model_name, dtype=torch.float16) #, gpu_memory_utilization=0.99)


# Print GPU memory usage before generating text
print("GPU Memory Usage Before Generating Text:")

print("Free memory in MB", get_gpu_memory())
print("*"*50)
print(torch.cuda.memory_summary())

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
print("*"*20)
print("Prompting first input")

# cot_prompt = [""" Follow instructions step by step:   1. Act as an experienced writer, write a story about daily life in 500 words.
# \n 2. Now include the following constraints into the story: {The protagonist must discover a mirror. The protagonist must visit a library.The protagonist must attend a festival."""]
# outputs = llm.generate(cot_prompt, sampling_params)

outputs = llm.generate(prompts, sampling_params)
print("First prompt output generated")
print("*"*20)

# Print GPU memory usage after generating text
print("GPU Memory Usage After Generating Text:")
print("Free memory in MB", get_gpu_memory())
print("*"*50)
print(torch.cuda.memory_summary())

# llm = LLM(model="facebook/opt-125m")


# Assuming you have already defined `outputs` and each output has a `prompt` and `outputs` attribute

# Open a file in write mode
with open(f"first_generated_story_output_{max_tokens}.txt", "w") as file:
    # Iterate through each output
    for output in outputs:
        # Get prompt and generated text
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        # Print the information
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        
        # Write the information to the file
        file.write(f"Prompt: {prompt!r}, \n Generated text: {generated_text!r}\n")

# Print a message indicating that the data has been saved
print(f"Generated text output has been saved to generated_story_output_{max_tokens}.txt")


print("*"*20)
print("Prompting second input (output of previous prompt)")
print(type(generated_text))


# cot_prompt = [""" Follow instructions step by step:   
# 1. Act as an experienced writer, write a story about daily life in 500 words.
# \n 2. Now include the following constraints into the story: {The protagonist must discover a mirror. The protagonist must visit a library.The protagonist must attend a festival."""]

final_outputs = llm.generate([f"Now modify the story to accommodate the constraints into the story, story: '{generated_text}', constraints: '{constraints}"], sampling_params)

print("second prompt output generated")
print("*"*20)

# Create a list to store results
results = []

with open(f"successive_output_{max_tokens}.txt", "w") as file:
    # Iterate through each output
    for final_output in final_outputs:
        # Get prompt and generated text
        prompt = final_output.prompt
        generated_text = final_output.outputs[0].text
        
        # Print the information
        print(f"Final Prompt: {prompt!r}, Final Generated text: {generated_text!r}")
        
        # Write the information to the file
        file.write(f"Prompt: {prompt!r}, \n Generated text: {generated_text!r}\n")
        
        # Append results to the list
        results.append({"prompt": prompt, "generated_text": generated_text})

# Save results to a JSON file
with open(f"output_results_{max_tokens}.json", "w") as json_file:
    json.dump(results, json_file, indent=2)
