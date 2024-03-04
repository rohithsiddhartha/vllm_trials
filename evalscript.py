import pandas as pd
from vllm import LLM, SamplingParams
import torch

def pairwise_eval(story1, story2, gen_prompt1, gen_prompt2):
    # Prompts
    system_prompt = """You are an English writing expert and you can compare and evaluate story essays on these following metrics -

    1. Grammer: Which story has better grammer with respect to the given instruction and constraints.
    2. Coherence: Which story has a better flow and coherence with respect to the given instrction and constraints. 
    3. Likability: Which story would be more enjoyable to read?
    
    You will be given the story Instruction, Constraints and two Stories - Story A and Story B. Specify which story you prefer for each metric by responding with just the letter "A" or "B" followed by a hyphen and reasoning for your preference.
        
    Here's an example - 
    1. Grammer: A - Story A follows the grammatical structure and vibrant tone that is specified by the constraints whereas Story B tends to deviate from the specified structure.
    2. Coherence: B - The writing of Story B flows better, keeping the reader gripped to the plot, while sticking to the main topic and constraints.
    3. Likability: B - Story B has better readability and is more creative inspite of the constraints, making it a lot more of an enjoyable read.

    The above example is for your reference. You can give more detailed reasoning with respect to the constraints if it's appropriate. 

    """
    prompt0 = f"""Instruction: 
    {instruction}

    Constraints: 
    {constraints}

    Story A: 
    {story1}

    Story B:
    {story2}
    """

    prompt1 = [f"""Story 1: {story1}
    Story 2: {story2}
    Evaluate which one is better, only give preference. On the following:
    Grammaticality: How grammatically correct is the text of the story fragment?
    Cohesiveness: How well do the sentences in the story fragment fit together?
    Likability: How enjoyable do you find the story fragment?
    Explain your reasoning and always give a preference, dont give equal ranking"""]
    prompt2 = [f"""Relevance: Now read the PROMPT based on which the story fragment was written.
    Prompt for Story 1: {gen_prompt1}
    Prompt for Story 2: {gen_prompt2}
    How relevant is the story fragment to the prompt?
    Explain your reasoning and always give a preference, dont give equal ranking. Out the reasoning and preference into a key value pair within the string
    with keys as reasoning, preferred story  and with values explanation behind the reasoning, preferred story (first or second)"""]
    # Model Definition
    model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    max_tokens = 4096
    sampling_params = SamplingParams(max_tokens= max_tokens, temperature=0.8, top_p=0.95)
    # Create an LLM.
    llm = LLM(model=model_name, dtype=torch.float16) #, gpu_memory_utilization=0.99)
    output1 = llm.generate(prompt1, sampling_params)
    for output in output1:
        generated_text = output.outputs[0].text
    final_prompt = [f"User: {prompt1}. LLM generated text: {generated_text}. User: {prompt2}"]
    output2 = llm.generate(final_prompt, sampling_params)
    for output in output2:
        # prompt = output.prompt
        generated_text_final = output.outputs[0].text
    return [1, 0, 1, 1, generated_text_final]


df = pd.read_csv('/home/rbheemreddy_umass_edu/vllm_trials/4096listnew.csv')
print("Input csv", df.head())
for category in ["Creativity", "Engagement", "Originality", "Impact"]:
    if category not in df.columns:
        df[category] = 0
df["Evaluation"]= ""


for i in range(len(df)):
    for j in range(i + 1, len(df)):
        story1, story2 = df.iloc[i], df.iloc[j]
        # Evaluate the stories
        results = pairwise_eval(story1['FinalGeneratedStory'], story2['FinalGeneratedStory'], story1['Final_Prompt'], story2["Final_Prompt"])
        # Update the win counts
        for k, category in enumerate(["Creativity", "Engagement", "Originality", "Impact"]):
            if results[k] == 1:
                df.loc[i, category] += 1
            elif results[k] == -1:
                df.loc[j, category] += 1
        df.loc[i,"Evaluation" ] = results[-1]

# story1, story2 = df.iloc[0], df.iloc[1]
# # Evaluate the stories
# results = pairwise_eval(story1['FinalGeneratedStory'], story2['FinalGeneratedStory'], story1['Final_Prompt'], story2["Final_Prompt"])
# # Update the win counts
# for k, category in enumerate(["Creativity", "Engagement", "Originality", "Impact"]):
#     if results[k] == 1:
#         df.loc[0, category] += 1
#     elif results[k] == -1:
#         df.loc[1, category] += 1
# df.loc[0,"Evaluation" ] = results[-1]

print(df.head())
# Save the updated DataFrame to a new CSV file
df.to_csv('updated_stories.csv', index=False)