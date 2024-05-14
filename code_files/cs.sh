#!/bin/bash

# Set the path to the Python script
PYTHON_SCRIPT="code_files/constraint_satisfaction.py"

# Set the base path for common directories
BASE_PATH="/home/rbheemreddy_umass_edu/vllm_trials"

# Set the specific directories relative to the base path
# FOLDER_PATH="$BASE_PATH/Expansion/direction2"
FOLDER_PATH="$BASE_PATH/Expansion/direction2/olmo_storygen/parsed"
SAVE_FOLDER="$BASE_PATH/Expansion/direction2/olmo_storygen/fullrun_baseeval"
SYSTEM_PROMPT_FILE="$BASE_PATH/code_files/system_prompt2.txt"
API_KEY_FILE="$BASE_PATH/code_files/api_key.txt"

# Set other parameters
DIRECTION="d2"
# MODEL="gpt-4-turbo"
MODEL="gpt-3.5-turbo-0125"
# MODEL="gpt-3.5-turbo"


# python "$PYTHON_SCRIPT" "$FOLDER_PATH" "$SYSTEM_PROMPT_FILE" "$DIRECTION" "$API_KEY_FILE" "$MODEL"

# Run the Python script with the specified command-line arguments
python3 "$PYTHON_SCRIPT" --folder_path "$FOLDER_PATH"  --save_folder "$SAVE_FOLDER" --system_prompt_file "$SYSTEM_PROMPT_FILE" --direction "$DIRECTION" --api_key_file "$API_KEY_FILE" --model "$MODEL"
