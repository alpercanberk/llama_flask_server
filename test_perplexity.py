import requests
import json
import os
from rich.console import Console
from rich.table import Table
from rich.style import Style

console = Console()

# Define styles for the columns
prompt_style = Style(color="cyan")
perplexity_style = Style(color="magenta")

# List of prompts with increasing perplexity
prompts = [
    "hello",
    ";ijl45",
    "tomatos are red",
    "tomatos are blue",
    "The quick brown fox jumps over the lazy dog",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
    "Asdf qwer zxcv poiu lkjh mnbv",
    "4f6s8d v1d6g d0k4s m6j9p 3l8h 7b3v 5c2x 9n8m",
    "To brush my teeth, I must first go to the garage and find a tire",
    "To brush my teeth, I must first go to the kitchen and find a fork",
    "To brush my teeth, I must first go to the bathroom and find my toothbrush",

]

# Make a request to the server for each prompt
results = []
for prompt in prompts:
    data = {"prompt": prompt, "prob_mode":True}
    headers = {"Content-Type": "application/json"}

    response = requests.post("http://localhost:54983/flask-inference/", json=data, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Convert the response to a dictionary
        response_dict = json.loads(response.text)
        # Append the result to the list of results
        results.append((prompt[:100], response_dict['info']['log_perplexity']))
    else:
        results.append((prompt[:100], "Request failed"))


# Set the path to the example plans directory
plan_directory = 'example_plans'

# Get a list of all the plan files in the directory
plan_files = os.listdir(plan_directory)

# Iterate over each plan file and pass it through the model
for plan_file in plan_files:
    # Open the plan file and read its contents
    with open(os.path.join(plan_directory, plan_file), 'r') as f:
        plan = f.read()

    # Make a request to the server with the plan as the prompt
    data = {"prompt": plan, "prob_mode":True}
    headers = {"Content-Type": "application/json"}
    response = requests.post("http://localhost:54983/flask-inference/", json=data, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Convert the response to a dictionary
        response_dict = json.loads(response.text)
        # Append the result to the list of results
        results.append((plan_file, response_dict['info']['log_perplexity']))
    else:
        results.append((plan_file, "Request failed"))

# Create a table to display the results
table = Table(title="Perplexity Results")
table.add_column("Prompt/File", justify="left", style=prompt_style)
table.add_column("Perplexity", justify="right", style=perplexity_style)

for result in results:
    table.add_row(result[0], f"{result[1]:.3f}")

# Print the table to the console
console.print(table)
