import requests
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List
## FIRST STAGE, GOING OVER THE SINGLE PROMPT

prompt_text = open("example_plans/incomplete_plan.txt", "r").read()
#make this request above in python using requests library
print("sending request")
print("length of prompt:", len(prompt_text))

started = False

log_probs = []
tokens = []

def extract_plan_begin_idx(sequence_tokens : List[str], plan_start_str : str = "[PLAN]") -> int:
    for i in range(len(sequence_tokens), 0, -1):
        if plan_start_str in "".join(sequence_tokens[i:]):
            return i

n_calls = 2
prob_prev_pos = 0
for i in range(n_calls):

    print("PROB PREV POS:", prob_prev_pos)

    request = requests.post("http://localhost:54983/flask-inference/", json={"prompt":prompt_text, "prob_mode":i==1, "prob_prev_pos":prob_prev_pos})
    print("Response code:", request.status_code)
    response = json.loads(request.text)

    if i == 0:
        print(response['result'])
    else:
        print(response)
        break

    prompt_text = response['result']
    prob_prev_pos = extract_plan_begin_idx(response['info']['chosen_decode_tokens'])
    
    

    