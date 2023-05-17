import requests
import json
import time
import numpy as np
import matplotlib.pyplot as plt
## FIRST STAGE, GOING OVER THE SINGLE PROMPT

prompt_text_1 = "Hello my name is Alper."
prompt_text_2 = "Lazy fox do the jump."
#make this request above in python using requests library
print("sending request")
print("length of prompt:", len(prompt_text_1))
assert len(prompt_text_1) > 10

n_calls = 8
started = False

log_probs = []
tokens = []

prev_prompt_text, prompt_text = "", ""
for i in range(8):
    prev_prompt_text = prompt_text
    if i in [0,1,4,5]:
        prompt_text = prompt_text_1
    else:
        prompt_text = prompt_text_2

    if prev_prompt_text != prompt_text:
        prob_prev_pos = 5
    else:
        prob_prev_pos = 5

    print("PROB PREV POS:", prob_prev_pos)

    request = requests.post("http://localhost:54983/flask-inference/", json={"prompt":prompt_text, "prob_mode":True, "prob_prev_pos":prob_prev_pos})
    print("Response code:", request.status_code)
    response = json.loads(request.text)
    # print(response)
    chosen_log_probs = np.array(response['info']['chosen_log_probs'])
    chosen_decode_tokens = np.array(response['info']['chosen_decode_tokens'])

    log_probs.append(chosen_log_probs)
    tokens.append(chosen_decode_tokens)

    for log_prob, token in zip(chosen_log_probs, chosen_decode_tokens):
        print(f"{token}: {log_prob}")
    # print(response['info']['tokens'])

