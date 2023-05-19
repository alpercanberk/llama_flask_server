import requests
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
## FIRST STAGE, GOING OVER THE SINGLE PROMPT

prompt_text = "Hello my name is Alper and I'm a "
n_calls = 4
started = False

log_probs = []
tokens = []
prob_prev_pos = 7

for i in range(n_calls):

    #generate random state using i
    g = np.random.RandomState(i)
    n = g.randint(10**3, 10**4)

    prompt_text_final = prompt_text + str(n)

    print("PROB PREV POS:", prob_prev_pos)

    request = requests.post("http://localhost:54983/flask-inference/", json={"prompt":prompt_text_final, "prob_mode":True, "prob_prev_pos":prob_prev_pos})
    print("Response code:", request.status_code)
    response = json.loads(request.text)
    # print(response)
    chosen_log_probs = np.array(response['info']['chosen_log_probs'])
    chosen_decode_tokens = np.array(response['info']['chosen_decode_tokens'])

    log_probs.append(chosen_log_probs)
    tokens.append(chosen_decode_tokens)

    # print(f"Log probs (length: {len(response['info']['chosen_log_probs'])})")
    # print(f"Tokens (length: {len(response['info']['chosen_decode_tokens'])})")
    for log_prob, token in zip(chosen_log_probs, chosen_decode_tokens):
        print(f"{token}: {log_prob}")
    # print(response['info']['tokens'])
