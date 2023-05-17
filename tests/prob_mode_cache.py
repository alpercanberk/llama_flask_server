import requests
import json
import time
import numpy as np
import matplotlib.pyplot as plt
## FIRST STAGE, GOING OVER THE SINGLE PROMPT

prompt_text = "Hello my name is Alper and I'm a computer science student at Columbia University."
#make this request above in python using requests library
print("sending request")
print("length of prompt:", len(prompt_text))
assert len(prompt_text) > 10

n_calls = 8
started = False

log_probs = []
tokens = []

for i in range(n_calls):
    prob_prev_pos = i

    print("PROB PREV POS:", prob_prev_pos)

    request = requests.post("http://localhost:54983/flask-inference/", json={"prompt":prompt_text, "prob_mode":True, "prob_prev_pos":prob_prev_pos})
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

all_log_probs_arr = np.zeros((len(log_probs), len(log_probs[0])))
for i, log_prob_arr in enumerate(log_probs):
    all_log_probs_arr[i, -len(log_prob_arr):] = log_prob_arr

differences = np.ones((len(all_log_probs_arr), len(all_log_probs_arr))) * np.inf
for i in range(len(all_log_probs_arr)):
    for j in range(i):
        differences[i,j] = np.mean(np.abs(all_log_probs_arr[i, i:] - all_log_probs_arr[j, i:]))
    
print(differences)
plt.imshow(differences)
#add colorbar
cbar = plt.colorbar()
cbar.set_label('Differences')
plt.show()

fig_path = '/local/crv/acanberk/llama_flask_server/tests/figs'
plt.savefig(f"{fig_path}/prob_mode_cache.png")

plt.close()
