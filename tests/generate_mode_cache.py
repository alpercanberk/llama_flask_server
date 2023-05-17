import requests
import json
import time
import numpy as np
import matplotlib.pyplot as plt
## FIRST STAGE, GOING OVER THE SINGLE PROMPT

prompt_text = "Hello my name is Alper and"
#make this request above in python using requests library
print("sending request")
print("length of prompt:", len(prompt_text))

started = False

log_probs = []
tokens = []

n_calls = 1
for i in range(n_calls):
    prob_prev_pos = 0

    print("PROB PREV POS:", prob_prev_pos)

    request = requests.post("http://localhost:54983/flask-inference/", json={"prompt":prompt_text, "prob_mode":False, "prob_prev_pos":0})
    print("Response code:", request.status_code)
    response = json.loads(request.text)
    print(response)