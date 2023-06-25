import requests
import json
import time
import numpy as np
import matplotlib.pyplot as plt

def reduce_precision(arr, precision=3):
    return [float(f'{x:0.{precision}f}') for x in arr]

prompt_text = "Hello my name is Alper and I'm a computer science student at Columbia University. I"
prob_prev_pos = 0
request = requests.post("http://localhost:54983/flask-inference/", json={"prompt":prompt_text, 
                        "prob_mode":True, "prob_prev_pos":prob_prev_pos})
response = json.loads(request.text)
# print(response)
print(response.keys())
print("Result: ", response['result'][:200], '...' if len(response['result']) > 200 else '')
info_keys = response['info'].keys()
print(info_keys)
print("=" * 20)
print('prompt text: ', response['info']['prompt_text'])
print("=" * 20)
print('chosen decode tokens: ', response['info']['chosen_decode_tokens'])
print("=" * 20)
print('chosen decode logprobs: ', reduce_precision(response['info']['chosen_log_probs']))



print("\n\n")

prompt_text = "Hello my name is Alper and I'm a computer science student at Columbia University. I"
request = requests.post("http://localhost:54983/flask-inference/", json={"prompt":prompt_text, "prob_mode":False, "prob_prev_pos":0, 'max_gen_len': 128})
response = json.loads(request.text)
# print(response)
print(response.keys())
print("Result: ", response['result'][:200], '...' if len(response['result']) > 200 else '')
info_keys = response['info'].keys()
print(info_keys)
#print prompt text, prompt and generated tokens (first 20), chosen decode tokens (first 20), chosen decode logprobs (first 20)
print("=" * 20)
print('prompt text: ', response['info']['prompt_text'])
print("=" * 20)
print('generated tokens: ', response['info']['generated_tokens'][:20])
print("=" * 20)
print('chosen decode tokens: ', response['info']['chosen_decode_tokens'])
print("=" * 20)