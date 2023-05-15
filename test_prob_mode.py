import requests
import json
import time

#curl -X POST -H "Content-Type: application/json" -d '{"prompt":"hello"}' http://localhost:54983/flask-inference/

prompt_text = open("example_plans/good_plan.txt", "r").read()
#make this request above in python using requests library
print("sending request")
print("length of prompt:", len(prompt_text))




n_calls = 2
started = False
for i in range(n_calls):
    if i == 0:
        prob_prev_pos = 0
    else:
        prob_prev_pos = 400
        if not started:
            start_time = time.time()
            started = True
    request = requests.post("http://localhost:54983/flask-inference/", json={"prompt":prompt_text, "prob_mode":True, "prob_prev_pos":prob_prev_pos})
    print("Response code:", request.status_code)
    response = json.loads(request.text)
    print(f"Log perplexity (length: {len(response['info']['log_perplexity'])}):")
    print(response['info']['log_perplexity'][-5:])
    print(f"Tokens (length: {len(response['info']['tokens'])}):")
    # print(response['info']['tokens'])

print(f"Time taken per call: {(time.time() - start_time)/n_calls}")

