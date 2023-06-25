import requests
import json
#curl -X POST -H "Content-Type: application/json" -d '{"prompt":"hello"}' http://localhost:54983/flask-inference/


request = {
    "prompt": "hello world!",
    "max_gen_len": 64,
    "stop_str": ["asdf", "It"]
}
#make this request above in python using requests library
request = requests.post("http://localhost:54984/flask-inference/", json=request)
response = json.loads(request.text)
print(response)
# print(response['info']['tokens'])
# print(f"Response text (length {len(request.text)}):", response['result'])

