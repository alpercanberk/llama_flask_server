import requests
import json
#curl -X POST -H "Content-Type: application/json" -d '{"prompt":"hello"}' http://localhost:54983/flask-inference/

prompt_text = open("test_prompt.txt", "r").read()

#make this request above in python using requests library
print("length of prompt:", len(prompt_text))
request = requests.post("http://localhost:54983/flask-inference/", json={"prompt":prompt_text})
response = json.loads(request.text)
print(response['info']['tokens'])
print(f"Response text (length {len(request.text)}):", response['result'][len(prompt_text):])

