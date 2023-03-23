import requests

#curl -X POST -H "Content-Type: application/json" -d '{"prompt":"hello"}' http://localhost:54983/flask-inference/

prompt_text = open("test_prompt.txt", "r").read()

#make this request above in python using requests library
for i in range(10):
    print("sending request", i)
    print("length of prompt:", len(prompt_text))
    request = requests.post("http://localhost:54983/flask-inference/", json={"prompt":prompt_text})
    print("Response code:", request.status_code)
    print(f"Response text (length {len(request.text)}):", request.text[len(prompt_text):])

