#taken from https://github.com/samueldashadrach/llama-flask-with-model/blob/main/flask-app.py

import flask
import string
import random
import os

app = flask.Flask(__name__)


@app.route("/")
def hello():
	print("flask server is running", flush=True)
	return "flask server is running"

@app.route("/flask-inference/", methods = ["POST"])
def flask_inference_no_batching():

	print("received POST request", flush=True)

	req_data = flask.request.json
	prompt = req_data["prompt"]
	print("request data", req_data, flush=True) # for testing
    
	# write to file named "prompt"
	tempname = "".join(random.choices(string.ascii_uppercase, k=20))
	try:
		with open(tempname, "w") as f_prompt_temp:
			f_prompt_temp.write(prompt)
			f_prompt_temp.close()
			os.rename(tempname, "prompt")

	except IOError:
		print("prompt could not be written!!!")

	# wait for file named "result" to be created, then read and destroy "result" file
	result_found = False
	while not result_found:
		try:
			with open("result", "r") as f_result:
				result = f_result.read()
				f_result.close()
				os.remove("result")
				result_found = True
		except IOError:
			pass

	res_data = {
		"result": result
	}

	return flask.jsonify(res_data)

if __name__ == "__main__":
	import argparse 

	parser = argparse.ArgumentParser()
	parser.add_argument("--port", type=int, default=54983)
	args = parser.parse_args()

	app.run(port=args.port, debug=True, host='127.0.0.1')
