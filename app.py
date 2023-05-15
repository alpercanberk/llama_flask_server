#taken from https://github.com/samueldashadrach/llama-flask-with-model/blob/main/flask-app.py

import flask
import string
import random
import os
import time
import json

app = flask.Flask(__name__)


@app.route("/")
def hello():
	print("flask server is running", flush=True)
	return "flask server is running"

@app.route("/flask-inference/", methods = ["POST"])
def flask_inference_no_batching():

	if os.path.exists("prompt"):
		os.remove("prompt")
	if os.path.exists("result"):
		os.remove("result")

	print("received POST request", flush=True)

	req_data = flask.request.json    
	# write to file named "prompt"
	tempname = "".join(random.choices(string.ascii_uppercase, k=20))
	try:
		with open(tempname, "w") as f_prompt_temp:
			f_prompt_temp.write(json.dumps(req_data))
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
				result = json.loads(result)				
				f_result.close()
				os.remove("result")
				result_found = True
		except IOError:
			time.sleep(0.1)
	
	print("[Server] sending result")

	return flask.jsonify(result)

if __name__ == "__main__":
	import argparse 

	parser = argparse.ArgumentParser()
	parser.add_argument("--port", type=int, default=54983)
	args = parser.parse_args()

	#remove any files named prompt
	if os.path.exists("prompt"):
		os.remove("prompt")

	app.run(port=args.port, debug=True, host='127.0.0.1')
