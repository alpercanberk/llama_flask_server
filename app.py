#taken from https://github.com/samueldashadrach/llama-flask-with-model/blob/main/flask-app.py

import flask
import string
import random
import os
import time
import json
import pathlib
import yaml

app = flask.Flask(__name__)

yaml_file_path = pathlib.Path(__file__).parent.absolute() / "default_request_settings.yaml"
with open(yaml_file_path, "r") as f:
	default_request_settings = yaml.load(f, Loader=yaml.FullLoader)
app.config["default_request_settings"] = default_request_settings

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

	print("[Server] Received POST request", flush=True)

	req_data = flask.request.json
	default_settings = app.config["default_request_settings"]

	for key, value in req_data.items():
		if key not in default_settings:
			return flask.jsonify({"error": f"Key '{key}' is not recognized"}), 400
		if not isinstance(value, type(default_settings[key])):
			return flask.jsonify({"error": f"Expected type {type(default_settings[key])} for key '{key}', but got {type(value)}"}), 400
		
	#fill default settings with values from req_data
	req_filled_in = {}
	for key, value in default_settings.items():
		if key in req_data:
			req_filled_in[key] = req_data[key]
		else:
			req_filled_in[key] = value

	# write to file named "prompt"
	tempname = "".join(random.choices(string.ascii_uppercase, k=20))
	try:
		with open(tempname, "w") as f_prompt_temp:
			f_prompt_temp.write(json.dumps(req_filled_in))
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
	
	print("[Server] sending back result")

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
