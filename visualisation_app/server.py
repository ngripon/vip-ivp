import subprocess
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

app = Flask(__name__)
app._static_folder = "C:/Users/Nathan/Documents/PetProjects/vip_ivp/visualisation_app/static"


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)


@app.route("/api/foo")
def send_text():
    return jsonify({"message": "Hey you, out there in the cold!"})


if __name__ == "__main__":
    subprocess.call(["npm", "run", "build"], shell=True, cwd="web-app")
    app.run(debug=True)
