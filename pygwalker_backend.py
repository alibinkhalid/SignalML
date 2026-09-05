import os
from flask import Flask, jsonify, request, Response, send_from_directory
import pandas as pd
import pygwalker as pyg

app = Flask(__name__, static_folder=".", static_url_path="")


@app.after_request
def allow_local_browser(response):
    origin = request.headers.get("Origin")
    if origin and origin.startswith(("http://localhost:", "http://127.0.0.1:")):
        response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


@app.get("/")
def index():
    return send_from_directory(".", "index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/visualize")
def visualize():
    payload = request.get_json(silent=False)
    columns = payload.get("columns", [])
    records = payload.get("rows", [])
    if not columns or not isinstance(records, list):
        return jsonify({"error": "A dataset with columns and rows is required."}), 400

    frame = pd.DataFrame.from_records(records, columns=columns)
    html = pyg.to_html(frame, use_kernel_calc=False)
    return Response(html, mimetype="text/html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
