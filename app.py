# -*- coding: utf-8 -*-
"""
app.py
------
Flask web server that exposes the chatbot as a REST API.
Run with:  python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from chatbot_core import init, ask
import os

app = Flask(__name__, static_folder="static")
CORS(app)  # allows the HTML page to call the API from any origin

# Initialize chatbot once at server startup
print("Starting chatbot initialization...")
init()
print("Server ready.")

# --- Serve the HTML frontend ---
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

# --- Chat API endpoint ---
@app.route("/api/ask", methods=["POST"])
def api_ask():
    data     = request.get_json()
    question = data.get("question", "").strip()
    pdf_filename = data.get("pdf_filename", None)

    if not question:
        return jsonify({"error": "No question provided"}), 400

    result = ask(question, pdf_filename)
    return jsonify(result)

# --- Serve PDFs ---
@app.route("/laws/<path:filename>")
def serve_pdf(filename):
    return send_from_directory("laws", filename)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
