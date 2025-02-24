from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# Your private Hugging Face Space URL (Make sure it's correct)
HF_API_URL = "https://huggingface.co/spaces/Gaurav-T/ai-podcast-summarizer"
HF_API_KEY = os.getenv("HF_API_KEY")

@app.route("/query", methods=["POST"])
def query():
    user_input = request.json.get("question")
    
    if not user_input:
        return jsonify({"error": "No question provided"}), 400

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}  # Use Hugging Face API Key
    response = requests.post(f"{HF_API_URL}/query", json={"question": user_input}, headers=headers)

    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({"error": "Failed to retrieve response"}), response.status_code

if __name__ == "__main__":
    app.run(debug=True)
