from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Replace with your private Hugging Face Space API URL
HF_API_URL = "https://huggingface.co/spaces/Gaurav-T/ai-podcast-summarizer"

@app.route("/query", methods=["POST"])
def query():
    user_input = request.json.get("question")
    response = requests.post(f"{HF_API_URL}/query", json={"question": user_input})
    return jsonify(response.json())

if __name__ == "__main__":
    app.run(debug=True)
