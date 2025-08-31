from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)  # Permite requests desde cualquier origen

# URL de tu Space
HF_SPACE_URL = "https://huggingface.co/spaces/Bencraft/clasificador-residuo-api/run/predict"

@app.route("/predict", methods=["POST"])
def proxy_predict():
    try:
        data = request.get_json()
        # Enviar la request al Space
        resp = requests.post(HF_SPACE_URL, json=data, timeout=30)
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return "Proxy corriendo 🚀"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
