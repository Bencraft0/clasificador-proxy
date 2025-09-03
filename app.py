from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

# URL de la API de Hugging Face
HF_SPACE_URL = "https://api-inference.huggingface.co/models/Bencraft/clasificador-residuo-api"
HF_API_TOKEN = "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"  # pegás acá tu token

@app.route("/predict", methods=["POST"])
def proxy_predict():
    try:
        data = request.json
        base64_img = data.get("data")[0]  # viene del frontend

        # Llamar a Hugging Face Inference API
        resp = requests.post(
            HF_SPACE_URL,
            headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
            json={"inputs": base64_img},
            timeout=60
        )
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return "Proxy corriendo 🚀"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


