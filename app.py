from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

HF_URL = "https://Bencraft-clasificador-residuo-api.hf.space/run/predict"

@app.route("/proxy", methods=["POST"])
def proxy():
    data = request.json
    try:
        res = requests.post(HF_URL, json={"data": data["data"]})
        res.raise_for_status()
        return jsonify(res.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Proxy corriendo 🚀"
