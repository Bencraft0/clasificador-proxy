from flask import Flask, request, jsonify
import openai
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://bencraft0.github.io"])  # Permitir tu dashboard

# API Key de OpenAI desde variable de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "data" not in data:
            return jsonify({"error": "No se recibió imagen"}), 400

        base64_img = data["data"][0]

        # Llamada al modelo multimodal GPT-4o
        response = openai.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Eres un clasificador de residuos. Devuelve solo una etiqueta: plástico, papel, cartón, aluminio, tetra pak, material peligroso, otro."
                        },
                        {
                            "type": "input_image",
                            "image_url": base64_img
                        }
                    ]
                }
            ]
        )

        # Extraer la etiqueta
        label = response.output[0].content[0].text.strip().lower()
        return jsonify({"label": label})

    except Exception as e:
        print("ERROR PREDICT:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
