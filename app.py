from flask import Flask, request, jsonify
import openai
import os
from flask_cors import CORS

app = Flask(__name__)

# ⚠️ Solo permitir CORS desde tu dashboard
CORS(app, origins=["https://tuusuario.github.io"])  # <-- reemplazá con tu URL real

# API Key de OpenAI desde variable de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "data" not in data:
            return jsonify({"error": "No se recibió imagen"}), 400

        base64_img = data["data"][0]

        # Llamada al modelo GPT para clasificación
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Eres un clasificador de residuos. Devuelve solo una etiqueta: plástico, papel, cartón, aluminio, tetra pak, material peligroso, otro."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Clasifica el residuo en la foto"},
                        {"type": "input_image", "image_url": base64_img}
                    ]
                }
            ]
        )

        # Extraer la etiqueta
        label = response.choices[0].message.content.strip().lower()
        return jsonify({"label": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # ⚠️ En producción usá gunicorn en Render
    app.run(host="0.0.0.0", port=5000, debug=True)
