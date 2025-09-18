from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)

# ⚠️ Importante: poné tu API Key en una variable de entorno
# Ejemplo en Windows PowerShell:
#   setx OPENAI_API_KEY "tu_api_key"
# Ejemplo en Linux/Mac:
#   export OPENAI_API_KEY="tu_api_key"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "data" not in data:
            return jsonify({"error": "No se recibió imagen"}), 400

        base64_img = data["data"][0]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un clasificador de residuos. Devuelve solo una etiqueta: plástico, papel, cartón, aluminio, tetra pak, material peligroso, otro."},
                {"role": "user", "content": [
                    {"type": "input_text", "text": "Clasifica el residuo en la foto"},
                    {"type": "input_image", "image_url": base64_img}
                ]}
            ]
        )

        label = response.choices[0].message.content.strip().lower()
        return jsonify({"label": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # ⚠️ En producción usá un server WSGI (gunicorn, uvicorn, etc.)
    app.run(host="0.0.0.0", port=5000, debug=True)
