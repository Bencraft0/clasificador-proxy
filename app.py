from flask import Flask, request, jsonify
import openai
import os
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app, origins=["https://bencraft0.github.io"])  # reemplazá con tu URL real

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "data" not in data:
            return jsonify({"error": "No se recibió imagen"}), 400

        base64_img = data["data"][0]

        # Prompt con instrucción de ignorar todo lo que no sea residuo
        prompt = (
            "Eres un clasificador de residuos. Ignora todo lo que no sea un residuo. "
            "Solo devuelve una de estas categorías: plástico, papel, cartón, aluminio, tetra pak, material peligroso. "
            "Si no hay residuo claro, devuelve 'ninguno'. "
            "Devuelve un JSON con 'categoria' y 'subcategoria' opcional."
        )

        response = openai.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": base64_img}
                    ]
                }
            ]
        )

        # Intentar extraer JSON del output
        label_data = getattr(response, "output_text", "{}").strip()
        try:
            label_json = json.loads(label_data)
            categoria = label_json.get("categoria", "ninguno")
            subcategoria = label_json.get("subcategoria", "")
        except:
            categoria = "ninguno"
            subcategoria = ""

        return jsonify({"categoria": categoria, "subcategoria": subcategoria})

    except Exception as e:
        print("ERROR PREDICT:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
