from flask import Flask, request, jsonify
import openai
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://bencraft0.github.io"])  # tu dashboard

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "data" not in data:
            return jsonify({"error": "No se recibió imagen"}), 400

        base64_img = data["data"][0]

        prompt = (
            "Eres un clasificador de residuos. Siempre devuelve una de estas categorías: "
            "plástico, papel, cartón, aluminio, tetra pak, material peligroso. "
            "No uses otro ni desconocido. "
            "Si no estás seguro, elige la categoría que más se parezca. "
            "Opcionalmente, agrega una subetiqueta describiendo el objeto (botella, lata, caja...)."
        )

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": [
                    {"type": "input_text", "text": "Clasifica el residuo en la foto"},
                    {"type": "input_image", "image_url": base64_img}
                ]}
            ]
        )

        label_text = response.choices[0].message.content.strip().lower()

        # Intentamos separar categoría y subetiqueta si el modelo lo devuelve como JSON simple
        import json
        try:
            label_json = json.loads(label_text)
            categoria = label_json.get("categoria", "plástico")
            subcategoria = label_json.get("subcategoria", "")
        except:
            # Si no es JSON, usamos toda la respuesta como categoría y subcategoria vacía
            categoria = label_text.split()[0] if label_text else "plástico"
            subcategoria = ""

        return jsonify({"categoria": categoria, "subcategoria": subcategoria})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

