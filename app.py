from flask import Flask, request, jsonify
import openai
import os
import base64
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://bencraft0.github.io"])  # reemplaza con tu URL real

openai.api_key = os.getenv("OPENAI_API_KEY")

# Lista de categorías
CATEGORIES = ["plástico", "papel", "cartón", "aluminio", "tetra pak", "material peligroso"]

# Precomputamos los embeddings de cada categoría
category_embeddings = {}
for cat in CATEGORIES:
    res = openai.embeddings.create(
        input=cat,
        model="text-embedding-3-small"
    )
    category_embeddings[cat] = np.array(res.data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "data" not in data:
            return jsonify({"error": "No se recibió imagen"}), 400

        base64_img = data["data"][0]

        # Convertimos la imagen base64 a URL para el API
        # Si la API acepta directamente base64, la enviamos así
        res = openai.embeddings.create(
            input=[base64_img],
            model="image-embedding-3-small"
        )
        img_embedding = np.array(res.data[0].embedding)

        # Comparamos con las categorías
        best_cat = None
        best_score = -1
        for cat, emb in category_embeddings.items():
            score = cosine_similarity(img_embedding, emb)
            if score > best_score:
                best_score = score
                best_cat = cat

        # Devolvemos categoría
        return jsonify({"categoria": best_cat, "score": float(best_score)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
