from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# ⚙️ Initialisation de l'application
app = FastAPI(
    title="AlertEpilepsy API",
    description="API pour prédire les crises d'épilepsie à partir de segments EEG",
    version="1.0.0",
)

# 🔓 Autoriser CORS pour accès Flutter/app mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tu peux restreindre ici à ton domaine mobile plus tard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📦 Chargement du modèle Keras (.h5)
try:
    model = tf.keras.models.load_model("equi1.h5")
    print("✅ Modèle chargé avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    raise e

# 🧠 Fonction de normalisation
def normalize(X):
    mean = np.mean(X, axis=(1, 2), keepdims=True)
    std = np.std(X, axis=(1, 2), keepdims=True) + 1e-6
    return (X - mean) / std

# 📤 Modèle Pydantic pour le POST
class EEGSegment(BaseModel):
    data: list  # 2D list: [channels][samples] soit (18,2048)

# 🌐 Route GET pour test
@app.get("/")
def root():
    return {"message": "✅ API opérationnelle", "status": "ok"}

# 🌐 Route POST pour prédiction
@app.post("/predict")
def predict_segment(segment: EEGSegment):
    try:
        arr = np.array(segment.data)

        # Vérification de la forme
        if arr.shape != (18, 2048):
            raise ValueError(f"Segment invalide : attendu (18,2048), reçu {arr.shape}")

        # Préparation du batch
        arr = normalize(arr.astype(np.float32).reshape(1, 18, 2048))  # (1, 18, 2048)
        arr = arr.transpose(0, 2, 1)  # (1, 2048, 18) pour Conv1D

        # Prédiction
        prob = float(model.predict(arr, verbose=0)[0][0])
        label = "preictal" if prob >= 0.5 else "interictal"

        print(f"[API] ✅ Prediction: {label} | Proba: {prob:.4f}")

        return {
            "prediction": label,
            "probability": round(prob, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
