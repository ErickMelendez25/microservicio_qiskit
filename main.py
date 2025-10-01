#!/usr/bin/env python3
"""
main.py - API con FastAPI

Expone 2 endpoints:
- /train   -> Entrena el modelo (usa la BD, modo científico/tesis)
- /predict -> Predice fertilidad con nuevas lecturas (modo agrícola)
"""

import logging
import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from fastapi.middleware.cors import CORSMiddleware

# Importamos el script de entrenamiento
import train_qsvc_local as trainer

# ------------------ CONFIG ------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

app = FastAPI(title="🌱 Quantum Agriculture API", version="1.0")

# 🔥 Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 en producción cambia a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ MODELO DE ENTRADA PARA PREDICCIÓN ------------------

class SensorInput(BaseModel):
    temperatura: float
    humedad: float
    ph: float
    nitrógeno: float
    fósforo: float
    potasio: float
    conductividad: float

# ------------------ ENDPOINTS ------------------

@app.post("/train")
def entrenar_modelo():
    """
    Ejecuta el entrenamiento del modelo usando datos de la BD.
    """
    try:
        trainer.main()
        return {
            "status": "✅ Entrenamiento completado",
            "archivos": [
                trainer.MODEL_OUT,
                trainer.SCALER_OUT,
                trainer.STATS_CSV,
                trainer.SUPERPOSICION_PNG,
                trainer.CLUSTER_PNG,
                trainer.IMPORTANCIA_PNG,
            ],
        }
    except Exception as e:
        logging.exception("❌ Error en entrenamiento")
        return {"status": "error", "detalle": str(e)}

@app.post("/predict")
def predecir(input_data: SensorInput):
    """
    Realiza una predicción usando el modelo ya entrenado.
    """
    try:
        modelo = load(trainer.MODEL_OUT)
        scaler = load(trainer.SCALER_OUT)

        X_nuevo = np.array([[getattr(input_data, col) for col in trainer.FEATURE_COLUMNS]], dtype=float)
        X_nuevo_scaled = scaler.transform(X_nuevo)

        clase_pred = int(modelo.predict(X_nuevo_scaled)[0])

        interpretaciones = {
            4: "🌱 Suelo con fertilidad baja, requiere nutrientes.",
            5: "🌾 Suelo con fertilidad media, condiciones aceptables.",
            6: "🌿 Suelo con buena fertilidad, óptimo para cultivos.",
            7: "🌳 Suelo con fertilidad muy alta, posible riesgo de exceso.",
        }
        resultado = interpretaciones.get(clase_pred, "Clase desconocida")

        return {"clase": clase_pred, "interpretacion": resultado}

    except Exception as e:
        logging.exception("❌ Error en predicción")
        return {"status": "error", "detalle": str(e)}

# ------------------ INICIO ------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))  # Railway asigna el puerto
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
