#!/usr/bin/env python3
"""
main.py - API con FastAPI

Endpoints:
- POST /train
- POST /predict
- GET  /outputs/{path}
"""

import os
import logging
from typing import Optional, Dict
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
import json

import train_qsvc_local as trainer
from dotenv import load_dotenv
import mysql.connector

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

app = FastAPI(title="🌱 Quantum Agriculture API", version="1.0")

# Servir carpeta outputs
if not os.path.exists("outputs"):
    os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# CORS libre (ajusta dominios en producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    zone_id: int

class PredictRequest(BaseModel):
    zone_id: int
    payload: Optional[Dict[str, float]] = None

# ----------------- DB -----------------
def conectar_bd():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306))
    )

def obtener_ultimas_lecturas_de_zona(zone_id):
    conn = conectar_bd()
    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT s.tipo_sensor as sensor, l.valor, l.fecha_lectura
        FROM lecturas_sensor l
        JOIN dispositivos_sensor d ON l.id_dispositivo_sensor = d.id_dispositivo_sensor
        JOIN sensores s ON d.id_sensor = s.id_sensor
        WHERE d.zone_id = %s
        ORDER BY l.fecha_lectura DESC
    """
    cursor.execute(query, (zone_id,))
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return {}

    result = {}
    for row in rows:
        if row["sensor"] not in result:
            result[row["sensor"]] = {"valor": float(row["valor"]), "fecha_lectura": row["fecha_lectura"]}
    ren = {"pH": "ph", "nitrogeno": "nitrógeno", "fosforo": "fósforo"}
    return {ren.get(k, k): v for k, v in result.items()}

def obtener_max_fecha_lectura(zone_id):
    conn = conectar_bd()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT MAX(l.fecha_lectura) 
        FROM lecturas_sensor l
        JOIN dispositivos_sensor d ON l.id_dispositivo_sensor = d.id_dispositivo_sensor
        WHERE d.zone_id = %s
    """, (zone_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

# ----------------- Endpoints -----------------
@app.post("/train")
def train(req: TrainRequest):
    try:
        out = trainer.train_zone(req.zone_id)
        base = f"/outputs/zone_{req.zone_id}/"
        files = {
            "model": base + "modelo_qsvc_zone.joblib",
            "scaler": base + "scaler_qsvc_zone.joblib",
            "stats_csv": base + "estadisticas_entrenamiento.csv",
            "pca_png": base + "superposicion_pca.png",
            "cluster_png": base + "clustering_emergente.png",
            "importance_png": base + "importancia_sensores.png",
            "last_trained_at": out["last_trained_at"]
        }
        return {"status": "ok", "zone_id": req.zone_id, "files": files}
    except Exception as e:
        logging.exception("Error en /train")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(req: PredictRequest = Body(...)):
    try:
        zone_id = req.zone_id
        payload = req.payload

        if not payload:
            latest = obtener_ultimas_lecturas_de_zona(zone_id)
            if not latest:
                raise HTTPException(status_code=400, detail="No hay lecturas para la zona.")
            payload = {k: float(v["valor"]) for k, v in latest.items()}

        # Asegurar todas las features
        missing = [c for c in trainer.FEATURE_COLUMNS if c not in payload]
        if missing:
            raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")

        clase_pred, _, zone_dir = trainer.predict_from_values(zone_id, payload)
        interpretacion = trainer.interpretacion_agronomica(payload)

        base = f"/outputs/zone_{zone_id}/"
        imgs = {
            "pca": base + "superposicion_pca.png",
            "clusters": base + "clustering_emergente.png",
            "importance": base + "importancia_sensores.png",
            "stats": base + "estadisticas_entrenamiento.csv"
        }

        return {
            "status": "ok",
            "zone_id": zone_id,
            "clase": int(clase_pred),
            "interpretacion": interpretacion,
            "imagenes": imgs
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error en /predict")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "ok", "message": "Quantum Agriculture API - use /docs"}

# ----------------- EntryPoint -----------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))  # ✅ Railway usa $PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port)
