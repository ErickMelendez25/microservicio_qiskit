#!/usr/bin/env python3
"""
main.py - API con FastAPI

Endpoints:
- POST /train            -> Entrena (o reentrena) para una zona: recibe {"zone_id": <int>}
- POST /predict          -> Predice para una zona. Body opcional:
                           { "zone_id": <int>, "payload": {<sensores...>} }
                           Si payload ausente, se obtiene últimas lecturas de la BD.
- GET  /outputs/{path}   -> Archivos estáticos (imágenes/CSV) ubicados en ./outputs
"""

import os
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
from datetime import datetime
import json

import train_qsvc_local as trainer

from dotenv import load_dotenv
import mysql.connector

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

app = FastAPI(title="🌱 Quantum Agriculture API", version="1.0")

# Servir carpeta outputs para imágenes/CSV
if not os.path.exists("outputs"):
    os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción limita a tus dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    zone_id: int

class PredictRequest(BaseModel):
    zone_id: int
    payload: Optional[Dict[str, float]] = None  # si no hay payload, tomamos últimas lecturas desde BD

# ----------------- DB helper -----------------
def conectar_bd():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306))
    )

def obtener_ultimas_lecturas_de_zona(zone_id):
    """
    Devuelve un dict con los últimos valores por sensor para esa zona.
    Asegúrate de adaptar nombres/columnas según tu BD.
    """
    conn = conectar_bd()
    cursor = conn.cursor(dictionary=True)
    query = f"""
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

    # Tomar el último valor por tipo de sensor
    result = {}
    for row in rows:
        key = row["sensor"]
        if key not in result:
            result[key] = {"valor": float(row["valor"]), "fecha_lectura": row["fecha_lectura"]}
    # Normalizar nombres (pH -> ph)
    ren = {"pH": "ph", "nitrogeno": "nitrógeno", "fosforo": "fósforo"}
    normalized = {}
    for k, v in result.items():
        nk = ren.get(k, k)
        normalized[nk] = v
    return normalized

def obtener_max_fecha_lectura(zone_id):
    conn = conectar_bd()
    cursor = conn.cursor()
    query = """
        SELECT MAX(l.fecha_lectura) FROM lecturas_sensor l
        JOIN dispositivos_sensor d ON l.id_dispositivo_sensor = d.id_dispositivo_sensor
        WHERE d.zone_id = %s
    """
    cursor.execute(query, (zone_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

# ----------------- Endpoints -----------------

@app.post("/train")
def train(req: TrainRequest):
    zone_id = req.zone_id
    try:
        out = trainer.train_zone(zone_id)
        # Crear URLs públicas para los archivos
        base = f"/outputs/zone_{zone_id}/"
        files = {
            "model": out["model_file"],
            "scaler": out["scaler_file"],
            "stats_csv": base + os.path.basename(out["stats_file"]),
            "pca_png": base + os.path.basename(out["pca_file"]),
            "cluster_png": base + os.path.basename(out["cluster_file"]),
            "importance_png": base + os.path.basename(out["importance_file"]),
            "last_trained_at": out["last_trained_at"]
        }
        return {"status": "ok", "zone_id": zone_id, "files": files}
    except Exception as e:
        logging.exception("Error en /train")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(req: PredictRequest = Body(...)):
    zone_id = req.zone_id
    payload = req.payload

    try:
        # 1) obtener últimas lecturas si no vienen en payload
        if payload is None or len(payload) == 0:
            latest = obtener_ultimas_lecturas_de_zona(zone_id)
            if not latest:
                raise HTTPException(status_code=400, detail="No hay lecturas para la zona.")
            # convertir a dict de valores simples
            payload = {k: float(v["valor"]) for k, v in latest.items()}

        # 2) comprobar si hay datos nuevos desde el último entrenamiento
        zone_dir = os.path.join("outputs", f"zone_{zone_id}")
        metadata_path = os.path.join(zone_dir, "metadata.json")
        last_db_ts = obtener_max_fecha_lectura(zone_id)
        need_retrain = True
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                meta = json.load(f)
            last_trained_at = meta.get("last_trained_at")
            if last_db_ts is None:
                need_retrain = False
            else:
                # comparar timestamps (convertir ambos a datetime)
                try:
                    db_ts = pd_to_dt(last_db_ts)
                    trained_ts = iso_to_dt(last_trained_at)
                    need_retrain = db_ts > trained_ts
                except Exception:
                    need_retrain = True
        else:
            need_retrain = True

        # 3) re-entrenar si hace falta (síncrono)
        if need_retrain:
            logging.info("Nuevos datos detectados, reentrenando zona %s...", zone_id)
            trainer.train_zone(zone_id)

        # 4) Realizar predicción usando el modelo de la zona
        # Asegurarnos de que el payload tenga todas las FEATURE_COLUMNS
        missing = [c for c in trainer.FEATURE_COLUMNS if c not in payload]
        if missing:
            raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")

        clase_pred, X_scaled, zone_dir = trainer.predict_from_values(zone_id, payload)

        # 5) Interpretación agronómica
        interpretacion = trainer.interpretacion_agronomica(payload)

        # 6) Preparar rutas a imágenes
        base = f"/outputs/zone_{zone_id}/"
        imgs = {
            "pca": base + "superposicion_pca.png",
            "clusters": base + "clustering_emergente.png",
            "importance": base + "importancia_sensores.png",
            "stats": base + "estadisticas_entrenamiento.csv"
        }

        # 7) Respuesta
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

# util helpers
def pd_to_dt(val):
    # val puede ser datetime, str...
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(val)
    except Exception:
        try:
            import dateutil.parser as dp
            return dp.parse(val)
        except Exception:
            return datetime.strptime(str(val), "%Y-%m-%d %H:%M:%S")

def iso_to_dt(val):
    if val is None:
        return datetime.fromtimestamp(0)
    return pd_to_dt(val)

@app.get("/")
def root():
    return {"status": "ok", "message": "Quantum Agriculture API - use /docs to explore endpoints"}

# Ejecución directa
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
