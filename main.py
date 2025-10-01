#!/usr/bin/env python3
"""
main.py - API con FastAPI

Endpoints:
- POST /train        -> Entrena (o reentrena) una zona: {"zone_id": <int>}
- POST /predict      -> Predice para una zona. Acepta:
                       1) {"zone_id": <int>, "payload": {<features>}}
                       2) {"payload": {<features>}} + query param ?zone_id=<int>
                       3) si payload ausente, toma últimas lecturas desde BD para zone_id
- Static files: /outputs/... (imágenes/CSV generados)
"""

import os
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Body, Query
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

# CORS - en prod restringe allow_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    zone_id: int

# No usamos un Pydantic rígido para predict: permitimos cuerpo flexible
# pero validamos internamente que contenga features o zone_id
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
    """
    Devuelve dict: { 'temperatura': {'valor':..., 'fecha_lectura':...}, ... }
    """
    conn = conectar_bd()
    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT s.tipo_sensor as sensor, l.valor, l.fecha_lectura
        FROM lecturas_sensor l
        JOIN dispositivos_sensor ds ON l.id_dispositivo_sensor = ds.id_dispositivo_sensor
        JOIN sensores s ON ds.id_sensor = s.id_sensor
        JOIN dispositivos d ON ds.id_dispositivo = d.id_dispositivo
        WHERE d.id_zona = %s
        ORDER BY l.fecha_lectura DESC
    """
    cursor.execute(query, (zone_id,))
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return {}

    result = {}
    for row in rows:
        key = row["sensor"]
        if key not in result:
            result[key] = {"valor": float(row["valor"]), "fecha_lectura": row["fecha_lectura"]}
    ren = {"pH": "ph", "nitrogeno": "nitrógeno", "fosforo": "fósforo"}
    return {ren.get(k, k): v for k, v in result.items()}

def obtener_max_fecha_lectura(zone_id):
    conn = conectar_bd()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT MAX(l.fecha_lectura)
        FROM lecturas_sensor l
        JOIN dispositivos_sensor ds ON l.id_dispositivo_sensor = ds.id_dispositivo_sensor
        JOIN dispositivos d ON ds.id_dispositivo = d.id_dispositivo
        WHERE d.id_zona = %s
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
            "model": base + os.path.basename(out["model_file"]),
            "scaler": base + os.path.basename(out["scaler_file"]),
            "stats_csv": base + os.path.basename(out["stats_file"]),
            "pca_png": base + os.path.basename(out["pca_file"]),
            "cluster_png": base + os.path.basename(out["cluster_file"]),
            "importance_png": base + os.path.basename(out["importance_file"]),
            "bloch_png": base + (os.path.basename(out["bloch_file"]) if out.get("bloch_file") else None),
            "last_trained_at": out["last_trained_at"]
        }
        return {"status": "ok", "zone_id": req.zone_id, "files": files}
    except Exception as e:
        logging.exception("Error en /train")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(raw_body: Dict[str, Any] = Body(...), zone_id: Optional[int] = Query(None)):
    """
    Entrada flexible:
      - raw_body puede contener {'zone_id': X, 'payload': {...}}
      - o puede ser directamente {'temperatura':..., ...} (payload)
      - también se puede pasar zone_id como query param ?zone_id=40
    """
    try:
        # 1) Determinar zone_id y payload
        payload = None
        body_zone = raw_body.get("zone_id") if isinstance(raw_body, dict) and "zone_id" in raw_body else None
        if body_zone:
            zone_id = int(body_zone)

        if isinstance(raw_body, dict) and "payload" in raw_body:
            payload = raw_body["payload"]
        else:
            # Si no tiene 'payload', es posible que raw_body ya sea el dict de features
            # o tenga otras keys; intentamos usar raw_body como payload si contiene features.
            candidate_keys = set(raw_body.keys())
            features_set = set(trainer.FEATURE_COLUMNS)
            # if raw_body contains at least one of the expected features, treat as payload
            if candidate_keys & features_set:
                # filtrar solo las keys de features
                payload = {k: float(v) for k, v in raw_body.items() if k in features_set}
            else:
                payload = None

        if zone_id is None and ("zone_id" in raw_body and raw_body["zone_id"]):
            zone_id = int(raw_body["zone_id"])

        # 2) Si no se proporcionó payload, tomar últimas lecturas desde BD (necesita zone_id)
        if payload is None or len(payload) == 0:
            if zone_id is None:
                raise HTTPException(status_code=400, detail="Falta zone_id o payload con lecturas.")
            latest = obtener_ultimas_lecturas_de_zona(zone_id)
            if not latest:
                raise HTTPException(status_code=400, detail="No hay lecturas para la zona.")
            payload = {k: float(v["valor"]) for k, v in latest.items()}

        # 3) zone_id obligatorio para usar modelo por zona
        if zone_id is None:
            raise HTTPException(status_code=400, detail="Falta zone_id. Pasa zone_id como query param o en el body.")

        # 4) Comprobar que payload incluye todas las FEATURES (puede completarse con NaN? mejor exigir)
        missing = [c for c in trainer.FEATURE_COLUMNS if c not in payload]
        if missing:
            raise HTTPException(status_code=400, detail=f"Faltan columnas en payload: {missing}")

        # 5) Verificar si hace falta reentrenar (metadata vs DB)
        zone_dir = os.path.join("outputs", f"zone_{zone_id}")
        metadata_path = os.path.join(zone_dir, "metadata.json")
        need_retrain = True
        db_ts = obtener_max_fecha_lectura(zone_id)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                meta = json.load(f)
            last_trained_at = meta.get("last_trained_at")
            try:
                # convertir a datetime y comparar
                trained_dt = datetime.fromisoformat(last_trained_at)
                if db_ts is None:
                    need_retrain = False
                else:
                    # db_ts puede venir como datetime o str
                    if isinstance(db_ts, (str,)):
                        db_dt = datetime.fromisoformat(db_ts)
                    else:
                        db_dt = db_ts
                    need_retrain = db_dt > trained_dt
            except Exception:
                need_retrain = True
        else:
            need_retrain = True

        if need_retrain:
            logging.info("Nuevos datos detectados o no existe modelo: (re)entrenando zona %s ...", zone_id)
            trainer.train_zone(zone_id)

        # 6) Predecir
        clase_pred, X_scaled, zone_dir = trainer.predict_from_values(zone_id, payload)

        # 7) Interpretación
        interpretacion = trainer.interpretacion_agronomica(payload)

        # 8) Rutas a imágenes
        base = f"/outputs/zone_{zone_id}/"
        imgs = {
            "pca": base + "superposicion_pca.png",
            "clusters": base + "clustering_emergente.png",
            "importance": base + "importancia_sensores.png",
            "stats": base + "estadisticas_entrenamiento.csv",
            "bloch": base + "bloch_superposicion.png"
        }

        return {
            "status": "ok",
            "zone_id": zone_id,
            "clase": int(clase_pred),
            "interpretacion": interpretacion,
            "imagenes": imgs,
            "input_used": payload
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
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
