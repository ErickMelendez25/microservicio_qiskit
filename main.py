#!/usr/bin/env python3
"""
main.py - FastAPI API
"""

import os, json, logging
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
import mysql.connector
from dotenv import load_dotenv

import train_qsvc_local as trainer

load_dotenv()
app = FastAPI(title="🌱 Quantum Agriculture API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

if not os.path.exists("outputs"):
    os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

class TrainRequest(BaseModel):
    zone_id: int

def conectar_bd():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306))
    )

def obtener_max_fecha_lectura(zone_id):
    conn = conectar_bd()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT MAX(l.fecha_lectura)
        FROM lecturas_sensor l
        JOIN dispositivos_sensor ds ON l.id_dispositivo_sensor = ds.id_dispositivo_sensor
        JOIN dispositivos d ON ds.id_dispositivo = d.id_dispositivo
        WHERE d.zona_agricola_id = %s
    """, (zone_id,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

@app.post("/train")
def train(req: TrainRequest):
    return trainer.train_zone(req.zone_id)

@app.post("/predict")
def predict(raw_body: dict = Body(...), zone_id: int = Query(...)):
    try:
        payload = raw_body.get("payload", raw_body)
        if not payload:
            raise HTTPException(status_code=400, detail="Falta payload")

        # check re-train
        zone_dir = os.path.join("outputs", f"zone_{zone_id}")
        meta_path = os.path.join(zone_dir, "metadata.json")
        need_retrain = True
        db_ts = obtener_max_fecha_lectura(zone_id)

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            trained_dt = datetime.fromisoformat(meta.get("last_trained_at"))
            db_dt = db_ts if not isinstance(db_ts, str) else datetime.fromisoformat(db_ts)
            need_retrain = db_dt > trained_dt

        if need_retrain:
            trainer.train_zone(zone_id)

        clase_pred = trainer.predict_from_values(zone_id, payload)
        interpretacion, cultivo = trainer.interpretacion_agronomica(payload)

        return {
            "status": "ok",
            "zone_id": zone_id,
            "clase": int(clase_pred),
            "interpretacion": interpretacion,
            "cultivo_recomendado": cultivo,
            "input_used": payload
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "ok", "message": "Quantum Agriculture API - use /docs"}
