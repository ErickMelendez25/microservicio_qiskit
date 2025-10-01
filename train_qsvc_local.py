#!/usr/bin/env python3
"""
train_qsvc_local.py

Entrena un QSVC por zona con FidelityQuantumKernel usando un sampler local.
Genera por zona:
 - modelo: outputs/zone_{zone_id}/modelo_qsvc_zone_{zone_id}.joblib
 - scaler: outputs/zone_{zone_id}/scaler_qsvc_zone_{zone_id}.joblib
 - estadísticas CSV
 - gráficos PNG (PCA, clustering, importancia)
 - metadata JSON con last_trained_timestamp
"""

import os
import sys
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from joblib import dump, load
import mysql.connector
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Qiskit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms.classifiers import QSVC
from qiskit.algorithms.state_fidelities import ComputeUncompute

# ---------- Config ----------
TRAIN_SIZE = 300   # ajustar entre 200-500 según datos por zona
RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "temperatura", "humedad", "ph",
    "nitrógeno", "fósforo", "potasio", "conductividad"
]
OUTPUT_DIR = "outputs"
# ----------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ----------- Helpers -----------

def cargar_env():
    load_dotenv()
    logging.info("Variables de entorno cargadas desde .env")

def conectar_bd():
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306))
    )
    return conn

def leer_datos(conn, zone_id=None, limit=TRAIN_SIZE*10):
    """
    Lee lecturas; si zone_id se provee, filtra por zona.
    Se asume que la tabla lecturas_sensor tiene referencia a dispositivos con zona/ubicación.
    Ajusta la consulta si tu esquema es distinto.
    """
    zone_filter = f"AND d.zone_id = {int(zone_id)}" if zone_id is not None else ""
    query = f"""
        SELECT s.tipo_sensor AS tipo_sensor, l.valor, l.fecha_lectura, d.zone_id
        FROM lecturas_sensor l
        JOIN dispositivos_sensor d ON l.id_dispositivo_sensor = d.id_dispositivo_sensor
        JOIN sensores s ON d.id_sensor = s.id_sensor
        WHERE 1=1 {zone_filter}
        ORDER BY l.fecha_lectura ASC
        LIMIT {limit}
    """
    df = pd.read_sql(query, conn)
    logging.info("Registros leídos (zone=%s): %d", str(zone_id), len(df))
    return df

def preparar_dataset(df):
    if df.empty:
        return None, None, None

    df["fecha_lectura"] = pd.to_datetime(df["fecha_lectura"])
    df["fecha_minuto"] = df["fecha_lectura"].dt.floor("1min")

    df_pivot = df.pivot_table(
        index="fecha_minuto",
        columns="tipo_sensor",
        values="valor",
        aggfunc="mean"
    ).reset_index()

    rename_map = {"pH": "ph", "nitrogeno": "nitrógeno", "fosforo": "fósforo"}
    df_pivot = df_pivot.rename(columns=rename_map)

    df_clean = df_pivot.dropna(how="all", subset=FEATURE_COLUMNS).copy()
    # rellenar missing con la media por columna
    df_clean[FEATURE_COLUMNS] = df_clean[FEATURE_COLUMNS].fillna(df_clean[FEATURE_COLUMNS].mean())

    df_clean = df_clean.head(TRAIN_SIZE)

    if df_clean.shape[0] == 0:
        raise ValueError("❌ No hay datos suficientes después del pivot y limpieza.")

    X = df_clean[FEATURE_COLUMNS].astype(float).values
    # etiqueta simple: promedio de sensores -> discretizar (ejemplo: fertility class)
    y = np.floor(np.mean(X, axis=1) / 10).astype(int)

    return X, y, df_clean

def escalar_y_guardar(X, zone_dir):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaler_file = os.path.join(zone_dir, f"scaler_qsvc_zone.joblib")
    dump(scaler, scaler_file)
    logging.info("Scaler guardado en %s", scaler_file)
    return X_scaled, scaler_file

# ----------- Quantum Helpers -----------

def entrenar_qsvc(X_train, y_train):
    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2)
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler)
    qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    model = QSVC(quantum_kernel=qkernel)
    model.fit(X_train, y_train)
    return model, qkernel

# ----------- Output Helpers -----------

def asegurar_dir(path):
    os.makedirs(path, exist_ok=True)

def generar_estadisticas(df, y, filename):
    df_stats = df[FEATURE_COLUMNS].describe().T
    df_stats["clase_media"] = pd.Series(y).mean()
    df_stats.to_csv(filename, index=True)
    logging.info("📊 Estadísticas guardadas en %s", filename)

def graficar_superposicion(X_scaled, y, filename):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="plasma", alpha=0.7)
    plt.colorbar(scatter, label="Clases (colapso esperado)")
    plt.title("🌌 Superposición cuántica (PCA de embeddings)")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    logging.info("🌌 Gráfico de superposición guardado en %s", filename)

def graficar_clustering(X_scaled, filename):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(X_pca)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Clusters emergentes")
    plt.title("🌱 Clustering emergente en embeddings cuánticos")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    logging.info("🌱 Gráfico de clustering guardado en %s", filename)

def graficar_importancia_sensores(X_scaled, y, filename):
    corr = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS).corrwith(pd.Series(y))
    ax = corr.plot(kind="bar", figsize=(8,5))
    ax.set_ylabel("Correlación")
    ax.set_title("📊 Correlación de sensores con clases (colapso)")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logging.info("📊 Gráfico de importancia de sensores guardado en %s", filename)

# -------------- Metadata ----------------

def guardar_metadata(zone_dir, last_data_ts):
    meta = {
        "last_trained_at": last_data_ts.isoformat() if isinstance(last_data_ts, datetime) else str(last_data_ts),
        "trained_on": datetime.utcnow().isoformat()
    }
    with open(os.path.join(zone_dir, "metadata.json"), "w") as f:
        json.dump(meta, f)
    logging.info("Metadata guardada en %s", zone_dir)

def leer_metadata(zone_dir):
    meta_path = os.path.join(zone_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r") as f:
        return json.load(f)

# -------------- Entrenamiento por zona --------------

def train_zone(zone_id):
    """
    Entrena (o reentrena) para una zona específica.
    """
    try:
        cargar_env()
        conn = conectar_bd()
        df = leer_datos(conn, zone_id=zone_id)
        conn.close()

        if df.empty:
            raise ValueError("No hay datos para la zona solicitada.")

        X, y, df_clean = preparar_dataset(df)
        X_scaled, scaler_file = escalar_y_guardar(X, prepare_zone_dir(zone_id))

        model, qkernel = entrenar_qsvc(X_scaled, y)

        zone_dir = prepare_zone_dir(zone_id)
        asegurar_dir(zone_dir)

        model_file = os.path.join(zone_dir, f"modelo_qsvc_zone.joblib")
        dump(model, model_file)
        logging.info("✅ Modelo guardado en %s", model_file)

        # Métricas y gráficos
        y_pred = model.predict(X_scaled)
        acc = accuracy_score(y, y_pred)
        logging.info("🎯 Accuracy del modelo (zone %s): %.2f%%", zone_id, acc*100)
        logging.info("\n" + classification_report(y, y_pred))

        # Guardar outputs
        stats_file = os.path.join(zone_dir, "estadisticas_entrenamiento.csv")
        pca_file = os.path.join(zone_dir, "superposicion_pca.png")
        cluster_file = os.path.join(zone_dir, "clustering_emergente.png")
        importancia_file = os.path.join(zone_dir, "importancia_sensores.png")

        generar_estadisticas(df_clean, y, stats_file)
        graficar_superposicion(X_scaled, y, pca_file)
        graficar_clustering(X_scaled, cluster_file)
        graficar_importancia_sensores(X_scaled, y, importancia_file)

        # metadata: usamos la última fecha de lectura incluida
        last_ts = df["fecha_lectura"].max()
        if isinstance(last_ts, str):
            last_ts = pd.to_datetime(last_ts)
        guardar_metadata(zone_dir, last_ts)

        return {
            "model_file": model_file,
            "scaler_file": scaler_file,
            "stats_file": stats_file,
            "pca_file": pca_file,
            "cluster_file": cluster_file,
            "importance_file": importancia_file,
            "last_trained_at": last_ts.isoformat() if last_ts is not None else None
        }

    except Exception as e:
        logging.exception("Error en train_zone: %s", str(e))
        raise

def prepare_zone_dir(zone_id):
    zone_dir = os.path.join(OUTPUT_DIR, f"zone_{zone_id}")
    asegurar_dir(zone_dir)
    return zone_dir

# -------------- Utilidades de inferencia ----------------

def load_model_for_zone(zone_id):
    zone_dir = prepare_zone_dir(zone_id)
    model_path = os.path.join(zone_dir, "modelo_qsvc_zone.joblib")
    scaler_path = os.path.join(zone_dir, "scaler_qsvc_zone.joblib")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Modelo o scaler no encontrado para la zona. Entrena primero.")
    model = load(model_path)
    scaler = load(scaler_path)
    return model, scaler, zone_dir

def predict_from_values(zone_id, values_dict):
    """
    values_dict: mapa con las mismas FEATURES en el mismo orden
    """
    X = np.array([[float(values_dict[col]) for col in FEATURE_COLUMNS]], dtype=float)
    model, scaler, zone_dir = load_model_for_zone(zone_id)
    X_scaled = scaler.transform(X)
    pred = int(model.predict(X_scaled)[0])
    return pred, X_scaled, zone_dir

# -------------- Reglas agrícolas sencillas --------------

def interpretacion_agronomica(values_dict):
    """
    Regla heurística para:
     - fertilidad
     - riego necesario (sí/no) y hora recomendada
     - recomendaciones de fertilizante
    """
    temp = float(values_dict.get("temperatura", np.nan))
    hum = float(values_dict.get("humedad", np.nan))
    ph = float(values_dict.get("ph", np.nan))
    nitr = float(values_dict.get("nitrógeno", np.nan))
    fosf = float(values_dict.get("fósforo", np.nan))
    pot = float(values_dict.get("potasio", np.nan))

    lines = []

    # Riego
    if not np.isnan(hum):
        if hum < 40:
            lines.append("💧 Riego recomendado: Sí (humedad baja).")
            # recomendar hora: si temp alta => temprano/temprano y tarde
            if temp >= 25:
                lines.append("⏰ Hora recomendada: temprano en la mañana (antes de 9am) o al atardecer.")
            else:
                lines.append("⏰ Hora recomendada: mañana o tarde según clima.")
        elif hum < 55:
            lines.append("💧 Riego recomendado: Monitorear (humedad moderada).")
        else:
            lines.append("💧 Riego recomendado: No necesario ahora (humedad adecuada).")
    else:
        lines.append("💧 Riego: dato de humedad no disponible.")

    # pH
    if not np.isnan(ph):
        if ph < 5.5:
            lines.append("🧪 pH ácido, puede necesitar calificación antes de sembrar.")
        elif ph > 7.5:
            lines.append("🧪 pH alcalino, estudiar enmiendas específicas.")
        else:
            lines.append("🧪 pH dentro de rango óptimo para la mayoría de cultivos.")
    else:
        lines.append("🧪 pH: dato no disponible.")

    # Nutrientes
    nutrient_avg = np.nanmean([nitr, fosf, pot])
    if np.isnan(nutrient_avg):
        lines.append("🌾 Nutrientes: datos incompletos.")
    else:
        if nutrient_avg < 10:
            lines.append("🌾 Nivel de nutrientes bajo — se recomienda fertilización.")
        elif nutrient_avg < 25:
            lines.append("🌾 Nivel de nutrientes moderado — fertilización leve si es cultivo exigente.")
        else:
            lines.append("🌾 Nivel de nutrientes adecuado.")

    # Recomendación general según combinación
    if (not np.isnan(hum) and hum < 35) or (not np.isnan(nutrient_avg) and nutrient_avg < 10):
        lines.append("✅ Recomendado: Preparar riego/fertilización antes de sembrar.")
    else:
        lines.append("✅ Recomendado: Condiciones aceptables para sembrar si el cultivo es resistente.")

    return "\n".join(lines)

# -------------- MAIN (util para pruebas) ----------------

if __name__ == "__main__":
    # Entrena zona ejemplo (si tecleas zone id)
    if len(sys.argv) >= 2:
        zid = sys.argv[1]
        out = train_zone(zid)
        print("Entrenamiento completado:", out)
    else:
        print("Uso: python train_qsvc_local.py <zone_id>")
