#!/usr/bin/env python3
"""
train_qsvc_local.py

Entrena un QSVC por zona con FidelityQuantumKernel usando un sampler local.
Genera por zona en ./outputs/zone_{zone_id}/:
 - modelo: modelo_qsvc_zone_{zone_id}.joblib
 - scaler: scaler_qsvc_zone_{zone_id}.joblib
 - estadisticas: estadisticas_entrenamiento.csv
 - imagenes: superposicion_pca.png, clustering_emergente.png, importancia_sensores.png
 - metadata: metadata.json (last_trained_at, trained_on)
 - interpretaciones: interpretaciones.txt

Notas:
 - Ajusta las consultas SQL si tu esquema difiere.
 - Intenta generar una imagen de Bloch multivector si Aer está disponible; si no, continúa sin ella.
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
try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.primitives import Sampler
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.algorithms.classifiers import QSVC
    from qiskit.algorithms.state_fidelities import ComputeUncompute
    QISKIT_AVAILABLE = True
except Exception:
    QISKIT_AVAILABLE = False

# Aer
try:
    from qiskit import Aer, transpile
    from qiskit.visualization import plot_bloch_multivector
    AER_AVAILABLE = True
except Exception:
    AER_AVAILABLE = False

# ---------- Config ----------
TRAIN_SIZE = 50
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
    if zone_id is None:
        query = """
            SELECT s.tipo_sensor AS tipo_sensor, l.valor, l.fecha_lectura
            FROM lecturas_sensor l
            JOIN dispositivos_sensor ds ON l.id_dispositivo_sensor = ds.id_dispositivo_sensor
            JOIN sensores s ON ds.id_sensor = s.id_sensor
            JOIN dispositivos d ON ds.id_dispositivo = d.id_dispositivo
            ORDER BY l.fecha_lectura ASC
            LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(limit,))
    else:
        query = f"""
            SELECT s.tipo_sensor AS tipo_sensor, l.valor, l.fecha_lectura
            FROM lecturas_sensor l
            JOIN dispositivos_sensor ds ON l.id_dispositivo_sensor = ds.id_dispositivo_sensor
            JOIN sensores s ON ds.id_sensor = s.id_sensor
            JOIN dispositivos d ON ds.id_dispositivo = d.id_dispositivo
            WHERE d.zona_agricola_id = %s
            ORDER BY l.fecha_lectura ASC
            LIMIT {limit}
        """
        df = pd.read_sql(query, conn, params=(zone_id,))
    logging.info("Registros leídos (zone=%s): %d", str(zone_id), len(df))
    return df

def preparar_dataset(df):
    if df is None or df.empty:
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
    df_clean[FEATURE_COLUMNS] = df_clean[FEATURE_COLUMNS].fillna(df_clean[FEATURE_COLUMNS].mean())

    df_clean = df_clean.head(TRAIN_SIZE)
    if df_clean.shape[0] == 0:
        raise ValueError("❌ No hay datos suficientes después del pivot y limpieza.")

    X = df_clean[FEATURE_COLUMNS].astype(float).values
    y = np.floor(np.mean(X, axis=1) / 10).astype(int)
    return X, y, df_clean

def escalar_y_guardar(X, zone_dir, zone_id):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaler_file = os.path.join(zone_dir, f"scaler_qsvc_zone_{zone_id}.joblib")
    dump(scaler, scaler_file)
    logging.info("Scaler guardado en %s", scaler_file)
    return X_scaled, scaler_file

# ----------- Quantum Helpers -----------

def entrenar_qsvc(X_train, y_train):
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit no está disponible en el entorno.")

    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2)
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler)
    qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    model = QSVC(quantum_kernel=qkernel)
    model.fit(X_train, y_train)
    return model, qkernel, feature_map

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
    plt.colorbar(scatter, label="Clases")
    plt.title("🌌 Superposición cuántica (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    logging.info("🌌 Gráfico PCA guardado en %s", filename)

def graficar_clustering(X_scaled, filename):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(X_pca)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Clusters")
    plt.title("🌱 Clustering emergente")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    logging.info("🌱 Clustering guardado en %s", filename)

def graficar_importancia_sensores(X_scaled, y, filename):
    corr = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS).corrwith(pd.Series(y))
    ax = corr.plot(kind="bar", figsize=(8,5))
    ax.set_ylabel("Correlación")
    ax.set_title("📊 Importancia de sensores")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    logging.info("📊 Importancia sensores guardada en %s", filename)

def generar_bloch_image(feature_map, sample_vector, filename):
    if not (QISKIT_AVAILABLE and AER_AVAILABLE):
        logging.info("Aer o Qiskit no disponibles.")
        return None
    try:
        circuit = feature_map.bind_parameters(sample_vector)
        backend = Aer.get_backend('statevector_simulator')
        qc_transpiled = transpile(circuit, backend=backend)
        result = backend.run(qc_transpiled).result()
        state = result.get_statevector()
        fig = plot_bloch_multivector(state)
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        logging.info("🔵 Bloch guardado en %s", filename)
        return filename
    except Exception as e:
        logging.exception("Bloch falló: %s", str(e))
        return None

# ----------- Interpretaciones -----------

def interpretar_superposicion(pca_file):
    return f"""
🌌 Interpretación de {os.path.basename(pca_file)}:
- Colores separados → modelo distingue condiciones.
- Colores mezclados → datos se solapan, recopilar más registros.
"""

def interpretar_clustering(cluster_file):
    return f"""
🌱 Interpretación de {os.path.basename(cluster_file)}:
- Cada color = grupo natural de suelo.
- Si un cluster domina → parcela homogénea.
- Si varios → variabilidad alta.
"""

def interpretar_importancia(importance_file):
    return f"""
📊 Interpretación de {os.path.basename(importance_file)}:
- Barras altas = sensor más influyente.
- Ej: si pH alto → acidez manda.
- Si humedad domina → riego clave.
"""

def interpretar_bloch(bloch_file):
    return f"""
🔵 Interpretación de {os.path.basename(bloch_file)}:
- La esfera Bloch confirma representación cuántica.
- Dispersión amplia → buena separación.
- Concentrado → posible subajuste.
"""

def cultivos_recomendados(values_dict):
    temp = float(values_dict.get("temperatura", np.nan))
    hum = float(values_dict.get("humedad", np.nan))
    ph = float(values_dict.get("ph", np.nan))
    nitr = float(values_dict.get("nitrógeno", np.nan))
    fosf = float(values_dict.get("fósforo", np.nan))
    pot = float(values_dict.get("potasio", np.nan))
    recomendaciones = []

    if not np.isnan(ph) and not np.isnan(hum):
        if ph < 5.5:
            if hum > 50:
                recomendaciones.append("🌱 Papa, camote, café, piña.")
            else:
                recomendaciones.append("🌱 Papa y camote.")
        elif 5.5 <= ph <= 7.5:
            if 45 <= hum <= 65:
                recomendaciones.append("🌱 Maíz, trigo, frijol, hortalizas.")
            elif hum > 65:
                recomendaciones.append("🌱 Arroz, alfalfa, pastos.")
            else:
                recomendaciones.append("🌱 Quinua, papa (con riego).")
        else:
            recomendaciones.append("🌱 Cebada, remolacha, espárrago.")

    nutrient_avg = np.nanmean([nitr, fosf, pot])
    if not np.isnan(nutrient_avg):
        if nutrient_avg < 10:
            recomendaciones.append("⚠️ Nutrientes bajos — leguminosas.")
        elif nutrient_avg < 25:
            recomendaciones.append("💡 Nutrientes moderados — maíz, papa.")
        else:
            recomendaciones.append("✅ Nutrientes altos — tomate, híbridos.")
    return "\n".join(recomendaciones)

def interpretacion_agronomica(values_dict):
    base = []
    temp = float(values_dict.get("temperatura", np.nan))
    hum = float(values_dict.get("humedad", np.nan))
    ph = float(values_dict.get("ph", np.nan))
    nitr = float(values_dict.get("nitrógeno", np.nan))
    fosf = float(values_dict.get("fósforo", np.nan))
    pot = float(values_dict.get("potasio", np.nan))

    if not np.isnan(hum):
        if hum < 40:
            base.append("💧 Riego recomendado: Sí (humedad baja).")
        elif hum < 55:
            base.append("💧 Riego recomendado: Monitorear.")
        else:
            base.append("💧 Riego recomendado: No necesario.")
    if not np.isnan(ph):
        if ph < 5.5:
            base.append("🧪 pH ácido, aplicar cal.")
        elif ph > 7.5:
            base.append("🧪 pH alcalino, usar enmiendas.")
        else:
            base.append("🧪 pH óptimo.")
    nutrient_avg = np.nanmean([nitr, fosf, pot])
    if not np.isnan(nutrient_avg):
        if nutrient_avg < 10:
            base.append("🌾 Nutrientes bajos — fertilizar urgente.")
        elif nutrient_avg < 25:
            base.append("🌾 Nutrientes moderados — fertilización leve.")
        else:
            base.append("🌾 Nutrientes adecuados.")
    base.append("🔍 Cultivos recomendados:")
    base.append(cultivos_recomendados(values_dict))
    return "\n".join(base)

def guardar_interpretaciones(zone_dir, pca_file, cluster_file, importance_file, bloch_file):
    interpretaciones = []
    interpretaciones.append(interpretar_superposicion(pca_file))
    interpretaciones.append(interpretar_clustering(cluster_file))
    interpretaciones.append(interpretar_importancia(importance_file))
    if bloch_file:
        interpretaciones.append(interpretar_bloch(bloch_file))
    with open(os.path.join(zone_dir, "interpretaciones.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(interpretaciones))
    logging.info("📝 Interpretaciones guardadas en interpretaciones.txt")

# -------------- Metadata ----------------

def guardar_metadata(zone_dir, last_data_ts):
    meta = {
        "last_trained_at": last_data_ts.isoformat() if isinstance(last_data_ts, datetime) else str(last_data_ts),
        "trained_on": datetime.utcnow().isoformat()
    }
    with open(os.path.join(zone_dir, "metadata.json"), "w") as f:
        json.dump(meta, f)
    logging.info("Metadata guardada en %s", zone_dir)

# -------------- Entrenamiento por zona --------------

def prepare_zone_dir(zone_id):
    zone_dir = os.path.join(OUTPUT_DIR, f"zone_{zone_id}")
    asegurar_dir(zone_dir)
    return zone_dir

def train_zone(zone_id):
    try:
        cargar_env()
        conn = conectar_bd()
        df = leer_datos(conn, zone_id=zone_id)
        conn.close()
        if df is None or df.empty:
            raise ValueError("No hay datos para la zona solicitada.")

        X, y, df_clean = preparar_dataset(df)
        zone_dir = prepare_zone_dir(zone_id)
        X_scaled, scaler_file = escalar_y_guardar(X, zone_dir, zone_id)

        model, qkernel, feature_map = entrenar_qsvc(X_scaled, y)
        model_file = os.path.join(zone_dir, f"modelo_qsvc_zone_{zone_id}.joblib")
        dump(model, model_file)
        logging.info("✅ Modelo guardado en %s", model_file)

        y_pred = model.predict(X_scaled)
        acc = accuracy_score(y, y_pred)
        logging.info("🎯 Accuracy (zone %s): %.2f%%", zone_id, acc*100)
        logging.info("\n" + classification_report(y, y_pred))

        stats_file = os.path.join(zone_dir, "estadisticas_entrenamiento.csv")
        pca_file = os.path.join(zone_dir, "superposicion_pca.png")
        cluster_file = os.path.join(zone_dir, "clustering_emergente.png")
        importancia_file = os.path.join(zone_dir, "importancia_sensores.png")
        bloch_file = os.path.join(zone_dir, "bloch_superposicion.png")

        generar_estadisticas(df_clean, y, stats_file)
        graficar_superposicion(X_scaled, y, pca_file)
        graficar_clustering(X_scaled, cluster_file)
        graficar_importancia_sensores(X_scaled, y, importancia_file)

        try:
            sample = np.mean(X, axis=0)
            bloch_path = generar_bloch_image(feature_map, sample, bloch_file)
            if not bloch_path and os.path.exists(bloch_file):
                os.remove(bloch_file)
        except Exception:
            logging.exception("No se pudo generar Bloch image.")
            bloch_path = None

        guardar_interpretaciones(zone_dir, pca_file, cluster_file, importancia_file, bloch_path)

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
            "bloch_file": bloch_path if bloch_path else None,
            "interpretaciones": os.path.join(zone_dir, "interpretaciones.txt"),
            "last_trained_at": last_ts.isoformat() if last_ts is not None else None
        }
    except Exception as e:
        logging.exception("Error en train_zone: %s", str(e))
        raise

# -------------- Inferencia ----------------

def load_model_for_zone(zone_id):
    zone_dir = prepare_zone_dir(zone_id)
    model_path = os.path.join(zone_dir, f"modelo_qsvc_zone_{zone_id}.joblib")
    scaler_path = os.path.join(zone_dir, f"scaler_qsvc_zone_{zone_id}.joblib")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Modelo o scaler no encontrado.")
    model = load(model_path)
    scaler = load(scaler_path)
    return model, scaler, zone_dir

def predict_from_values(zone_id, values_dict):
    X = np.array([[float(values_dict.get(col, np.nan)) for col in FEATURE_COLUMNS]], dtype=float)
    if np.isnan(X).any():
        raise ValueError("Faltan valores en values_dict.")
    model, scaler, zone_dir = load_model_for_zone(zone_id)
    X_scaled = scaler.transform(X)
    pred = int(model.predict(X_scaled)[0])
    return pred, X_scaled, zone_dir

# -------------- MAIN ----------------

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        zid = sys.argv[1]
        out = train_zone(zid)
        print("Entrenamiento completado:", out)
    else:
        print("Uso: python train_qsvc_local.py <zone_id>")
