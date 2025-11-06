# Usa una versión estable y fija de Debian (Bullseye) compatible con todos los paquetes
FROM python:3.10-slim-bullseye

# Evitar interacción durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema necesarias (compiladores, librerías matemáticas y gráficas)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgmp-dev \
    libmpfr-dev \
    cmake \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libffi-dev \
    libssl-dev \
    libgl1 \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos del proyecto
COPY . .

# Instalar dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Railway usa por defecto la variable de entorno PORT, pero exponemos 8080 como fallback
EXPOSE 8080

# Comando para iniciar el microservicio FastAPI con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
