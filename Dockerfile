FROM python:3.10-slim

# Instalar dependencias del sistema necesarias
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
    libgl1-mesa-glx \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Puerto expuesto (Railway inyecta PORT automáticamente)
EXPOSE 8080

# Usa PORT en lugar de un número fijo
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
