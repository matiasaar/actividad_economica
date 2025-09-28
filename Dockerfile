 
# Imagen base con soporte CUDA, cuDNN y PyTorch (de NVIDIA)
#FROM nvcr.io/nvidia/pytorch:2.2.0-cuda12.1-cudnn8-runtime #hay que logearse para usarla 
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime


# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar herramientas necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Instalar el CLI de Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copiar el archivo de requerimientos e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todos los archivos del proyecto al contenedor
COPY . .

# Variables de entorno para Ollama
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_MODELS=/app/ollama_models

# Crear directorio para modelos
RUN mkdir -p /app/ollama_models
