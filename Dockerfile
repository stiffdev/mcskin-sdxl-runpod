# Imagen base optimizada de RunPod con PyTorch + CUDA ya instalado
FROM runpod/pytorch:3.10-2.1.0-12.1.1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models-cache \
    HUGGINGFACE_HUB_CACHE=/models-cache \
    TRANSFORMERS_CACHE=/models-cache

# Dependencias de Python (torch ya viene en la base)
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Variables para elegir modelos en runtime (SDXL por defecto)
ENV MODEL_ID_SDXL=monadical-labs/minecraft-skin-generator-sdxl \
    MODEL_ID_SD2=monadical-labs/minecraft-skin-generator

# No activamos offline en runtime (necesitamos bajar del Hub al primer arranque)
# Si más adelante montas el modelo en disco, podrás activar:
# ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

WORKDIR /app
COPY . /app

# Lanza el handler de RunPod
CMD ["python3", "-u", "handler.py"]