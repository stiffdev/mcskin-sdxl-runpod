FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models-cache \
    HUGGINGFACE_HUB_CACHE=/models-cache \
    TRANSFORMERS_CACHE=/models-cache

RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

# Instala PyTorch con soporte CUDA 12.1
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Reqs (no pongas "torch" en requirements.txt)
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Config por defecto (sobreescribes en RunPod → Environment Variables)
ENV MODEL_ID_SDXL=monadical-labs/minecraft-skin-generator-sdxl \
    MODEL_ID_SD2=monadical-labs/minecraft-skin-generator

WORKDIR /app
COPY . /app

CMD ["python3", "-u", "handler.py"]