FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

ENV HF_HOME=/models TRANSFORMERS_CACHE=/models HUGGINGFACE_HUB_CACHE=/models HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app
COPY handler.py /app

# No hay puerto: Serverless invoca tu handler directamente
CMD ["python3", "handler.py"]
