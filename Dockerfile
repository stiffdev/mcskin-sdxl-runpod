FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

# Instala requirements con versiones fijadas
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# (Opcional pero recomendado) cache de modelos para build reproducible
ENV HF_HOME=/models TRANSFORMERS_CACHE=/models HUGGINGFACE_HUB_CACHE=/models HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app
COPY app.py mapper.py /app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host","0.0.0.0","--port","8000"]