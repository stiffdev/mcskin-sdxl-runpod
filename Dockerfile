# Base oficial de RunPod con PyTorch 2.1.0, Python 3.10 y CUDA 12.1
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models-cache \
    HUGGINGFACE_HUB_CACHE=/models-cache \
    TRANSFORMERS_CACHE=/models-cache

# Reqs (torch ya viene instalado en la base)
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# IDs de modelo configurables por ENV (puedes sobreescribirlos desde RunPod)
ENV MODEL_ID_SDXL=monadical-labs/minecraft-skin-generator-sdxl \
    MODEL_ID_SD2=monadical-labs/minecraft-skin-generator

WORKDIR /app
COPY . /app

CMD ["python3", "-u", "handler.py"]