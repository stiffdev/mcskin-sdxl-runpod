FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models-cache \
    HUGGINGFACE_HUB_CACHE=/models-cache \
    TRANSFORMERS_CACHE=/models-cache

RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*

# Torch CUDA 12.1
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ---------- Bake de modelos para evitar cold start ----------
ARG HF_TOKEN
# SDXL (mejor calidad)
ARG MODEL_ID_SDXL=monadical-labs/minecraft-skin-generator-sdxl
# SD 2.x (opcional fallback)
ARG MODEL_ID_SD2=monadical-labs/minecraft-skin-generator

RUN python3 - <<'PY'
import os
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
tok = os.environ.get("HF_TOKEN")

print("[bake] bajando SDXL...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    os.environ["MODEL_ID_SDXL"], use_safetensors=True, token=tok
)
pipe.save_pretrained("/models/sdxl")
print("[bake] SDXL en /models/sdxl")

print("[bake] bajando SD2 (fallback)...")
pipe2 = StableDiffusionPipeline.from_pretrained(
    os.environ["MODEL_ID_SD2"], use_safetensors=True, token=tok, safety_checker=None
)
pipe2.save_pretrained("/models/sd2")
print("[bake] SD2 en /models/sd2")
PY

# Modo offline en runtime
ENV BASE_MODEL_PATH_SDXL=/models/sdxl \
    BASE_MODEL_PATH_SD2=/models/sd2 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app
COPY . /app

CMD ["python3", "-u", "handler.py"]
