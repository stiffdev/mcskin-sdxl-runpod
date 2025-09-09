FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# deps system mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

# deps python
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY . /app

# Variables que puedes ajustar en RunPod (Environment Variables)
#   HF_SPACE_ID  = phenixrhyder/3D-Minecraft-Skin-Generator
#   HF_TOKEN     = <tu token de HF para evitar rate limit>
#   HF_SPACE_URL = (opcional) URL directa del Space si quieres
ENV HF_SPACE_ID=phenixrhyder/3D-Minecraft-Skin-Generator

CMD ["python", "-u", "handler.py"]