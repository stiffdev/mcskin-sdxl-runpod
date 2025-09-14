FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Clonamos el repo EXACTO que usa la Space
WORKDIR /deps
RUN git clone --depth 1 https://github.com/Monadical-SAS/minecraft_skin_generator.git

# Cachés HF a disco persistente (se sobreescriben en tiempo de ejecución por handler)
ENV HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    HUGGINGFACE_HUB_CACHE=/models \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

WORKDIR /app
COPY handler.py /app

CMD ["python", "-u", "handler.py"]
