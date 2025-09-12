FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# upgrade pip y dependencias
RUN pip install --upgrade pip

# instalar requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# caches persistentes
ENV HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    HUGGINGFACE_HUB_CACHE=/models \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

WORKDIR /app

# copiamos código de la app
COPY handler.py /app
COPY mapper.py /app

# entrypoint
CMD ["python", "-u", "handler.py"]
