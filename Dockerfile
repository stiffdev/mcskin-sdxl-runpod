# PyTorch 2.3.1 + CUDA 12.1 + cuDNN, ya listo
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Opcional pero recomendado
ENV DEBIAN_FRONTEND=noninteractive
RUN pip install --upgrade pip

# Instala deps (sin torch)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Caches de HF y memoria de PyTorch
ENV HF_HOME=/models \
    TRANSFORMERS_CACHE=/models \
    HUGGINGFACE_HUB_CACHE=/models \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

WORKDIR /app
COPY handler.py /app

# Serverless: el proceso queda en bucle esperando jobs
CMD ["python", "-u", "handler.py"]