FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

WORKDIR /app

# Install Python and system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir --break-system-packages \
    torch torchvision --extra-index-url https://download.pytorch.org/whl/cu126

# Install remaining Python packages
RUN pip install --no-cache-dir --break-system-packages \
    fastapi \
    uvicorn \
    python-multipart \
    pydantic \
    jinja2 \
    Pillow \
    torchmetrics \
    lpips

# Copy app code
COPY main.py .
COPY database.py .
COPY worker.py .
COPY metrics/ metrics/
COPY templates/ templates/
COPY static/ static/

RUN mkdir -p /app/data/uploads /app/data/references /app/data/extracted

EXPOSE 8085

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8085"]
