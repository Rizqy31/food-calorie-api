FROM python:3.10-slim

WORKDIR /app

# Install dependensi OS biar YOLO bisa load gambar
RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

# Install PyTorch langsung dari repo resmi (CPU only, cepat banget)
RUN pip install torch==2.3.1 torchvision --index-url https://download.pytorch.org/whl/cpu

# Install library lain
RUN pip install --no-cache-dir ultralytics fastapi uvicorn python-multipart Pillow numpy

# Copy semua file ke container
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
