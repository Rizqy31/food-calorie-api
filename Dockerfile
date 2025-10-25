FROM python:3.10-slim

WORKDIR /app

# Dependensi OS minimal buat PIL/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# Kode dan model
COPY app.py .
COPY models ./models

# Port service (Render akan baca dari EXPOSE ini)
EXPOSE 8000

# Jalankan FastAPI (tanpa --reload di produksi)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
