from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ===== Konfigurasi =====
MODEL_PATH = "models/best.pt"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
DEVICE = "cpu"

# Kalori per 100 gram
CALORIES = {
    "nasi putih": 175.0,
    "telur mata sapi": 90.0
}

# Konstanta untuk kalibrasi perkiraan berat dari area gambar
SCALE_FACTOR = 2000  # semakin besar = berat semakin tinggi

# ===== Inisialisasi FastAPI =====
app = FastAPI(
    title="Food Calorie YOLO API",
    description="Deteksi nasi putih & telur mata sapi, estimasi berat & total kalori.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ===== Load model sekali di awal =====
try:
    model = YOLO(MODEL_PATH)
    model_names = model.model.names if hasattr(model, "model") else model.names
    missing = [name for name in model_names.values() if name not in CALORIES]
    if missing:
        print(f"[WARN] Kelas tanpa mapping kalori: {missing}")
    print(f"[INFO] Model loaded: {MODEL_PATH}")
    print(f"[INFO] Classes: {model_names}")
except Exception as e:
    print(f"[FATAL] Gagal load model: {e}")
    raise


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "classes": model_names}


@app.get("/labels")
def labels():
    return {"model_classes": model_names, "calorie_mapping": CALORIES}


@app.post("/predict")
async def predict(file: UploadFile):
    # Validasi tipe file
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="File harus jpg/png/webp")

    # Baca gambar
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Gagal membaca gambar")

    # Inference
    try:
        results = model.predict(
            source=np.array(img),
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    r = results[0]
    detections = []
    aggregate = {}
    names = r.names
    w, h = img.size

    if r.boxes is not None and r.boxes.xyxy is not None:
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(clss)):
            cls_id = int(clss[i])
            name = names.get(cls_id, str(cls_id))
            kcal_per_100g = float(CALORIES.get(name, 0.0))

            x1, y1, x2, y2 = map(float, xyxy[i])
            area = (x2 - x1) * (y2 - y1)
            relative_area = area / (w * h)

            # Estimasi berat dan kalori berdasarkan luas area
            weight_est = relative_area * SCALE_FACTOR
            kcal_est = (kcal_per_100g / 100.0) * weight_est

            detections.append({
                "class_id": cls_id,
                "class_name": name,
                "confidence": float(conf[i]),
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "estimated_weight_g": round(weight_est, 2),
                "calories_per_100g": kcal_per_100g,
                "estimated_calories": round(kcal_est, 2)
            })

            if name not in aggregate:
                aggregate[name] = {"count": 0, "weight_g": 0.0, "calories": 0.0}
            aggregate[name]["count"] += 1
            aggregate[name]["weight_g"] += weight_est
            aggregate[name]["calories"] += kcal_est

    total_calories = float(sum(v["calories"] for v in aggregate.values()))
    total_weight = float(sum(v["weight_g"] for v in aggregate.values()))

    return {
        "image_size": {"width": w, "height": h},
        "detections": detections,
        "aggregate": aggregate,
        "total_weight_g": round(total_weight, 2),
        "total_calories": round(total_calories, 2)
    }
