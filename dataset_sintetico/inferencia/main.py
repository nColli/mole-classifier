import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="Mole Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://dominio.com.ar"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

model = YOLO("models/best.pt")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "El archivo debe ser una imagen")

    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "No se pudo leer la imagen")

    # imgsz chico = menos RAM. Si tu modelo fue entrenado a 640, dejalo en 640.
    results = model.predict(image, conf=0.1, imgsz=640, verbose=False)
    r = results[0]

    detections = []
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf[0]),
                "class_id": int(box.cls[0]),
                "class_name": r.names[int(box.cls[0])],
            }
        )

    return {
        "detections": detections,
        "image_size": {"width": image.width, "height": image.height},
    }
