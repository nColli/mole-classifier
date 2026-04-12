from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import HTMLResponse
from pathlib import Path
from services.detector import MoleDetector

app = FastAPI()
detector = MoleDetector(model_path="best.pt")

@app.get("/", response_class=HTMLResponse)
def home():
    html = Path("templates/index.html").read_text()
    return HTMLResponse(content=html)

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.45),
    iou: float = Query(0.45),
    agnostic_nms: bool = Query(True)
):
    contents = await file.read()
    return detector.predict(contents, conf=conf, iou=iou, agnostic_nms=agnostic_nms)