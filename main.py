import os
import json
import uuid
import datetime
import threading
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import torch

from models_and_xai import (
    manager, DEVICE, STATIC_DIR, TEMP_DIR,
    ULTRASOUND_CLASSES, ULTRASOUND_MALIGNANT_CLASSES,
    HISTO_CLASSES, HISTO_CLASS_LABELS,
    preprocess_pil, get_ultrasound_category,
    generate_gradcam, generate_lime_and_agreement,
    save_image_array, save_pil_image
)
from report_generator import build_pdf_report

# Track background loading state
_models_loading = False
_models_ready = False

def _load_models_background():
    """Load all models in a background thread so the server starts immediately."""
    global _models_loading, _models_ready
    _models_loading = True
    print("Background thread: loading models into memory...")
    try:
        manager.load_all()
        print(f"Background thread: all models loaded on {DEVICE}")
    except Exception as e:
        print(f"Background thread: model loading error — {e}")
    finally:
        _models_ready = True
        _models_loading = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start model loading in a background thread — server binds the port immediately
    thread = threading.Thread(target=_load_models_background, daemon=True)
    thread.start()
    print(f"Server started. Models loading in background on {DEVICE}...")
    yield

app = FastAPI(title="Ovarian Cancer Two-Phase AI Pipeline", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def validate_img(file: UploadFile) -> Image.Image:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        raise HTTPException(400, f"Unsupported file type '{ext}'")
    try:
        file.file.seek(0)
        img = Image.open(file.file).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/")
@app.get("/api/health")
def health():
    if _models_loading:
        return {
            "status": "loading",
            "detail": "Models are loading in the background. The server is online but inference is not yet available.",
            "device": DEVICE,
            "models_loaded": manager.status,
        }
    return {
        "status": "healthy" if all(manager.status.values()) else "degraded",
        "device": DEVICE,
        "models_loaded": manager.status,
    }

# ── Phase 1: Ultrasound ───────────────────────────────────────────────────────
@app.post("/api/ultrasound/predict")
async def predict_ultrasound(file: UploadFile = File(...)):
    if not manager.us_resnet or not manager.us_effnet:
        msg = "Ultrasound models are still loading, please wait a moment and retry." if _models_loading else "Ultrasound models failed to load."
        raise HTTPException(503, msg)

    img = validate_img(file)
    result_id = str(uuid.uuid4())
    ts = datetime.datetime.utcnow().isoformat() + "Z"

    orig_url = save_pil_image(img, result_id, "original_ultrasound.png")
    rgb, tensor = preprocess_pil(img)

    # ResNet50
    with torch.no_grad():
        probs_r = torch.softmax(manager.us_resnet(tensor).float(), dim=1)[0].cpu().numpy()
    pred_r = int(probs_r.argmax())
    cam_r, over_r = generate_gradcam(manager.us_resnet, manager.us_resnet_target, tensor, pred_r, rgb)
    r_cam_url = save_image_array(cam_r, result_id, "resnet_gradcam.png")
    r_over_url = save_image_array(over_r, result_id, "resnet_overlay.png")

    # EfficientNet-B0
    with torch.no_grad():
        probs_e = torch.softmax(manager.us_effnet(tensor).float(), dim=1)[0].cpu().numpy()
    pred_e = int(probs_e.argmax())
    cam_e, over_e = generate_gradcam(manager.us_effnet, manager.us_effnet_target, tensor, pred_e, rgb)
    e_cam_url = save_image_array(cam_e, result_id, "effnet_gradcam.png")
    e_over_url = save_image_array(over_e, result_id, "effnet_overlay.png")

    # Ensemble
    ens_probs = (probs_r + probs_e) / 2.0
    pred_ens = int(ens_probs.argmax())
    ens_class = ULTRASOUND_CLASSES[pred_ens]
    ens_conf = float(ens_probs[pred_ens])
    category = get_ultrasound_category(ens_class)

    is_mal = (category == "Malignant-Suspicious" or
              ULTRASOUND_CLASSES[pred_r] in ULTRASOUND_MALIGNANT_CLASSES or
              ULTRASOUND_CLASSES[pred_e] in ULTRASOUND_MALIGNANT_CLASSES)

    prob_list = [
        {"class_name": c, "probability": round(float(p), 4), "percentage": round(float(p)*100, 2), "category": get_ultrasound_category(c)}
        for c, p in zip(ULTRASOUND_CLASSES, ens_probs)
    ]
    prob_list.sort(key=lambda x: x["probability"], reverse=True)

    resp = {
        "result_id": result_id,
        "timestamp": ts,
        "original_image_url": orig_url,
        "requires_phase2": is_mal,
        "recommendation": "Malignant-suspicious ultrasound lesion detected. Urgent Phase 2 referral recommended." if is_mal else
                         ("Normal ovary visualized. Routine clinical follow-up suggested." if category == "Normal" else
                          "Ultrasound findings consistent with benign pathology. Routine gynecological monitoring suggested."),
        "ensemble": {
            "predicted_class": ens_class,
            "confidence": round(ens_conf, 4),
            "category": category,
            "is_malignant_suspicious": is_mal,
            "explanation": f"Ensemble prediction identifies '{ens_class}' ({category}) with {ens_conf*100:.1f}% confidence.",
            "probabilities": prob_list
        },
        "models": {
            "EfficientNet-B0": {
                "model_name": "EfficientNet-B0",
                "predicted_class": ULTRASOUND_CLASSES[pred_e],
                "confidence": round(float(probs_e[pred_e]), 4),
                "category": get_ultrasound_category(ULTRASOUND_CLASSES[pred_e]),
                "gradcam_url": e_cam_url, "overlay_url": e_over_url,
                "probabilities": [{"class_name": c, "percentage": round(float(p)*100, 2), "probability": round(float(p), 4), "category": get_ultrasound_category(c)} for c, p in zip(ULTRASOUND_CLASSES, probs_e)]
            },
            "ResNet50": {
                "model_name": "ResNet50",
                "predicted_class": ULTRASOUND_CLASSES[pred_r],
                "confidence": round(float(probs_r[pred_r]), 4),
                "category": get_ultrasound_category(ULTRASOUND_CLASSES[pred_r]),
                "gradcam_url": r_cam_url, "overlay_url": r_over_url,
                "probabilities": [{"class_name": c, "percentage": round(float(p)*100, 2), "probability": round(float(p), 4), "category": get_ultrasound_category(c)} for c, p in zip(ULTRASOUND_CLASSES, probs_r)]
            }
        }
    }

    # Persist session
    with open(TEMP_DIR / f"{result_id}.json", "w", encoding="utf-8") as f:
        json.dump({"result_id": result_id, "created_at": ts, "ultrasound": resp}, f, indent=2)

    return resp

# ── Phase 2: Histopathology ──────────────────────────────────────────────────
@app.post("/api/histopathology/predict")
async def predict_histopathology(files: List[UploadFile] = File(...), ultrasound_result_id: Optional[str] = Form(None)):
    if not manager.histo_resnet or not manager.histo_effnet:
        msg = "Histopathology models are still loading, please wait a moment and retry." if _models_loading else "Histopathology models failed to load."
        raise HTTPException(503, msg)

    result_id = ultrasound_result_id if ultrasound_result_id else str(uuid.uuid4())
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    images_res = []

    for idx, f in enumerate(files):
        img = validate_img(f)
        sid = f"slide_{idx+1}"
        orig_url = save_pil_image(img, result_id, f"{sid}_original.png")
        rgb, tensor = preprocess_pil(img)

        # ResNet50
        with torch.no_grad():
            probs_r = torch.softmax(manager.histo_resnet(tensor).float(), dim=1)[0].cpu().numpy()
        pred_r = int(probs_r.argmax())
        code_r = HISTO_CLASSES[pred_r]
        cam_r, over_r = generate_gradcam(manager.histo_resnet, manager.histo_resnet_target, tensor, pred_r, rgb)
        r_cam_url = save_image_array(cam_r, result_id, f"{sid}_resnet_gradcam.png")
        r_over_url = save_image_array(over_r, result_id, f"{sid}_resnet_overlay.png")

        # EfficientNet-B4
        with torch.no_grad():
            probs_e = torch.softmax(manager.histo_effnet(tensor).float(), dim=1)[0].cpu().numpy()
        pred_e = int(probs_e.argmax())
        code_e = HISTO_CLASSES[pred_e]
        cam_e, over_e = generate_gradcam(manager.histo_effnet, manager.histo_effnet_target, tensor, pred_e, rgb)
        e_cam_url = save_image_array(cam_e, result_id, f"{sid}_effnet_gradcam.png")
        e_over_url = save_image_array(over_e, result_id, f"{sid}_effnet_overlay.png")

        # LIME & Agreement
        mask, lime_over, soft, precision = generate_lime_and_agreement(manager.histo_resnet, rgb, pred_r, cam_r)
        lime_over_url = save_image_array(lime_over, result_id, f"{sid}_lime_overlay.png")
        lime_mask_url = save_image_array(mask, result_id, f"{sid}_lime_mask.png")
        soft_url = save_image_array(soft, result_id, f"{sid}_soft_agreement.png")

        def fmt_histo(probs):
            return [{"class_name": c, "full_name": HISTO_CLASS_LABELS[c], "probability": round(float(p), 4), "percentage": round(float(p)*100, 2)} for c, p in zip(HISTO_CLASSES, probs)]

        images_res.append({
            "image_id": sid,
            "filename": f.filename or "slide",
            "original_image_url": orig_url,
            "lime_overlay_url": lime_over_url,
            "lime_mask_url": lime_mask_url,
            "soft_agreement_url": soft_url,
            "agreement_score": round(precision, 3),
            "models": {
                "ResNet50": {
                    "model_name": "ResNet50", "predicted_class": code_r, "predicted_full_name": HISTO_CLASS_LABELS[code_r],
                    "confidence": round(float(probs_r[pred_r]), 4), "gradcam_url": r_cam_url, "overlay_url": r_over_url,
                    "probabilities": fmt_histo(probs_r)
                },
                "EfficientNet-B4": {
                    "model_name": "EfficientNet-B4", "predicted_class": code_e, "predicted_full_name": HISTO_CLASS_LABELS[code_e],
                    "confidence": round(float(probs_e[pred_e]), 4), "gradcam_url": e_cam_url, "overlay_url": e_over_url,
                    "probabilities": fmt_histo(probs_e)
                }
            }
        })

    resp = {
        "result_id": result_id,
        "ultrasound_result_id": ultrasound_result_id,
        "timestamp": ts,
        "images": images_res,
        "summary_notes": f"Histopathological classification completed across {len(files)} slide image(s)."
    }

    # Save to session JSON
    sess_file = TEMP_DIR / f"{result_id}.json"
    data = {}
    if sess_file.is_file():
        try:
            with open(sess_file, "r", encoding="utf-8") as fp: data = json.load(fp)
        except Exception: pass
    data["result_id"] = result_id
    data["histopathology"] = resp
    with open(sess_file, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)

    return resp

# ── Results & PDF ─────────────────────────────────────────────────────────────
@app.get("/api/results/{result_id}")
def get_result(result_id: str):
    sess_file = TEMP_DIR / f"{result_id}.json"
    if not sess_file.is_file():
        raise HTTPException(404, "Session not found")
    with open(sess_file, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/reports/{result_id}")
def get_report(result_id: str):
    sess_file = TEMP_DIR / f"{result_id}.json"
    if not sess_file.is_file():
        raise HTTPException(404, "Session not found")
    with open(sess_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    pdf = build_pdf_report(result_id, data)
    return FileResponse(pdf, filename=f"Ovarian_Report_{result_id[:8]}.pdf", media_type="application/pdf")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
