import os
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import cv2
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ── Paths & Hardware ──────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MODELS_DIR = BACKEND_DIR / "models"
STATIC_DIR = BACKEND_DIR / "static"
TEMP_DIR = BACKEND_DIR / "temp"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Class Taxonomy ────────────────────────────────────────────────────────────
ULTRASOUND_CLASSES = [
    "Chocolate cyst",
    "Serous cystadenoma",
    "Teratoma",
    "Theca cell tumor",
    "Simple cyst",
    "Normal ovary",
    "Mucinous cystadenoma",
    "High-grade serous",
]

# Verified Malignant-Suspicious Classes
ULTRASOUND_MALIGNANT_CLASSES = ["High-grade serous", "Mucinous cystadenoma"]
ULTRASOUND_NORMAL_CLASSES = ["Normal ovary"]

HISTO_CLASSES = ["CC", "EC", "HGSC", "LGSC", "MC"]
HISTO_CLASS_LABELS = {
    "CC": "Clear Cell Carcinoma",
    "EC": "Endometrioid Carcinoma",
    "HGSC": "High-Grade Serous Carcinoma",
    "LGSC": "Low-Grade Serous Carcinoma",
    "MC": "Mucinous Carcinoma",
}

# Image Preprocessing Constants
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MEAN_NP = np.array(IMAGENET_MEAN, dtype=np.float32)
STD_NP = np.array(IMAGENET_STD, dtype=np.float32)

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Image Utilities ───────────────────────────────────────────────────────────
def preprocess_pil(image: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
    resized = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    rgb_arr = np.array(resized, dtype=np.float32) / 255.0
    tensor = eval_transform(resized).unsqueeze(0).to(DEVICE)
    return rgb_arr, tensor

def preprocess_numpy(img_rgb: np.ndarray) -> torch.Tensor:
    norm = (img_rgb - MEAN_NP) / STD_NP
    return torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    cam_uint8 = np.uint8(255 * np.clip(cam, 0, 1))
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip((1.0 - alpha) * rgb + alpha * heatmap, 0.0, 1.0)

def save_image_array(arr: np.ndarray, result_id: str, filename: str) -> str:
    target_dir = STATIC_DIR / result_id
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / filename
    if arr.dtype in (np.float32, np.float64):
        uint8_arr = np.uint8(np.clip(arr * 255.0, 0, 255))
    else:
        uint8_arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(uint8_arr).save(out_path, quality=95)
    return f"/static/{result_id}/{filename}"

def save_pil_image(image: Image.Image, result_id: str, filename: str) -> str:
    target_dir = STATIC_DIR / result_id
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / filename
    image.save(out_path, quality=95)
    return f"/static/{result_id}/{filename}"

def get_ultrasound_category(class_name: str) -> str:
    if class_name in ULTRASOUND_MALIGNANT_CLASSES:
        return "Malignant-Suspicious"
    elif class_name in ULTRASOUND_NORMAL_CLASSES:
        return "Normal"
    return "Benign"

# ── Model Registry & Lazy Loaders ─────────────────────────────────────────────
import gc

class ModelManager:
    """
    Lazy, phase-swapping model manager.
    Only one phase's models live in memory at a time.
    When the other phase is requested, the current phase is unloaded first.
    This keeps peak RAM under ~400 MB (PyTorch + 2 models).
    """

    def __init__(self):
        self.us_resnet: Optional[nn.Module] = None
        self.us_resnet_target: Optional[nn.Module] = None
        self.us_effnet: Optional[nn.Module] = None
        self.us_effnet_target: Optional[nn.Module] = None

        self.histo_resnet: Optional[nn.Module] = None
        self.histo_resnet_target: Optional[nn.Module] = None
        self.histo_effnet: Optional[nn.Module] = None
        self.histo_effnet_target: Optional[nn.Module] = None

        self._active_phase: Optional[str] = None  # "ultrasound" | "histopathology" | None

        self.status = {
            "ultrasound_resnet50": False,
            "ultrasound_efficientnet": False,
            "histopathology_resnet50": False,
            "histopathology_efficientnet_b4": False,
        }

    # ── Unload helpers ────────────────────────────────────────────────────────
    def _unload_ultrasound(self):
        self.us_resnet = None
        self.us_resnet_target = None
        self.us_effnet = None
        self.us_effnet_target = None
        self.status["ultrasound_resnet50"] = False
        self.status["ultrasound_efficientnet"] = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Unloaded ultrasound models to free memory.")

    def _unload_histopathology(self):
        self.histo_resnet = None
        self.histo_resnet_target = None
        self.histo_effnet = None
        self.histo_effnet_target = None
        self.status["histopathology_resnet50"] = False
        self.status["histopathology_efficientnet_b4"] = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Unloaded histopathology models to free memory.")

    # ── Load helpers ──────────────────────────────────────────────────────────
    def ensure_ultrasound(self):
        """Load ultrasound models if not already loaded. Unloads histopathology first."""
        if self.us_resnet and self.us_effnet:
            return  # already loaded

        # Free the other phase first
        if self._active_phase == "histopathology":
            self._unload_histopathology()

        print("Loading ultrasound models...")

        # ResNet50
        try:
            m = models.resnet50(weights=None)
            m.fc = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(m.fc.in_features, len(ULTRASOUND_CLASSES)))
            ckpt_p = MODELS_DIR / "ultrasound" / "resnet50_best.pth"
            ckpt = torch.load(ckpt_p, map_location=DEVICE, weights_only=False)
            m.load_state_dict(ckpt.get("model_state_dict", ckpt))
            del ckpt
            gc.collect()
            m.to(DEVICE).eval()
            self.us_resnet = m
            self.us_resnet_target = m.layer4[-1]
            self.status["ultrasound_resnet50"] = True
        except Exception as e:
            print(f"Error loading Ultrasound ResNet50: {e}")

        # EfficientNet-B0
        try:
            m = models.efficientnet_b0(weights=None)
            m.classifier = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(m.classifier[1].in_features, len(ULTRASOUND_CLASSES)))
            ckpt_p = MODELS_DIR / "ultrasound" / "efficientnet_best.pth"
            ckpt = torch.load(ckpt_p, map_location=DEVICE, weights_only=False)
            m.load_state_dict(ckpt.get("model_state_dict", ckpt))
            del ckpt
            gc.collect()
            m.to(DEVICE).eval()
            self.us_effnet = m
            self.us_effnet_target = m.features[-1]
            self.status["ultrasound_efficientnet"] = True
        except Exception as e:
            print(f"Error loading Ultrasound EfficientNet: {e}")

        self._active_phase = "ultrasound"
        print("Ultrasound models ready.")

    def ensure_histopathology(self):
        """Load histopathology models if not already loaded. Unloads ultrasound first."""
        if self.histo_resnet and self.histo_effnet:
            return  # already loaded

        # Free the other phase first
        if self._active_phase == "ultrasound":
            self._unload_ultrasound()

        print("Loading histopathology models...")

        # ResNet50
        try:
            m = models.resnet50(weights=None)
            in_f = m.fc.in_features
            m.fc = nn.Sequential(nn.BatchNorm1d(in_f), nn.Dropout(p=0.3), nn.Linear(in_f, len(HISTO_CLASSES)))
            ckpt_p = MODELS_DIR / "histopathology" / "resnet_best.pth"
            ckpt = torch.load(ckpt_p, map_location=DEVICE, weights_only=False)
            m.load_state_dict(ckpt.get("model_state_dict", ckpt))
            del ckpt
            gc.collect()
            m.to(DEVICE).eval()
            self.histo_resnet = m
            self.histo_resnet_target = m.layer4[-1]
            self.status["histopathology_resnet50"] = True
        except Exception as e:
            print(f"Error loading Histopathology ResNet: {e}")

        # EfficientNet-B4
        try:
            m = timm.create_model("efficientnet_b4", pretrained=False, num_classes=0, global_pool="avg")
            in_f = m.num_features
            m.classifier = nn.Sequential(
                nn.BatchNorm1d(in_f), nn.Dropout(p=0.3), nn.Linear(in_f, 512),
                nn.GELU(), nn.Dropout(p=0.15), nn.Linear(512, len(HISTO_CLASSES))
            )
            ckpt_p = MODELS_DIR / "histopathology" / "effnet_best.pth"
            ckpt = torch.load(ckpt_p, map_location=DEVICE, weights_only=False)
            m.load_state_dict(ckpt.get("model_state_dict", ckpt))
            del ckpt
            gc.collect()
            m.to(DEVICE).eval()
            self.histo_effnet = m
            self.histo_effnet_target = m.blocks[-1][-1]
            self.status["histopathology_efficientnet_b4"] = True
        except Exception as e:
            print(f"Error loading Histopathology EfficientNet: {e}")

        self._active_phase = "histopathology"
        print("Histopathology models ready.")

    def load_all(self):
        """Backward compat — just load ultrasound (the first phase the user hits)."""
        self.ensure_ultrasound()

manager = ModelManager()

# ── Explainable AI (Grad-CAM & LIME) ───────────────────────────────────────────
def generate_gradcam(model: nn.Module, target_layer: nn.Module, tensor: torch.Tensor, target_class: int, rgb: np.ndarray):
    cam = GradCAM(model=model, target_layers=[target_layer])
    try:
        raw_cam = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(target_class)])[0]
    finally:
        if hasattr(cam, 'activations_and_grads'):
            cam.activations_and_grads.release()

    c_min, c_max = raw_cam.min(), raw_cam.max()
    norm_cam = (raw_cam - c_min) / (c_max - c_min + 1e-8) if c_max > c_min else np.zeros_like(raw_cam)
    overlay = overlay_cam(rgb, norm_cam)
    return norm_cam, overlay

def generate_lime_and_agreement(model: nn.Module, rgb: np.ndarray, target_class: int, cam_map: np.ndarray, samples: int = 160):
    def predict_fn(images):
        batch = torch.cat([preprocess_numpy(im.astype(np.float32)) for im in images], dim=0)
        with torch.no_grad():
            return torch.softmax(model(batch).float(), dim=1).cpu().numpy()

    explainer = lime_image.LimeImageExplainer(random_state=42)
    explanation = explainer.explain_instance(
        rgb.astype(np.double), predict_fn, top_labels=len(HISTO_CLASSES), hide_color=0, num_samples=samples
    )
    temp, mask = explanation.get_image_and_mask(target_class, positive_only=False, num_features=8, hide_rest=False, min_weight=0.03)
    lime_overlay = mark_boundaries(temp, mask, color=(0.1, 0.8, 0.9), mode='thick')

    # Agreement score
    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
    h, w = cam_map.shape
    lime_resized = np.array(Image.fromarray(mask.astype(np.float32)).resize((w, h), Image.Resampling.NEAREST))
    gc_n = norm(cam_map)
    lime_pos = (norm(lime_resized) < 0.5).astype(np.uint8)
    gc_b = (gc_n >= 0.4).astype(np.uint8)
    soft = norm(gc_n * lime_pos.astype(np.float32))
    precision = float((gc_b & lime_pos).sum() / (gc_b.sum() + 1e-8))

    return mask.astype(np.float32), lime_overlay.astype(np.float32), soft, precision
