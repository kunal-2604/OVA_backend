import sys
import os
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.model_loader import registry
from app.services.ultrasound_service import UltrasoundService
from app.services.histopathology_service import HistopathologyService
from app.services.report_service import ReportService
from PIL import Image

def main():
    print("=" * 60)
    print("Testing Backend Pipeline...")
    print("=" * 60)

    # 1. Load models
    print("\n1. Loading all models...")
    registry.load_all_models()
    status = registry.get_status()
    print("Status:", status)
    assert status["all_loaded"], "Not all models loaded!"

    # 2. Test Ultrasound Pipeline
    print("\n2. Testing Ultrasound Analysis on sample image...")
    us_img_path = Path("../sample_images/ultrasound/1.JPG")
    if not us_img_path.is_file():
        us_img_path = Path("sample_images/ultrasound/1.JPG")
    img = Image.open(us_img_path).convert("RGB")
    us_res = UltrasoundService.analyze_image(img, "1.JPG")
    print("Ultrasound Result ID:", us_res.result_id)
    print("Ensemble Predicted Class:", us_res.ensemble.predicted_class)
    print("Ensemble Category:", us_res.ensemble.category)
    print("Requires Phase 2:", us_res.requires_phase2)
    print("ResNet50 top class:", us_res.models["ResNet50"].predicted_class)
    print("EfficientNet-B0 top class:", us_res.models["EfficientNet-B0"].predicted_class)

    # 3. Test Histopathology Pipeline
    print("\n3. Testing Histopathology Analysis on sample image...")
    h_img_path = Path("../sample_images/histopathology/CC/1170.png")
    if not h_img_path.is_file():
        h_img_path = Path("sample_images/histopathology/CC/1170.png")
    h_img = Image.open(h_img_path).convert("RGB")
    h_res = HistopathologyService.analyze_images(
        [(h_img, "1170.png")],
        ultrasound_result_id=us_res.result_id
    )
    print("Histo Result ID:", h_res.result_id)
    print("ResNet50 Subtype:", h_res.images[0].models["ResNet50"].predicted_full_name)
    print("EffNet Subtype:", h_res.images[0].models["EfficientNet-B4"].predicted_full_name)
    print("XAI Agreement Score:", h_res.images[0].agreement_score)

    # 4. Test PDF Report Generation
    print("\n4. Testing PDF Report Generation...")
    pdf_path = ReportService.generate_pdf(us_res.result_id)
    print("PDF generated at:", pdf_path)
    assert pdf_path.is_file(), "PDF was not created!"
    print("PDF size:", pdf_path.stat().st_size, "bytes")

    print("\n" + "=" * 60)
    print("ALL BACKEND PIPELINE TESTS PASSED 100%!")
    print("=" * 60)

if __name__ == "__main__":
    main()
