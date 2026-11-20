import os
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
)
from models_and_xai import STATIC_DIR

def build_pdf_report(result_id: str, data: dict) -> Path:
    pdf_path = STATIC_DIR / result_id / f"Report_{result_id[:8]}.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Heading1"], fontSize=18, leading=22, textColor=colors.HexColor("#0F2942"))
    sub_style = ParagraphStyle("S", parent=styles["Normal"], fontSize=10, leading=13, textColor=colors.HexColor("#334E68"))
    h2_style = ParagraphStyle("H", parent=styles["Heading2"], fontSize=12, leading=16, textColor=colors.HexColor("#102A43"))
    body_style = ParagraphStyle("B", parent=styles["Normal"], fontSize=9, leading=12, textColor=colors.HexColor("#243B53"))
    disc_style = ParagraphStyle("D", parent=styles["Normal"], fontSize=8, leading=11, textColor=colors.HexColor("#627D98"))

    story = [
        Paragraph("OVARIAN CANCER MULTIMODAL AI PIPELINE", title_style),
        Paragraph("Phase 1 Screening & Phase 2 Histopathological Subtype Classification Report", sub_style),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0EA5E9"), spaceAfter=10),
    ]

    # Meta
    meta = [
        [Paragraph(f"<b>Report ID:</b> {result_id}", body_style), Paragraph(f"<b>Timestamp:</b> {data.get('created_at', 'N/A')}", body_style)],
        [Paragraph("<b>Pipeline:</b> Ultrasound (MMOTU) + Histopathology Subtypes", body_style), Paragraph("<b>Status:</b> Complete Analysis", body_style)],
    ]
    t = Table(meta, colWidths=[270, 270])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F0F4F8")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#D9E2EC")),
        ('TOPPADDING', (0,0), (-1,-1), 5), ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Phase 1 Ultrasound
    us = data.get("ultrasound")
    if us:
        ens = us.get("ensemble", {})
        story.append(Paragraph("Phase 1: Ovarian Ultrasound Lesion Screening", h2_style))
        story.append(Paragraph(
            f"<b>Finding:</b> {ens.get('predicted_class')} &nbsp;|&nbsp; <b>Category:</b> {ens.get('category')} "
            f"&nbsp;|&nbsp; <b>Confidence:</b> {ens.get('confidence', 0)*100:.1f}%<br/>"
            f"<b>Assessment:</b> {ens.get('explanation', '')}", body_style
        ))
        story.append(Spacer(1, 8))

        # Images
        dir_p = STATIC_DIR / result_id
        imgs, lbls = [], []
        for fn, lbl in [("original_ultrasound.png", "Original Ultrasound"), ("effnet_overlay.png", "EfficientNet-B0 Grad-CAM"), ("resnet_overlay.png", "ResNet50 Grad-CAM")]:
            fp = dir_p / fn
            if fp.is_file():
                imgs.append(RLImage(str(fp), width=160, height=140))
                lbls.append(Paragraph(f"<b>{lbl}</b>", body_style))
        if imgs:
            it = Table([imgs, lbls], colWidths=[180]*len(imgs))
            it.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('BOTTOMPADDING', (0,0), (-1,-1), 4)]))
            story.append(it)
        story.append(Spacer(1, 10))

    # Phase 2 Histopathology
    histo = data.get("histopathology")
    if histo:
        story.append(Paragraph("Phase 2: Histopathological Subtype Classification & XAI", h2_style))
        for idx, im in enumerate(histo.get("images", [])):
            e = im.get("models", {}).get("EfficientNet-B4", {})
            r = im.get("models", {}).get("ResNet50", {})
            story.append(Paragraph(
                f"<b>Slide {idx+1}:</b> {im.get('filename')} — <b>ResNet50:</b> {r.get('predicted_full_name')} ({r.get('confidence', 0)*100:.1f}%) | "
                f"<b>EfficientNet-B4:</b> {e.get('predicted_full_name')} ({e.get('confidence', 0)*100:.1f}%) | "
                f"<b>XAI Agreement:</b> {im.get('agreement_score', 0):.3f}", body_style
            ))
            story.append(Spacer(1, 6))

            dir_p = STATIC_DIR / result_id
            sid = im.get("image_id", f"slide_{idx+1}")
            himgs, hlbls = [], []
            for fn, lbl in [(f"{sid}_original.png", "Original Slide"), (f"{sid}_resnet_overlay.png", "Grad-CAM Overlay"), (f"{sid}_lime_overlay.png", "LIME Superpixels")]:
                fp = dir_p / fn
                if fp.is_file():
                    himgs.append(RLImage(str(fp), width=160, height=140))
                    hlbls.append(Paragraph(f"<b>{lbl}</b>", body_style))
            if himgs:
                ht = Table([himgs, hlbls], colWidths=[180]*len(himgs))
                ht.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('BOTTOMPADDING', (0,0), (-1,-1), 4)]))
                story.append(ht)
            story.append(Spacer(1, 10))

    # Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CBD5E1"), spaceBefore=10, spaceAfter=6))
    story.append(Paragraph(
        "<b>CLINICAL RESEARCH NOTICE:</b> Generated by an AI deep learning research model. "
        "Does not constitute medical advice or autonomous diagnosis. Specialised clinical review required.", disc_style
    ))

    doc.build(story)
    return pdf_path
