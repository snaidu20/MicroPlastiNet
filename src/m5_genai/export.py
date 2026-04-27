"""
export.py — Report export utilities for MicroPlastiNet M5 GenAI

Supports:
  - Markdown (.md) export
  - PDF export via ReportLab
"""

import os
import re
from datetime import datetime
from pathlib import Path


# ─── Markdown export ───────────────────────────────────────────────────────────

def export_markdown(report_text: str, output_path: str | Path) -> Path:
    """Write report text to a .md file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Markdown report saved: {output_path}")
    return output_path


# ─── PDF export via ReportLab ──────────────────────────────────────────────────

def export_pdf(report_text: str, output_path: str | Path,
               station_id: str = "", event_id: str = "") -> Path:
    """
    Render report text to a polished PDF using ReportLab.

    The PDF uses a two-column masthead, section headers with rule lines,
    and body text in a readable serif-style font.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Colors ─────────────────────────────────────────────────────────────────
    DEEP_NAVY   = colors.HexColor("#0a0e1a")
    TEAL        = colors.HexColor("#00b8a9")
    CYAN        = colors.HexColor("#00d4ff")
    DARK_GRAY   = colors.HexColor("#1a2235")
    MID_GRAY    = colors.HexColor("#64748b")
    TEXT        = colors.HexColor("#1e293b")
    WHITE       = colors.white

    # ── Document ───────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.85 * inch,
        leftMargin=0.85 * inch,
        topMargin=1.0 * inch,
        bottomMargin=0.85 * inch,
        title=f"MicroPlastiNet Report — {station_id}",
        author="MicroPlastiNet Automated Monitoring System",
        subject="Environmental Compliance Report",
    )

    # ── Styles ─────────────────────────────────────────────────────────────────
    base = getSampleStyleSheet()

    doc_title_style = ParagraphStyle(
        "DocTitle",
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=WHITE,
        spaceAfter=4,
        leading=22,
    )
    doc_subtitle_style = ParagraphStyle(
        "DocSubtitle",
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#94a3b8"),
        spaceAfter=2,
    )
    section_header_style = ParagraphStyle(
        "SectionHeader",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=CYAN,
        spaceBefore=16,
        spaceAfter=4,
        leading=14,
    )
    body_style = ParagraphStyle(
        "Body",
        fontName="Times-Roman",
        fontSize=10,
        textColor=TEXT,
        spaceAfter=8,
        leading=15,
        alignment=TA_JUSTIFY,
    )
    bold_body_style = ParagraphStyle(
        "BoldBody",
        fontName="Times-Bold",
        fontSize=10,
        textColor=TEXT,
        spaceAfter=4,
        leading=14,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        fontName="Times-Roman",
        fontSize=10,
        textColor=TEXT,
        spaceAfter=3,
        leading=14,
        leftIndent=14,
        bulletIndent=4,
    )
    meta_style = ParagraphStyle(
        "Meta",
        fontName="Helvetica",
        fontSize=8,
        textColor=MID_GRAY,
        spaceAfter=2,
    )
    cite_style = ParagraphStyle(
        "Cite",
        fontName="Helvetica",
        fontSize=8,
        textColor=MID_GRAY,
        spaceAfter=3,
        leading=11,
    )
    footer_style = ParagraphStyle(
        "Footer",
        fontName="Helvetica",
        fontSize=7,
        textColor=MID_GRAY,
        alignment=TA_CENTER,
    )

    # ── Build content ──────────────────────────────────────────────────────────
    story = []

    # Masthead header band (simulated with a colored table)
    header_data = [[
        Paragraph("MicroPlastiNet", ParagraphStyle(
            "hdr", fontName="Helvetica-Bold", fontSize=20, textColor=WHITE)),
        Paragraph(
            f"Environmental Compliance Report<br/>"
            f"<font size='8' color='#94a3b8'>Georgia Coastal Watershed Monitoring Network</font>",
            ParagraphStyle("hdr2", fontName="Helvetica", fontSize=11, textColor=CYAN,
                           alignment=TA_RIGHT)),
    ]]
    header_table = Table(header_data, colWidths=["45%", "55%"])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DEEP_NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, -1), 14),
        ("RIGHTPADDING", (-1, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))

    # Meta row
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    meta_data = [[
        Paragraph(f"Station: <b>{station_id}</b>", meta_style),
        Paragraph(f"Event: <b>{event_id}</b>", meta_style),
        Paragraph(f"Generated: {now}", meta_style),
    ]]
    meta_table = Table(meta_data, colWidths=["33%", "33%", "34%"])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (0, -1), 10),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # ── Parse Markdown-ish text → ReportLab flowables ─────────────────────────
    lines = report_text.split("\n")
    in_list = False

    def flush_list():
        nonlocal in_list
        if in_list:
            story.append(Spacer(1, 4))
            in_list = False

    for line in lines:
        line_stripped = line.strip()

        # Skip the first H1 (we have the header table)
        if line_stripped.startswith("# ") and not line_stripped.startswith("## "):
            continue

        # Section headers
        if line_stripped.startswith("## "):
            flush_list()
            story.append(HRFlowable(width="100%", thickness=0.5, color=CYAN,
                                     spaceAfter=4, spaceBefore=12))
            story.append(Paragraph(line_stripped[3:], section_header_style))
            continue

        # Bold metadata lines (key: value)
        if line_stripped.startswith("**") and line_stripped.endswith("**"):
            flush_list()
            story.append(Paragraph(line_stripped[2:-2], bold_body_style))
            continue

        # Horizontal rule
        if line_stripped == "---":
            story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GRAY,
                                     spaceAfter=6, spaceBefore=6))
            continue

        # Bullet items
        if line_stripped.startswith("- "):
            in_list = True
            content = line_stripped[2:]
            # Inline bold
            content = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", content)
            story.append(Paragraph(f"• {content}", bullet_style))
            continue

        # Numbered items (1. 2. etc.)
        num_match = re.match(r"^(\d+)\.\s+(.+)$", line_stripped)
        if num_match:
            in_list = True
            num_text = num_match.group(2)
            num_text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", num_text)
            story.append(Paragraph(f"{num_match.group(1)}. {num_text}", bullet_style))
            continue

        # Bold/italic inline
        if line_stripped:
            flush_list()
            content = line_stripped
            content = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", content)
            content = re.sub(r"\*(.+?)\*", r"<i>\1</i>", content)

            # Citation lines (start with URL-like content)
            if "http" in content or content.startswith("- "):
                story.append(Paragraph(content, cite_style))
            else:
                story.append(Paragraph(content, body_style))
        else:
            flush_list()
            story.append(Spacer(1, 4))

    flush_list()

    # Footer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.3, color=MID_GRAY))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Generated by MicroPlastiNet — a research prototype by Saikumar Reddy Naidu (CS Graduate, Florida Atlantic University). "
        "Ongoing research; not affiliated with any agency.",
        footer_style,
    ))

    doc.build(story)
    print(f"PDF report saved: {output_path}")
    return output_path


# ─── CLI convenience ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from report_generator import generate_report

    event = {
        "station_id": "STN-003",
        "polymers": {"PE": 0.18, "PET": 0.45, "PP": 0.21, "PS": 0.08, "PVC": 0.05, "Other": 0.03},
        "confidence": {"PE": 0.89, "PET": 0.94, "PP": 0.82, "PS": 0.77, "PVC": 0.71, "Other": 0.65},
        "total_particles": 1847,
    }
    attr = {
        "station_id": "STN-003",
        "event_id": "EVT-7731",
        "event_date": "2025-06-15",
        "sources": [
            {"rank": 1, "name": "Upstream Wastewater Outfall", "probability": 0.52,
             "confidence": 0.88, "distance_km": 6.3, "lat": 32.31, "lon": -81.52},
            {"rank": 2, "name": "Urban Stormwater Runoff",    "probability": 0.26,
             "confidence": 0.81, "distance_km": 11.7, "lat": 32.41, "lon": -81.48},
            {"rank": 3, "name": "Agricultural Drainage",       "probability": 0.13,
             "confidence": 0.74, "distance_km": 18.2, "lat": 32.28, "lon": -81.61},
            {"rank": 4, "name": "Industrial Discharge",        "probability": 0.07,
             "confidence": 0.69, "distance_km": 24.5, "lat": 32.35, "lon": -81.39},
            {"rank": 5, "name": "Marine Vessel Traffic",       "probability": 0.02,
             "confidence": 0.55, "distance_km": 3.1, "lat": 32.18, "lon": -81.45},
        ],
    }

    report = generate_report("STN-003", event, attr, mode="template")
    assets_dir = Path(__file__).parent.parent.parent / "assets"
    export_markdown(report, assets_dir / "sample_report.md")
    export_pdf(report, assets_dir / "sample_report.pdf", station_id="STN-003", event_id="EVT-7731")
