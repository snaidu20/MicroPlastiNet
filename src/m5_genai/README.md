# M5 — MicroPlastiNet GenAI Report Generator

Automated generation of plain-English regulator reports from sensor + GNN outputs.
Supports offline template mode and GPT-4o mode.

---

## Modes

### Template Mode (offline, default)

Uses a Jinja2 template with dynamic data injection. Produces a polished ~800-word
report covering all required sections. Works with no API key.

```python
from report_generator import generate_report

report = generate_report(
    station_id="STN-003",
    event_data={
        "station_id": "STN-003",
        "polymers": {"PE": 0.18, "PET": 0.45, "PP": 0.21, "PS": 0.08, "PVC": 0.05, "Other": 0.03},
        "confidence": {"PE": 0.89, "PET": 0.94, "PP": 0.82, "PS": 0.77, "PVC": 0.71, "Other": 0.65},
        "total_particles": 1847,
    },
    attribution_data={
        "station_id": "STN-003",
        "event_id": "EVT-7731",
        "event_date": "2025-06-15",
        "sources": [
            {"rank": 1, "name": "Upstream Wastewater Outfall", "probability": 0.52,
             "confidence": 0.88, "distance_km": 6.3, "lat": 32.31, "lon": -81.52},
            # ...
        ],
    },
    mode="template",   # or "openai"
)
print(report)
```

### OpenAI GPT-4o Mode

To enable GPT-4o generation:

1. `pip install openai`
2. `export OPENAI_API_KEY=sk-...`
3. Uncomment the OpenAI block in `report_generator.py` (`_generate_openai()`)
4. Call with `mode="openai"`

The prompt system in `prompts.py` provides:
- **System prompt:** Sets GPD analyst persona, output contract, format rules
- **Few-shot examples:** One complete Q→A pair demonstrating the expected style
- **User prompt template:** Injects real data (particles, polymers, sources)

---

## Report Sections

1. **Executive Summary** — Risk classification, particle count, top polymer, primary source
2. **Detection Details** — Full polymer composition table, classifier confidence, particle morphology
3. **Source Attribution** — Top 5 GNN sources with probabilities and uncertainty statement
4. **Recommended Actions** — Immediate (0–72h), short-term (1–4 weeks), regulatory
5. **Citations** — NOAA, EPA, Rochman Lab, HydroSHEDS, CWA references

---

## Export

```python
from export import export_markdown, export_pdf

# Save as Markdown
export_markdown(report, "assets/my_report.md")

# Save as PDF (ReportLab, dark masthead, justified body text)
export_pdf(report, "assets/my_report.pdf", station_id="STN-003", event_id="EVT-7731")
```

---

## File Structure

```
src/m5_genai/
├── report_generator.py   ← Main generate_report() function
├── prompts.py            ← System prompt + few-shot examples + user template
├── schemas.py            ← Pydantic models (ReportInput, ReportOutput, etc.)
├── export.py             ← PDF + Markdown export
├── requirements.txt
└── README.md
```

---

## Sample Output

See `assets/sample_report.md` and `assets/sample_report.pdf` for a generated
report on a simulated contamination event at Ogeechee Station 3:

- **Station:** STN-003 (Ogeechee River)
- **Event:** PET-dominant spike (45% PET), 1,847 particles
- **Primary attribution:** Upstream Wastewater Outfall (52% probability, 6.3 km)

---

## Integration with M4 Dashboard

The M4 dashboard's **Reports** tab calls `generate_report()` directly when the
"Generate Report" button is clicked. Ensure `src/m5_genai/` is in the Python path
or installed as a package.
