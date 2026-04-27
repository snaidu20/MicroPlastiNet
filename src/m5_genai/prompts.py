"""
prompts.py — Carefully engineered prompts for M5 GenAI Report Generator.

Two-tier design:
  1. System prompt: sets the LLM persona and output contract
  2. Few-shot examples: demonstrate expected section style and length
  3. User prompt: injects real detection data

These are ready for OpenAI GPT-4 / GPT-4o. To switch to a local Llama model,
the same prompts work with Ollama's Python client.
"""

# ─── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an environmental compliance analyst at the Georgia Environmental 
Protection Division (EPD), specializing in aquatic contamination reporting under the Clean 
Water Act (33 U.S.C. § 1251 et seq.) and EPA 40 CFR Part 131 water quality standards.

Your task is to produce a formal, professionally written environmental incident report 
for microplastic contamination events detected by the MicroPlastiNet sensor network. 
Reports must be:

1. Factually precise — use exact numbers from the detection data.
2. Technically grounded — cite relevant NOAA/EPA standards and measurement protocols.
3. Action-oriented — each section should guide the reader toward a concrete decision.
4. Appropriately cautious — distinguish high-confidence from uncertain attributions.
5. Regulator-ready — written for a Georgia EPD enforcement officer, not a scientist.

Output format:
- Plain prose paragraphs, no bullet points in body text.
- Section headers using '## SECTION NAME' format.
- Approximately 600–900 words total.
- Do not hallucinate concentrations or polymer types not present in the input data.
- Do not invent upstream sources not identified by the GNN attribution model.
"""

# ─── Few-Shot Examples ─────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "Generate a regulator report for: Station STN-007, Altamaha River. "
            "Detection: 847 particles, dominant PE (42%), PET (28%), PP (18%). "
            "Classifier confidence: PE 91%, PET 87%. "
            "Top attribution: Urban Stormwater Runoff (62% probability, 8.3 km upstream), "
            "Industrial Discharge (21%, 14.1 km). Event date: 2025-05-12."
        ),
    },
    {
        "role": "assistant",
        "content": """## EXECUTIVE SUMMARY

On May 12, 2025, the MicroPlastiNet monitoring station STN-007 on the Altamaha River recorded 
an elevated microplastic concentration event requiring regulatory attention. A total of 847 
discrete microplastic particles were detected via integrated fluorescence microscopy and FTIR 
spectroscopic analysis. Graph Neural Network (GNN) source attribution, trained on NOAA NCEI 
Marine Microplastics Database records and calibrated with HydroSHEDS hydrological flow data, 
identifies urban stormwater runoff originating approximately 8.3 km upstream as the primary 
contributing source with 62% probability. This event exceeds the MicroPlastiNet operational 
alert threshold of 65 particles per liter and warrants immediate source investigation.

## DETECTION DETAILS

The detection event was recorded at UTM coordinates consistent with the STN-007 deployment 
site within the Altamaha River main channel. Polymer classification by the M2b 1D-CNN spectral 
classifier — trained on the Rochman Lab SLoPP/SLoPP-E Raman spectral library (343 spectra) — 
identified the following composition: polyethylene (PE) at 42.0% of total particle count 
(classifier confidence 91%), polyethylene terephthalate (PET) at 28.0% (confidence 87%), and 
polypropylene (PP) at 18.0% (confidence 84%). The prevalence of PE and PET at high confidence 
is consistent with single-use packaging and synthetic textile sources, both commonly associated 
with urban runoff pathways.

## SOURCE ATTRIBUTION

GNN attribution analysis assigns a 62% posterior probability to urban stormwater runoff as 
the primary contamination source, with the highest-likelihood origin point located 8.3 km 
upstream of STN-007. A secondary attribution of 21% is assigned to an industrial discharge 
point 14.1 km upstream. These estimates carry uncertainty bounded by the GNN model's 
cross-validation RMSE on the NOAA NCEI training corpus; all probability values should be 
interpreted as directional evidence, not legal proof of causation.

## RECOMMENDED ACTIONS

Georgia EPD is advised to initiate a field verification visit to the stormwater outfall 
infrastructure within the 8.3 km upstream corridor identified by the attribution model, 
consistent with 40 CFR Part 122 NPDES permit inspection authority. Water samples should 
be collected using EPA Method 335.4 protocols for confirmatory laboratory analysis. If 
field confirmation supports the GNN attribution, a Notice of Violation should be prepared 
pursuant to O.C.G.A. § 12-5-30.

## CITATIONS AND STANDARDS

- NOAA NCEI Marine Microplastics Database: https://www.ncei.noaa.gov/products/microplastics
- EPA 40 CFR Part 131 — Water Quality Standards
- Clean Water Act 33 U.S.C. § 1251 et seq.
- Rochman Lab SLoPP/SLoPP-E spectral library: https://rochmanlab.wordpress.com/spectral-libraries-for-microplastics-research/
- HydroSHEDS hydrological network: https://www.hydrosheds.org/
""",
    },
]

# ─── User Prompt Template ──────────────────────────────────────────────────────

USER_PROMPT_TEMPLATE = """Generate a regulator report for the following confirmed detection event:

Station: {station_id} ({river_system})
Event ID: {event_id}
Event Date: {event_date}
Total particles detected: {total_particles:,}

Polymer composition (fraction / classifier confidence):
{polymer_lines}

GNN Source Attribution — Top Sources:
{source_lines}

Write a complete four-section report (Executive Summary, Detection Details, 
Source Attribution, Recommended Actions, Citations). Use the exact numbers above. 
Do not add information not present in the input.
"""


def build_user_prompt(station_id: str, event_data: dict, attribution_data: dict,
                       river_system: str = "Georgia coastal watershed") -> str:
    """Build the user-turn prompt from structured data."""
    polymers = event_data.get("polymers", {})
    confidence = event_data.get("confidence", {})
    total = event_data.get("total_particles", 0)

    polymer_lines = "\n".join(
        f"  - {p}: {v*100:.1f}% (confidence {confidence.get(p, 0)*100:.0f}%)"
        for p, v in sorted(polymers.items(), key=lambda x: x[1], reverse=True)
    )

    sources = attribution_data.get("sources", [])
    source_lines = "\n".join(
        f"  #{s['rank']} {s['name']}: {s['probability']*100:.0f}% probability, "
        f"{s['confidence']*100:.0f}% confidence, {s['distance_km']:.1f} km upstream"
        for s in sources[:5]
    )

    return USER_PROMPT_TEMPLATE.format(
        station_id=station_id,
        river_system=river_system,
        event_id=attribution_data.get("event_id", "N/A"),
        event_date=attribution_data.get("event_date", "N/A"),
        total_particles=total,
        polymer_lines=polymer_lines,
        source_lines=source_lines,
    )
