# MicroPlastiNet Environmental Compliance Report

**Prepared by:** MicroPlastiNet Automated Monitoring System
**Author / Maintainer:** Saikumar Reddy Naidu — CS Graduate, Florida Atlantic University
**Project Status:** Research prototype — ongoing research
**Station:** STN-003 — Ogeechee Station 4
**Event ID:** EVT-7731
**Event Date:** 2025-06-15
**Report Generated:** 2026-04-25 21:52 UTC
**Classification:** HIGH PRIORITY — REGULATORY REVIEW REQUIRED

---

## EXECUTIVE SUMMARY

On 2025-06-15, the MicroPlastiNet sensor station STN-003, positioned along the Ogeechee river corridor in coastal Georgia, recorded a microplastic contamination event with a composite risk classification of **HIGH**. A total of **1,847** discrete microplastic particles were quantified via integrated fluorescence microscopy and Fourier-transform infrared (FTIR) spectroscopy, with a mean confidence across polymer classes of **79.7%**. The dominant polymer identified was **PET** (45.0% of total particle load), which is characteristically associated with beverage container and synthetic textile.

Source attribution analysis, performed using a Graph Attention Network (GAT) trained on the NOAA NCEI Marine Microplastics Database (~22,000 records) and calibrated with HydroSHEDS hydrological flow topology and ECMWF ERA5 atmospheric transport covariates, identifies **Upstream Wastewater Outfall** as the highest-probability upstream contributor (posterior probability: **52.0%**, model confidence: 88.0%, estimated transport distance: 6.3 km). Given the concentration levels and the model's attribution confidence, this event warrants immediate field verification and enforcement consideration under the Georgia Environmental Protection Division (EPD) regulatory framework.

## DETECTION DETAILS

The detection event was recorded at the STN-003 monitoring buoy (32.31, -81.52) during the normal 6-hour sampling cycle. Particle enumeration was performed using the M2a YOLOv8-based computer vision pipeline (EfficientNet-B0 backbone), with polymer identity confirmed by the M2b one-dimensional convolutional neural network (1D-CNN) spectral classifier, trained on the Rochman Lab SLoPP/SLoPP-E Raman spectral library (343 reference spectra) and FLOPP/FLOPP-e FTIR library (381 spectra).

**Polymer Composition by Fraction:**

- **PET:** 45.0% of detected particles (classifier confidence: 94.0%)
- **PP:** 21.0% of detected particles (classifier confidence: 82.0%)
- **PE:** 18.0% of detected particles (classifier confidence: 89.0%)
- **PS:** 8.0% of detected particles (classifier confidence: 77.0%)
- **PVC:** 5.0% of detected particles (classifier confidence: 71.0%)
- **Other:** 3.0% of detected particles (classifier confidence: 65.0%)


The predominance of PET (45.0%) and PP (21.0%) is consistent with a beverage container and synthetic textile contamination profile. Particle morphologies were distributed across fragment (est. 55%), fiber (est. 27%), and film (est. 18%) categories based on shape classification. Size distribution is consistent with secondary microplastics in the 100–500 μm range, indicating transport from upstream sources rather than local primary release.

## SOURCE ATTRIBUTION

The MicroPlastiNet Graph Neural Network (M3 GNN module) performed hydrological inversion across the sensor network graph to estimate upstream source contributions. The graph topology was constructed from HydroSHEDS Level-4 sub-basin delineations, with edge weights parameterized by USGS stream flow records and ECMWF ERA5 wind transport vectors.

**Top 5 Attributed Sources:**

1. **Upstream Wastewater Outfall** — Attribution probability: 52.0% | GNN confidence: 88.0% | Estimated distance: 6.3 km upstream
2. **Urban Stormwater Runoff** — Attribution probability: 26.0% | GNN confidence: 81.0% | Estimated distance: 11.7 km upstream
3. **Agricultural Drainage** — Attribution probability: 13.0% | GNN confidence: 74.0% | Estimated distance: 18.2 km upstream
4. **Industrial Discharge** — Attribution probability: 7.0% | GNN confidence: 69.0% | Estimated distance: 24.5 km upstream
5. **Marine Vessel Traffic** — Attribution probability: 2.0% | GNN confidence: 55.0% | Estimated distance: 3.1 km upstream


The highest-ranked source, Upstream Wastewater Outfall, carries a 52.0% posterior probability. This figure reflects both the hydrological connectivity of the candidate source node to the detection station and the polymer signature match between the observed distribution and known emission profiles for this source type.

**Important caveat:** GNN attribution probabilities represent statistical likelihoods derived from training data and should be treated as evidence to guide field investigation, not as legal proof of causation. Attribution confidence intervals were computed via MC-dropout (50 forward passes, α = 0.1).

## RECOMMENDED ACTIONS

**Immediate (0–72 hours):**
Initiate field inspection of the stormwater and drainage infrastructure within the 6.3-km upstream reach identified by the attribution model. Collect grab samples at the identified source node location in accordance with EPA Method 335.4 and Standard Method 2540D for suspended solids. Document GPS coordinates and photographic evidence of any visible discharge points.

**Short-term (1–4 weeks):**
If field samples confirm elevated microplastic concentrations consistent with the sensor readings, issue a preliminary notice of concern to the responsible facility or municipal stormwater authority under O.C.G.A. § 12-5-30. Increase MicroPlastiNet sampling frequency at STN-003 from 6-hour to 2-hour cycles for a minimum 14-day observation window.

**Regulatory:**
Should confirmed field measurements exceed the NOAA-recommended action threshold (50 particles/L in freshwater systems, per the NOAA Marine Debris Program monitoring framework), prepare a Notice of Violation under CWA Section 301. Coordinate with the Georgia EPD Watershed Protection Branch to assess cumulative loading across the Ogeechee sub-basin, referencing EPA 40 CFR Part 131 water quality standards.

## CITATIONS AND STANDARDS

- NOAA NCEI Marine Microplastics Database: https://www.ncei.noaa.gov/products/microplastics
- Rochman Lab Spectral Libraries (SLoPP/SLoPP-E, FLOPP/FLOPP-e): https://rochmanlab.wordpress.com/spectral-libraries-for-microplastics-research/
- HydroSHEDS Hydrological Data: https://www.hydrosheds.org/
- ERA5 Atmospheric Reanalysis (ECMWF): https://www.ecmwf.int/
- EPA 40 CFR Part 131 — Water Quality Standards
- Clean Water Act, 33 U.S.C. § 1251 et seq.
- Georgia EPD Water Protection Program: https://epd.georgia.gov/water
- NOAA Marine Debris Program Monitoring Protocol (2024)
- Kaggle Microplastic CV Dataset: https://www.kaggle.com/code/mathieuduverne/microplastic-detection-yolov8-map-50-76-2
