# Dehydration Risk Stratification — SNF Assessment Report

Interactive Streamlit dashboard that scores MDS 3.0 nursing home assessments for dehydration risk and produces a stakeholder-ready demand forecast.

## What It Does

- Reads de-identified MDS assessment data from a local SQLite database
- Scores each assessment using a **33-factor weighted risk model** (3/2/1 points by clinical severity)
- Classifies patients into **Low (0-5)**, **Moderate (6-10)**, **High (11+)**, or **Already Dehydrated**
- Computes per-facility demand estimates using rolling 6-month windows
- Presents results in a single-page interactive report

## Data

- **97,774 OBRA assessments** (Jan 2022 – Mar 2026)
- **29,023 unique patients** across **41 skilled nursing facilities**
- MDS item set versions 1.17, 1.18, 1.19, 1.20
- All data is de-identified (surrogate patient and facility IDs)

## Scoring Model

33 clinical risk factors extracted from MDS sections A through V:

| Weight | Points | Examples |
|--------|--------|----------|
| High | 3 pts | Comatose, fever, vomiting, hospice, IV fluids, CAA dehydration trigger, cognitive impairment (severe) |
| Moderate | 2 pts | Diabetes, CHF, renal failure, malnutrition, diuretics, delirium, feeding tube, weight loss |
| Low | 1 pt | Female sex, depression, pneumonia, stroke, dental problems, incontinence, pressure ulcers |

**MDS version handling:** Where CMS changed item codes between versions (e.g., eating ADL from Section G to GG, diuretics from N0400A to N0415G1), old and new fields are coalesced so no assessments are missed.

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires `dehydration_data.db` in the same directory (produced by `extract_to_sqlite.py` in the PHI environment).

## Data Pipeline

1. `extract_to_sqlite.py` — Run in the PHI environment. Connects to PostgreSQL, extracts relevant MDS items, joins to de-identified records, filters to OBRA 01-04 (2022+), writes `dehydration_data.db`
2. `app.py` — Reads SQLite, scores assessments, builds facility summaries, renders the Streamlit report

## References

- Thomas DR et al. *JAMDA* 2008;9:292-301
- Bunn DK, Hooper L. *JAMDA* 2018
- Shimizu M et al. *Nutrients* 2020;12:3562
- CMS MDS 3.0 RAI User's Manual v1.20.1 (October 2025)
- CMS Form CMS-20092, Hydration Critical Element Pathway (2017)
