# Dehydration Risk Stratification — SNF Assessment Report

Interactive Streamlit dashboard that scores MDS 3.0 nursing home assessments for dehydration risk and produces a stakeholder-ready demand forecast.

## What It Does

- Reads de-identified MDS assessment data from VDP (Vitaline Data Platform)
- Scores each assessment using a **33-factor weighted risk model** (3/2/1 points by clinical severity)
- Classifies patients into **Low (0-5)**, **Moderate (6-10)**, **High (11+)**, or **Already Dehydrated**
- Computes per-facility demand estimates using rolling 6-month windows
- Presents results in a single-page interactive report

## Data

- OBRA 01-04 assessments (Jan 2022 onwards) from VDP Postgres
- All data is de-identified — surrogate patient and facility IDs only, no PHI
- MDS item set versions 1.17, 1.18, 1.19, 1.20

## Scoring Model

33 clinical risk factors extracted from MDS sections A through V:

| Weight | Points | Examples |
|--------|--------|----------|
| High | 3 pts | Comatose, fever, vomiting, hospice, IV fluids, CAA dehydration trigger, cognitive impairment (severe) |
| Moderate | 2 pts | Diabetes, CHF, renal failure, malnutrition, diuretics, delirium, feeding tube, weight loss |
| Low | 1 pt | Female sex, depression, pneumonia, stroke, dental problems, incontinence, pressure ulcers |

**MDS version handling:** Where CMS changed item codes between versions (e.g., eating ADL from Section G to GG, diuretics from N0400A to N0415G1), old and new fields are coalesced so no assessments are missed.

---

## Running Locally (live VDP data)

```bash
pip install -r requirements.txt
# Start Fly.io proxy tunnel in a separate terminal:
fly proxy 16380:5432 -a <app-name>
# Copy env template and fill in your DATABASE_URL:
cp .env.example .env
streamlit run app.py
```

The app reads live data from VDP Postgres. Cache refreshes every 5 minutes.

---

## Generating / Refreshing the Snapshot

The deployed (Streamlit Cloud) version reads from `dehydration_snapshot.parquet`.
To update it, run with the Fly tunnel active:

```bash
fly proxy 16380:5432 -a <app-name>
python export_snapshot.py
git add dehydration_snapshot.parquet
git commit -m "snapshot: refresh VDP data $(date +%Y-%m-%d)"
git push
```

Streamlit Community Cloud will automatically redeploy within ~1 minute of the push.

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repo, branch `main`, and main file `app.py`.
4. Click **Deploy**. No secrets needed — the app reads from `dehydration_snapshot.parquet` automatically when `DATABASE_URL` is not configured.

The app will be live at a URL like:
`https://<repo-name>-<hash>.streamlit.app/`

---

## Data Pipeline

| Step | Script | Environment | Output |
|------|--------|-------------|--------|
| Extract from VDP | `export_snapshot.py` | Local (Fly tunnel) | `dehydration_snapshot.parquet` |
| Score & visualise | `app.py` | Streamlit Cloud or local | Interactive dashboard |

---

## References

- Thomas DR et al. *JAMDA* 2008;9:292-301
- Bunn DK, Hooper L. *JAMDA* 2018
- Shimizu M et al. *Nutrients* 2020;12:3562
- CMS MDS 3.0 RAI User's Manual v1.20.1 (October 2025)
- CMS Form CMS-20092, Hydration Critical Element Pathway (2017)
