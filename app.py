"""Dehydration Risk Stratification v2 — Stakeholder Report.

Launch:  streamlit run app.py
Reads:   VDP Postgres (Fly.io, requires fly proxy 16380:5432 running)
         Connects via DATABASE_URL in .env (see .env.example).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import db
from pathlib import Path

st.set_page_config(
    page_title="Dehydration Risk Report",
    page_icon="💧",
    layout="wide",
)

# ─── Constants ────────────────────────────────────────────────────────────────

TIER_COLORS = {
    "Low (0-5)": "#2ECC71",
    "Moderate (6-10)": "#F39C12",
    "High (11+)": "#E74C3C",
    "Already Dehydrated": "#8E44AD",
}
TIER_ORDER = ["Low (0-5)", "Moderate (6-10)", "High (11+)", "Already Dehydrated"]

FACTOR_META = {
    "pts_cognitive":        ("Cognitive Impairment",               "C0500 / C1000",          "3 or 2", "high"),
    "pts_dehydrated":       ("Already Dehydrated",                 "J1550C",                 "3",      "high"),
    "pts_comatose":         ("Comatose",                           "B0100",                  "3",      "high"),
    "pts_dysphagia":        ("Swallowing Disorder",                "K0100A-D",               "3",      "high"),
    "pts_fever":            ("Fever",                              "J1550A",                 "3",      "high"),
    "pts_vomiting":         ("Vomiting",                           "J1550B",                 "3",      "high"),
    "pts_hospice":          ("Hospice",                            "O0100K2",                "3",      "high"),
    "pts_iv_parenteral":    ("Parenteral / IV Fluids",             "K0520A1/A2/A3",          "3",      "high"),
    "pts_caa_dehydration":  ("CAA Dehydration Trigger",            "V0200A14A",              "3",      "high"),
    "pts_malnutrition":     ("Malnutrition",                       "I5600",                  "2",      "moderate"),
    "pts_diabetes":         ("Diabetes Mellitus",                  "I0600",                  "2",      "moderate"),
    "pts_chf":              ("Heart Failure (CHF)",                "I4000",                  "2",      "moderate"),
    "pts_renal":            ("Renal Insufficiency",                "I1500",                  "2",      "moderate"),
    "pts_diuretics":        ("Diuretics",                          "N0415G1",                "2",      "moderate"),
    "pts_weight_loss":      ("Weight Loss",                        "K0300",                  "2",      "moderate"),
    "pts_feeding_tube":     ("Feeding Tube",                       "K0520B1/B2/B3",          "2",      "moderate"),
    "pts_adl_eating":       ("Eating ADL Dependence",              "G0110H1 / GG0130A1",    "2",      "moderate"),
    "pts_delirium":         ("Delirium Signs",                     "C1310A-D",               "2",      "moderate"),
    "pts_uti":              ("UTI",                                "I2300",                  "2",      "moderate"),
    "pts_antipsychotics":   ("Antipsychotics",                     "N0415A1",                "2",      "moderate"),
    "pts_caa_nutritional":  ("CAA Nutritional Trigger",            "V0200A12A",              "2",      "moderate"),
    "pts_incontinence_mod": ("Always Incontinent",                 "H0300 = 3",              "2",      "moderate"),
    "pts_female":           ("Female Sex",                         "A0800",                  "1",      "low"),
    "pts_depression":       ("Depression",                         "D0160 / D0600 / I6000",  "1",      "low"),
    "pts_pneumonia":        ("Pneumonia",                          "I2000",                  "1",      "low"),
    "pts_stroke":           ("Stroke / CVA",                       "I4900",                  "1",      "low"),
    "pts_dental":           ("Dental Problem",                     "L0200D",                 "1",      "low"),
    "pts_communication":    ("Communication Impairment",           "B0700",                  "1",      "low"),
    "pts_incontinence_low": ("Occasionally / Frequently Incontinent", "H0300 = 1, 2",       "1",      "low"),
    "pts_dialysis":         ("Dialysis",                           "O0100J2",                "1",      "low"),
    "pts_dementia_dx":      ("Alzheimer's / Dementia Dx",          "I5250 / I5300",          "1",      "low"),
    "pts_constipation":     ("Constipation",                       "H0600",                  "1",      "low"),
    "pts_pressure_ulcer":   ("Pressure Ulcers Present",            "M0210",                  "1",      "low"),
}


# ─── Scoring ──────────────────────────────────────────────────────────────────

def score_assessments(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the v2 dehydration risk scoring model to raw MDS data."""

    num_cols = [
        "a0800", "b0100", "b0700",
        "c0100", "c0500", "c1000",
        "c1310a", "c1310b", "c1310c", "c1310d",
        "d0160", "d0600",
        "g0110h1", "h0300", "h0600",
        "i0600", "i1500", "i2000", "i2300", "i4000", "i4900",
        "i5250", "i5300", "i5600", "i6000",
        "j1550a", "j1550b", "j1550c",
        "k0100a", "k0100b", "k0100c", "k0100d",
        "k0300",
        "k0520a1", "k0520a2", "k0520a3",
        "k0520b1", "k0520b2", "k0520b3",
        "l0200d", "m0210",
        "n0415a1", "n0415g1",
        "o0100j2", "o0100k2",
        "v0200a14a", "v0200a12a",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Cognitive impairment (BIMS / C1000 skip pattern)
    bims = df["c0500"]
    c1000 = df["c1000"]
    bims_valid = bims.notna() & (bims != 99)
    cog = pd.Series(0, index=df.index, dtype="int")
    cog = np.where(bims_valid & (bims <= 7), 3, cog)
    cog = np.where(bims_valid & (bims >= 8) & (bims <= 12), 2, cog)
    cog = np.where(~bims_valid & (c1000 == 3), 3, cog)
    cog = np.where(~bims_valid & (c1000 == 2), 2, cog)
    df["pts_cognitive"] = cog

    # HIGH-WEIGHT (3 points)
    df["pts_dehydrated"]      = np.where(df["j1550c"] == 1, 3, 0)
    df["pts_comatose"]        = np.where(df["b0100"] == 1, 3, 0)
    df["pts_dysphagia"]       = np.where(
        (df["k0100a"] == 1) | (df["k0100b"] == 1) |
        (df["k0100c"] == 1) | (df["k0100d"] == 1), 3, 0)
    df["pts_fever"]           = np.where(df["j1550a"] == 1, 3, 0)
    df["pts_vomiting"]        = np.where(df["j1550b"] == 1, 3, 0)
    df["pts_hospice"]         = np.where(df["o0100k2"] == 1, 3, 0)
    df["pts_iv_parenteral"]   = np.where(
        (df["k0520a1"] == 1) | (df["k0520a2"] == 1) | (df["k0520a3"] == 1), 3, 0)
    df["pts_caa_dehydration"] = np.where(df["v0200a14a"] == 1, 3, 0)

    # MODERATE-WEIGHT (2 points)
    df["pts_malnutrition"]    = np.where(df["i5600"] == 1, 2, 0)
    df["pts_diabetes"]        = np.where(df["i0600"] == 1, 2, 0)
    df["pts_chf"]             = np.where(df["i4000"] == 1, 2, 0)
    df["pts_renal"]           = np.where(df["i1500"] == 1, 2, 0)
    df["pts_diuretics"]       = np.where(df["n0415g1"] == 1, 2, 0)
    df["pts_weight_loss"]     = np.where(df["k0300"].isin([1, 2]), 2, 0)
    df["pts_feeding_tube"]    = np.where(
        (df["k0520b1"] == 1) | (df["k0520b2"] == 1) | (df["k0520b3"] == 1), 2, 0)

    # Eating ADL: coalesce old G0110H1 with new GG0130A1
    old_eat = df["g0110h1"].isin([3, 4])
    new_eat = df["gg0130a1"].isin(["01", "02", "88"])
    df["pts_adl_eating"] = np.where(old_eat | new_eat, 2, 0)

    df["pts_delirium"] = np.where(
        (df["c1310a"] == 1) |
        (df["c1310b"].isin([1, 2])) |
        (df["c1310c"].isin([1, 2])) |
        (df["c1310d"].isin([1, 2])), 2, 0)
    df["pts_uti"]              = np.where(df["i2300"] == 1, 2, 0)
    df["pts_antipsychotics"]   = np.where(df["n0415a1"] == 1, 2, 0)
    df["pts_caa_nutritional"]  = np.where(df["v0200a12a"] == 1, 2, 0)
    df["pts_incontinence_mod"] = np.where(df["h0300"] == 3, 2, 0)

    # LOW-WEIGHT (1 point)
    df["pts_female"]           = np.where(df["a0800"] == 2, 1, 0)
    df["pts_depression"]       = np.where(
        (df["d0160"].fillna(0) >= 10) |
        (df["d0600"].fillna(0) >= 10) |
        (df["i6000"] == 1), 1, 0)
    df["pts_pneumonia"]        = np.where(df["i2000"] == 1, 1, 0)
    df["pts_stroke"]           = np.where(df["i4900"] == 1, 1, 0)
    df["pts_dental"]           = np.where(df["l0200d"] == 1, 1, 0)
    df["pts_communication"]    = np.where(df["b0700"].isin([2, 3]), 1, 0)
    df["pts_incontinence_low"] = np.where(df["h0300"].isin([1, 2]), 1, 0)
    df["pts_dialysis"]         = np.where(df["o0100j2"] == 1, 1, 0)
    df["pts_dementia_dx"]      = np.where(
        (df["i5250"] == 1) | (df["i5300"] == 1), 1, 0)
    df["pts_constipation"]     = np.where(df["h0600"] == 1, 1, 0)
    df["pts_pressure_ulcer"]   = np.where(df["m0210"] == 1, 1, 0)

    # Sum + tier
    pts_cols = [c for c in df.columns if c.startswith("pts_")]
    df["risk_score"] = df[pts_cols].sum(axis=1)
    df["risk_tier"] = pd.cut(
        df["risk_score"], bins=[-1, 5, 10, 999],
        labels=["Low (0-5)", "Moderate (6-10)", "High (11+)"],
    ).astype(str)
    df.loc[df["j1550c"] == 1, "risk_tier"] = "Already Dehydrated"

    return df


# ─── Rolling-window facility summary ─────────────────────────────────────────

def build_facility_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-facility averages over rolling 6-month windows."""

    df["ard_date"] = pd.to_datetime(df["assessment_reference_date"], errors="coerce")
    df = df.dropna(subset=["ard_date"])

    data_start = df["ard_date"].min()
    data_end = df["ard_date"].max()

    window_starts = pd.date_range(
        start=pd.Timestamp(data_start.year, data_start.month, 1),
        end=data_end, freq="QS",
    )
    windows = []
    for ws in window_starts:
        we = ws + pd.DateOffset(months=6) - pd.DateOffset(days=1)
        if ws >= data_start.normalize() - pd.DateOffset(days=data_start.day - 1) and we <= data_end:
            windows.append((ws, we))

    all_results = []
    for ws, we in windows:
        wl = f"{ws.strftime('%b%y')}-{we.strftime('%b%y')}"
        mask = (df["ard_date"] >= ws) & (df["ard_date"] <= we)
        dw = df[mask]

        ac = (dw.groupby(["surrogate_facility_id", "risk_tier"], observed=True)
              .size().reset_index(name="assessment_count"))

        dl = (dw.sort_values("ard_date")
              .groupby(["surrogate_facility_id", "surrogate_patient_id"])
              .last().reset_index())
        pc = (dl.groupby(["surrogate_facility_id", "risk_tier"], observed=True)
              ["surrogate_patient_id"].nunique().reset_index(name="unique_patients"))

        tp = (dw.groupby("surrogate_facility_id")["surrogate_patient_id"]
              .nunique().reset_index(name="total_unique_patients"))

        merged = ac.merge(pc, on=["surrogate_facility_id", "risk_tier"], how="outer").fillna(0)
        merged["window"] = wl
        merged = merged.merge(tp, on="surrogate_facility_id", how="left").fillna(0)
        for c in ["assessment_count", "unique_patients", "total_unique_patients"]:
            merged[c] = merged[c].astype(int)
        all_results.append(merged)

    agg = pd.concat(all_results, ignore_index=True)

    n_win = agg.groupby("surrogate_facility_id")["window"].nunique()
    avg_tier = (agg.groupby(["surrogate_facility_id", "risk_tier"], observed=True)
                .agg(sum_a=("assessment_count", "sum"), sum_p=("unique_patients", "sum"))
                .reset_index())
    avg_tier = avg_tier.merge(n_win.reset_index().rename(columns={"window": "n_win"}),
                              on="surrogate_facility_id")
    avg_tier["avg_assessments"] = (avg_tier["sum_a"] / avg_tier["n_win"]).round(1)
    avg_tier["avg_patients"] = (avg_tier["sum_p"] / avg_tier["n_win"]).round(1)

    fac_win = (agg.groupby(["surrogate_facility_id", "window"])
               .agg(total_a=("assessment_count", "sum"),
                    total_p=("total_unique_patients", "first"))
               .reset_index())
    fac_summary = (fac_win.groupby("surrogate_facility_id")
                   .agg(n_windows=("window", "nunique"),
                        avg_total_assessments=("total_a", "mean"),
                        avg_total_patients=("total_p", "mean"))
                   .reset_index())
    for c in ["avg_total_assessments", "avg_total_patients"]:
        fac_summary[c] = fac_summary[c].round(1)

    tier_order = ["Low", "Moderate", "High", "Already"]
    rows = []
    for fac in sorted(fac_summary["surrogate_facility_id"].unique()):
        fs = fac_summary[fac_summary["surrogate_facility_id"] == fac].iloc[0]
        ft = avg_tier[avg_tier["surrogate_facility_id"] == fac]
        r = {"Facility": fac}
        for tier_full, tk in [
            ("Low (0-5)", "Low"), ("Moderate (6-10)", "Moderate"),
            ("High (11+)", "High"), ("Already Dehydrated", "Already"),
        ]:
            tr = ft[ft["risk_tier"] == tier_full]
            r[f"#Assess {tk}"] = tr["avg_assessments"].values[0] if len(tr) else 0.0
            r[f"#Patients {tk}"] = tr["avg_patients"].values[0] if len(tr) else 0.0
        r["#Assessments"] = fs["avg_total_assessments"]
        r["#Unique Patients"] = fs["avg_total_patients"]
        for tk_full, tk in [("Moderate", "Moderate"), ("High", "High"), ("Already", "Already")]:
            r[f"% {tk_full} Patients"] = (
                round(r[f"#Patients {tk}"] / r["#Unique Patients"] * 100, 1)
                if r["#Unique Patients"] else 0
            )
        rows.append(r)

    avg_r = {"Facility": "AVG"}
    for col in [c for c in rows[0] if c != "Facility"]:
        avg_r[col] = round(np.mean([r[col] for r in rows]), 1)
    rows.append(avg_r)

    return pd.DataFrame(rows)


# ─── Data loading & caching ──────────────────────────────────────────────────

# SQL query: extract all MDS item codes used by the scoring model from deid_json.
# assessment_type_obra is stored as decoded strings like "01 (Admission)" in VDP;
# we filter by the leading 2-char code and also expose the raw code for display.
_LOAD_QUERY = """
SELECT
    ma.mds_pid                                           AS surrogate_patient_id,
    COALESCE(ma.pid, ma.mds_pid)                         AS surrogate_patient_id_matched,
    ma.fid                                               AS surrogate_facility_id,
    ma.cid,
    ma.assessment_reference_date,
    LEFT(ma.assessment_type_obra, 2)                     AS assessment_type_obra,
    ma.deid_json->>'ITM_SET_VRSN_CD'                     AS item_set_version,

    -- Section A
    ma.deid_json->>'A0800'                               AS a0800,

    -- Section B
    ma.deid_json->>'B0100'                               AS b0100,
    ma.deid_json->>'B0700'                               AS b0700,

    -- Section C
    ma.deid_json->>'C0100'                               AS c0100,
    ma.deid_json->>'C0500'                               AS c0500,
    ma.deid_json->>'C1000'                               AS c1000,
    ma.deid_json->>'C1310A'                              AS c1310a,
    ma.deid_json->>'C1310B'                              AS c1310b,
    ma.deid_json->>'C1310C'                              AS c1310c,
    ma.deid_json->>'C1310D'                              AS c1310d,

    -- Section D
    ma.deid_json->>'D0160'                               AS d0160,
    ma.deid_json->>'D0600'                               AS d0600,

    -- Section G (old eating ADL)
    ma.deid_json->>'G0110H1'                             AS g0110h1,

    -- Section GG (new eating ADL)
    ma.deid_json->>'GG0130A1'                            AS gg0130a1,

    -- Section H
    ma.deid_json->>'H0300'                               AS h0300,
    ma.deid_json->>'H0600'                               AS h0600,

    -- Section I
    ma.deid_json->>'I0600'                               AS i0600,
    ma.deid_json->>'I1500'                               AS i1500,
    ma.deid_json->>'I2000'                               AS i2000,
    ma.deid_json->>'I2300'                               AS i2300,
    ma.deid_json->>'I4000'                               AS i4000,
    ma.deid_json->>'I4900'                               AS i4900,
    ma.deid_json->>'I5250'                               AS i5250,
    ma.deid_json->>'I5300'                               AS i5300,
    ma.deid_json->>'I5600'                               AS i5600,
    ma.deid_json->>'I6000'                               AS i6000,

    -- Section J
    ma.deid_json->>'J1550A'                              AS j1550a,
    ma.deid_json->>'J1550B'                              AS j1550b,
    ma.deid_json->>'J1550C'                              AS j1550c,

    -- Section K
    ma.deid_json->>'K0100A'                              AS k0100a,
    ma.deid_json->>'K0100B'                              AS k0100b,
    ma.deid_json->>'K0100C'                              AS k0100c,
    ma.deid_json->>'K0100D'                              AS k0100d,
    ma.deid_json->>'K0300'                               AS k0300,
    ma.deid_json->>'K0520A1'                             AS k0520a1,
    ma.deid_json->>'K0520A2'                             AS k0520a2,
    ma.deid_json->>'K0520A3'                             AS k0520a3,
    ma.deid_json->>'K0520B1'                             AS k0520b1,
    ma.deid_json->>'K0520B2'                             AS k0520b2,
    ma.deid_json->>'K0520B3'                             AS k0520b3,

    -- Section L
    ma.deid_json->>'L0200D'                              AS l0200d,

    -- Section M
    ma.deid_json->>'M0210'                               AS m0210,

    -- Section N
    ma.deid_json->>'N0415A1'                             AS n0415a1,
    ma.deid_json->>'N0415G1'                             AS n0415g1,

    -- Section O
    ma.deid_json->>'O0100J2'                             AS o0100j2,
    ma.deid_json->>'O0100K2'                             AS o0100k2,

    -- Section V (CAA triggers)
    ma.deid_json->>'V0200A12A'                           AS v0200a12a,
    ma.deid_json->>'V0200A14A'                           AS v0200a14a

FROM app.mds_assessments ma
WHERE LEFT(ma.assessment_type_obra, 2) IN ('01', '02', '03', '04')
  AND ma.assessment_reference_date >= '2022-01-01'
  AND ma.fid IS NOT NULL
"""


_SNAPSHOT_PATH = Path(__file__).parent / "dehydration_snapshot.parquet"


@st.cache_data(ttl=300)
def load_and_score():
    """Load OBRA assessments, score, and summarise.

    Data source priority:
      1. Live VDP Postgres  — when DATABASE_URL is set in .env (local dev with tunnel)
      2. dehydration_snapshot.parquet — static snapshot for Streamlit Community Cloud
    """
    import os

    use_live = bool(os.getenv("DATABASE_URL", ""))

    if use_live:
        try:
            conn = db.get_connection()
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(
                f"Could not connect to VDP Postgres. "
                f"Is the Fly tunnel running on port 16380?\n\n`{exc}`"
            )
            st.stop()
        try:
            with conn.cursor() as cur:
                cur.execute(_LOAD_QUERY)
                cols = [desc[0] for desc in cur.description]
                df = pd.DataFrame(cur.fetchall(), columns=cols)
        finally:
            conn.close()
    else:
        if not _SNAPSHOT_PATH.exists():
            st.error(
                f"`{_SNAPSHOT_PATH.name}` not found and `DATABASE_URL` is not set. "
                f"Either:\n"
                f"- Run `python export_snapshot.py` (with Fly tunnel) to generate a snapshot, or\n"
                f"- Create a `.env` file with `DATABASE_URL` for live data."
            )
            st.stop()
        df = pd.read_parquet(_SNAPSHOT_PATH)

    df = score_assessments(df)
    summary = build_facility_summary(df)
    return df, summary


scores, summary = load_and_score()
pts_cols = sorted([c for c in scores.columns if c.startswith("pts_")])
fac_rows = summary[summary["Facility"] != "AVG"]
avg = summary[summary["Facility"] == "AVG"].iloc[0]
n_fac = len(fac_rows)
n_assess = len(scores)
n_patients = scores["surrogate_patient_id"].nunique()
versions = sorted(scores["item_set_version"].dropna().unique())

# Patient-level tier counts (most recent assessment per patient)
_ard = pd.to_datetime(scores["assessment_reference_date"], errors="coerce")
_latest = scores.assign(_ard=_ard).sort_values("_ard").groupby("surrogate_patient_id").last()
patient_tier_counts = _latest["risk_tier"].value_counts()
patient_tier_pcts = (patient_tier_counts / n_patients * 100).round(1)


# ═════════════════════════════════════════════════════════════════════════════
# STYLE
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.3rem !important; margin-top: 1.5rem !important; }
    h3 { font-size: 1.1rem !important; }
    .caption-text {
        color: #666; font-size: 0.85rem; line-height: 1.4;
        margin-top: -0.5rem; margin-bottom: 1rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 8px; padding: 1.2rem;
        text-align: center; border: 1px solid #e9ecef;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; margin: 0; }
    .metric-label { font-size: 0.8rem; color: #666; margin: 0; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR — data overview
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Data Overview")
    st.markdown(f"**Assessments:** {n_assess:,}")
    st.markdown(f"**Unique patients:** {n_patients:,}")
    st.markdown(f"**Facilities:** {n_fac}")
    ard = pd.to_datetime(scores["assessment_reference_date"], errors="coerce")
    st.markdown(f"**Period:** {ard.min().strftime('%b %Y')} — {ard.max().strftime('%b %Y')}")
    st.markdown(f"**MDS versions:** {', '.join(str(v) for v in versions)}")
    st.markdown("---")
    st.markdown("### Patient Risk Tiers")
    st.caption("Based on each patient's most recent assessment")
    for tier in TIER_ORDER:
        n = patient_tier_counts.get(tier, 0)
        pct = patient_tier_pcts.get(tier, 0)
        color = TIER_COLORS[tier]
        st.markdown(f"<span style='color:{color}'>●</span> **{tier}**: {n:,} ({pct}%)",
                    unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<div style='color:#999;font-size:0.75rem'>De-identified · No PHI · "
        "MDS 3.0 RAI Manual v1.20.1</div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════

st.title("Dehydration Risk Assessment Report")
st.markdown(
    f"Analysis of **{n_fac} skilled nursing facilities** using "
    f"**{n_assess:,} MDS 3.0 assessments** ({n_patients:,} unique patients, "
    f"Jan 2022+, item set versions {', '.join(str(v) for v in versions)})."
)
st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# KEY METRICS
# ═════════════════════════════════════════════════════════════════════════════

st.header("Key Findings")

c1, c2, c3, c4, c5 = st.columns(5)
pct_low_patients = 100 - avg["% Moderate Patients"] - avg["% High Patients"] - avg.get("% Already Patients", 0)
for col, label, value, color in [
    (c1, "Patients / Facility", f"{avg['#Unique Patients']:.0f}", "#2C3E50"),
    (c2, "Low Risk (0-5)", f"{avg['#Patients Low']:.0f} ({pct_low_patients:.0f}%)", TIER_COLORS["Low (0-5)"]),
    (c3, "Moderate (6-10)", f"{avg['#Patients Moderate']:.0f} ({avg['% Moderate Patients']:.1f}%)", TIER_COLORS["Moderate (6-10)"]),
    (c4, "High Risk (11+)", f"{avg['#Patients High']:.0f} ({avg['% High Patients']:.1f}%)", TIER_COLORS["High (11+)"]),
    (c5, "Already Dehydrated", f"{avg['#Patients Already']:.1f} ({avg['% Already Patients']:.1f}%)", TIER_COLORS["Already Dehydrated"]),
]:
    col.markdown(f"""
    <div class="metric-card">
        <p class="metric-label">{label}</p>
        <p class="metric-value" style="color:{color}">{value}</p>
        <p class="metric-label">avg per facility / 6-mo</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

total_target = avg["#Patients Moderate"] + avg["#Patients High"] + avg["#Patients Already"]
st.info(
    f"**Demand Forecast:** Per facility in a typical 6-month period, "
    f"about **{avg['#Patients Moderate']:.0f}** moderate-risk + **{avg['#Patients High']:.0f}** high-risk + "
    f"**{avg['#Patients Already']:.1f}** actively dehydrated = "
    f"**{total_target:.0f} addressable patients**. "
    f"Across {n_fac} facilities, that is approximately **{total_target * n_fac:,.0f} target patients** per 6-month cycle."
)

st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# 1. SCORE HISTOGRAM
# ═════════════════════════════════════════════════════════════════════════════

st.header("1. Score Distribution")

score_dist = scores["risk_score"].value_counts().sort_index().reset_index()
score_dist.columns = ["Score", "Count"]
score_dist["Tier"] = pd.cut(
    score_dist["Score"], bins=[-1, 5, 10, 999],
    labels=["Low (0-5)", "Moderate (6-10)", "High (11+)"],
).astype(str)

fig_hist = px.bar(
    score_dist, x="Score", y="Count", color="Tier",
    color_discrete_map=TIER_COLORS,
    category_orders={"Tier": TIER_ORDER[:3]},
)
fig_hist.update_layout(
    height=380, margin=dict(t=20, b=40),
    xaxis_title="Risk Score", yaxis_title="Assessments",
    legend=dict(orientation="h", yanchor="top", y=1.12, xanchor="center", x=0.5, title=None),
    plot_bgcolor="white",
)
fig_hist.update_xaxes(showgrid=False)
fig_hist.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
st.plotly_chart(fig_hist, use_container_width=True, key="hist")

st.markdown(f"""<div class="caption-text">
<b>How to read:</b> Each bar is a single score value (0 through {int(scores['risk_score'].max())}). Color shows which
risk tier that score falls into —
<span style="color:{TIER_COLORS['Low (0-5)']}"><b>green = Low (0-5)</b></span>,
<span style="color:{TIER_COLORS['Moderate (6-10)']}"><b>orange = Moderate (6-10)</b></span>,
<span style="color:{TIER_COLORS['High (11+)']}"><b>red = High (11+)</b></span>.
By unique patients (most recent assessment): <b>{patient_tier_pcts.get('Low (0-5)', 0)}%</b> Low,
<b>{patient_tier_pcts.get('Moderate (6-10)', 0)}%</b> Moderate,
<b>{patient_tier_pcts.get('High (11+)', 0)}%</b> High,
<b>{patient_tier_pcts.get('Already Dehydrated', 0)}%</b> Already Dehydrated.
Thresholds were calibrated to avoid overestimating high-risk in this high-acuity SNF population.
</div>""", unsafe_allow_html=True)

st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# 2. RISK FACTOR PREVALENCE
# ═════════════════════════════════════════════════════════════════════════════

st.header("2. Risk Factor Prevalence")

factor_data = []
for col in pts_cols:
    if col in FACTOR_META:
        name, mds, pts, weight = FACTOR_META[col]
    else:
        name = col.replace("pts_", "").replace("_", " ").title()
        mds, pts, weight = "—", "?", "low"
    fired = (scores[col] > 0).sum()
    pct = fired / n_assess * 100
    wlabel = "High (3 pts)" if weight == "high" else ("Moderate (2 pts)" if weight == "moderate" else "Low (1 pt)")
    factor_data.append({"Factor": name, "MDS Code": mds, "Points": pts, "Prevalence": pct, "Weight": wlabel})

fdf = pd.DataFrame(factor_data).sort_values("Prevalence", ascending=True)

weight_colors = {"High (3 pts)": "#E74C3C", "Moderate (2 pts)": "#F39C12", "Low (1 pt)": "#2ECC71"}
fig_prev = px.bar(
    fdf, x="Prevalence", y="Factor", orientation="h",
    color="Weight", color_discrete_map=weight_colors,
    hover_data=["MDS Code", "Points"],
    text="Prevalence",
    category_orders={"Weight": ["High (3 pts)", "Moderate (2 pts)", "Low (1 pt)"]},
)
fig_prev.update_traces(texttemplate="%{text:.1f}%", textposition="outside", textfont_size=10)
fig_prev.update_layout(
    height=750, margin=dict(t=20, b=40, l=200),
    xaxis_title="% of assessments where factor is present",
    yaxis=dict(categoryorder="total ascending", title=None),
    legend=dict(orientation="h", yanchor="top", y=1.04, xanchor="center", x=0.5, title=None),
    plot_bgcolor="white",
)
fig_prev.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
fig_prev.update_yaxes(showgrid=False)
st.plotly_chart(fig_prev, use_container_width=True)

# dynamic top-2 factors
top2 = fdf.nlargest(2, "Prevalence")
top_text = " and ".join(f"{r['Factor']} ({r['Prevalence']:.0f}%)" for _, r in top2.iterrows())
st.markdown(f"""<div class="caption-text">
<b>How to read:</b> Each bar shows what percentage of all assessments have that risk factor present.
Color indicates point weight —
<span style="color:#E74C3C"><b>red = 3 pts (high-weight)</b></span>,
<span style="color:#F39C12"><b>orange = 2 pts (moderate)</b></span>,
<span style="color:#2ECC71"><b>green = 1 pt (low)</b></span>.
{top_text} are the most common.
Acute factors like fever, vomiting, and comatose are rare in stable SNF residents but carry heavy weight.
A factor's total impact depends on both its prevalence and its point weight.
</div>""", unsafe_allow_html=True)

st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# 3. CO-OCCURRING RISK FACTORS (Dehydrated vs Others)
# ═════════════════════════════════════════════════════════════════════════════

st.header("3. Profile of Already-Dehydrated Patients")

dehyd = scores[scores["risk_tier"] == "Already Dehydrated"]
others = scores[scores["risk_tier"] != "Already Dehydrated"]

cooccur = []
for col in pts_cols:
    if col == "pts_dehydrated":
        continue
    if col in FACTOR_META:
        name = FACTOR_META[col][0]
    else:
        name = col.replace("pts_", "").replace("_", " ").title()
    rd = (dehyd[col] > 0).sum() / len(dehyd) * 100 if len(dehyd) > 0 else 0
    ro = (others[col] > 0).sum() / len(others) * 100 if len(others) > 0 else 0
    if rd > 15:
        cooccur.append({"Factor": name, "Dehydrated": rd, "Others": ro})

co_df = pd.DataFrame(cooccur).sort_values("Dehydrated", ascending=True)

fig_co = go.Figure()
fig_co.add_trace(go.Bar(
    name="Already Dehydrated", y=co_df["Factor"], x=co_df["Dehydrated"],
    orientation="h", marker_color=TIER_COLORS["Already Dehydrated"],
    text=co_df["Dehydrated"].round(1).astype(str) + "%", textposition="outside", textfont_size=10,
))
fig_co.add_trace(go.Bar(
    name="All Others", y=co_df["Factor"], x=co_df["Others"],
    orientation="h", marker_color="#BDC3C7",
    text=co_df["Others"].round(1).astype(str) + "%", textposition="outside", textfont_size=10,
))
fig_co.update_layout(
    barmode="group", height=max(380, len(co_df) * 35),
    margin=dict(t=20, b=40, l=200),
    xaxis_title="Prevalence %",
    yaxis=dict(title=None),
    legend=dict(orientation="h", yanchor="top", y=1.08, xanchor="center", x=0.5),
    plot_bgcolor="white",
)
fig_co.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
fig_co.update_yaxes(showgrid=False)
st.plotly_chart(fig_co, use_container_width=True)

st.markdown(f"""<div class="caption-text">
<b>How to read:</b> Purple bars show how often each factor appears in the <b>{len(dehyd):,}</b>
already-dehydrated patients (J1550C = 1). Gray bars show the same factor's prevalence in everyone else.
Where purple extends well beyond gray, that factor is disproportionately associated with dehydration.
The pattern shows multi-system frailty — cognitive impairment, incontinence, malnutrition, and CAA triggers
all cluster together in dehydrated patients, validating the composite scoring approach.
</div>""", unsafe_allow_html=True)

st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# 4. HOW IT WORKS — STEP BY STEP
# ═════════════════════════════════════════════════════════════════════════════

st.header("4. How This Analysis Works")

st.markdown("Expand each step below to see the details.")

with st.expander("Step 1 — Data Source: MDS 3.0 assessments from CMS-mandated nursing home evaluations"):
    st.markdown("""
**What:** MDS 3.0 (Minimum Data Set) assessments from CMS-mandated standardized nursing home evaluations.

**Where:** Extracted from the facility management database (PostgreSQL). Each assessment contains hundreds
of coded clinical items stored as structured JSON.

**De-identification:** All patient and facility identifiers are replaced with surrogate IDs before analysis.
No PHI leaves the source environment.
""")

with st.expander(f"Step 2 — Filtering: Jan 2022+, OBRA assessments only → {n_assess:,} assessments across {n_fac} facilities"):
    st.markdown(f"""
**Date filter:** Only assessments from **January 2022 onward** (assessment reference date A2300 >= 2022-01-01).

**Assessment type filter:** Only OBRA assessment types are scored:
- **01** = Admission
- **02** = Quarterly
- **03** = Annual
- **04** = Significant Change

**Excluded:** Entry/discharge tracking records (non-clinical snapshots).

**Result:** {n_assess:,} scoreable assessments across {n_fac} facilities.
""")

with st.expander(f"Step 3 — MDS Version Handling: coalescing old + new item codes across versions {', '.join(str(v) for v in versions)}"):
    st.markdown(f"""
Each assessment carries an **Item Set Version** (ITM_SET_VRSN_CD) that tells us which version of MDS
coding was used. Our data spans versions **{', '.join(str(v) for v in versions)}**.

**Why this matters:** CMS periodically changes which MDS item codes map to which clinical concepts:
- **Eating ADL** moved from Section G (G0110H1) to Section GG (GG0130A1) around v1.18
- **Diuretics** moved from N0400A to N0415G1 in newer versions
- **Renal Insufficiency** is I1500 (not I1550, which became neurogenic bladder)

**How we handle it:** Where codes shifted, we **coalesce old + new fields** — if either is present,
the factor fires. This ensures no assessments are missed due to version transitions.
""")

with st.expander("Step 4 — Scoring: 33 risk factors checked per assessment, weighted 3 / 2 / 1 points"):
    st.markdown("""
For **each individual assessment**, the algorithm checks 33 clinical risk factors from the MDS data.
Each factor is a simple yes/no check against specific MDS item codes. If the condition is met,
points are added to the assessment's total score:

- :red[**3 points**] — Severe / acute risk factors (e.g., comatose, fever, vomiting, already dehydrated, hospice, IV fluids, CAA dehydration trigger)
- :orange[**2 points**] — Moderate / chronic risk factors (e.g., diabetes, CHF, renal failure, malnutrition, diuretics, delirium, feeding tube, weight loss)
- :green[**1 point**] — Contributing risk factors (e.g., female sex, depression, pneumonia, stroke, dental problems, incontinence, pressure ulcers)

**Special case — Cognitive Impairment:** Uses the RAI Manual's skip pattern. BIMS score (C0500) is checked
first. If BIMS is unavailable (score = 99 or blank), Staff Assessment of cognition (C1000) is used instead.
Severe impairment = 3 pts, moderate = 2 pts.

**The total score is the sum of all points that fired.** Possible range: 0 to 35+.
""")

with st.expander("Step 5 — Risk Tier Classification: Low / Moderate / High / Already Dehydrated"):
    st.markdown("""
After scoring, each assessment is classified into a risk tier:

| Rule | Tier | Action |
|------|------|--------|
| Score 0–5 | :green[**Low Risk**] | Standard hydration monitoring |
| Score 6–10 | :orange[**Moderate Risk**] | Enhanced monitoring, proactive fluid offering |
| Score 11+ | :red[**High Risk**] | Aggressive hydration protocol |
| J1550C = 1 (overrides score) | :violet[**Already Dehydrated**] | Immediate intervention required |

**Key:** J1550C is the MDS item for "Signs and Symptoms of Dehydration" — if checked, the patient
has 2+ clinical dehydration indicators per the RAI Manual. This **overrides** the point-based tier
because the patient is already experiencing dehydration regardless of their risk score.
""")

with st.expander("Step 6 — Demand Forecasting: rolling 6-month windows averaged across facilities"):
    st.markdown("""
To produce stable per-facility estimates for service line planning:

1. **Rolling 6-month windows** with a 3-month step are created across the full date range.
2. Within each window, **unique patients per facility** are counted by their **most recent** assessment's tier.
3. Partial windows at the edges are dropped.
4. All metrics are **averaged across valid windows** to smooth out seasonal variation.

This gives a reliable "per facility, per 6-month period" estimate — the numbers shown in the
Key Findings and Facility Summary sections above.
""")

# --- Full scoring model with evidence from Excel ---

_EVIDENCE_DATA = [
    {"Factor": "Cognitive Impairment (severe)", "MDS Code": "C0500 / C1000", "Points": 3, "Weight Tier": "High", "Trigger": "If BIMS score C0500 is 0 to 7. If BIMS is unavailable, staff assessment C1000 = 3", "Prevalence": "60.2%", "Clinical Reasoning": "Cognitive impairment is the single strongest and most consistently confirmed predictor of dehydration in nursing home residents. Impaired cognition reduces self-initiated drinking, recognition of thirst, and ability to request fluids. The BIMS/C1000 skip pattern follows the RAI Manual\u2019s prescribed assessment logic.", "Primary Source": "Bunn et al. JAMDA 2019 \u2014 Systematic review of 49 risk factors; cognitive impairment was one of only two factors confirmed across multiple studies.", "Source Link": "https://pubmed.ncbi.nlm.nih.gov/30056949/", "Supporting Sources": "Nagae et al. Nutrients 2020 \u2014 Dementia OR=6.29 (strongest predictor); Hooper et al. UK DRIE Study 2015 \u2014 consistently associated with dehydration; CMS RAI Manual v1.20.1 \u2014 BIMS/C1000 skip pattern specification.", "Supporting Links": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7709028/ | https://pmc.ncbi.nlm.nih.gov/articles/PMC5018558/"},
    {"Factor": "Cognitive Impairment (moderate)", "MDS Code": "C0500 / C1000", "Points": 2, "Weight Tier": "Moderate", "Trigger": "If BIMS score C0500 is 8 to 12. If BIMS is unavailable, staff assessment C1000 = 2", "Prevalence": "(included in 60.2% above)", "Clinical Reasoning": "Moderate cognitive impairment still impairs self-care and fluid intake behavior but to a lesser degree than severe impairment. Weighted at 2 pts to differentiate from severe (3 pts).", "Primary Source": "Same as severe cognitive impairment \u2014 Bunn et al. 2019, Nagae et al. 2020.", "Source Link": "https://pubmed.ncbi.nlm.nih.gov/30056949/", "Supporting Sources": "CMS RAI Manual v1.20.1 \u2014 C0500 scoring ranges.", "Supporting Links": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7709028/"},
    {"Factor": "Already Dehydrated", "MDS Code": "J1550C", "Points": 3, "Weight Tier": "High", "Trigger": "J1550C = 1 (also overrides tier to \u2018Already Dehydrated\u2019)", "Prevalence": "0.4%", "Clinical Reasoning": "J1550C indicates 2+ clinical signs of dehydration are already present per the RAI Manual. This is not a risk factor \u2014 it is active dehydration. The tier override ensures immediate clinical attention regardless of cumulative risk score.", "Primary Source": "CMS MDS 3.0 RAI Manual v1.20.1, Section J \u2014 Problem Conditions. J1550C is coded when the resident shows output exceeding intake AND one or more signs (dry mucous membranes, poor skin turgor, etc.).", "Source Link": "https://mdslearninghub.com/j1550c-problem-conditions-dehydrated-step-step", "Supporting Sources": "CMS Form CMS-20092, Hydration Critical Element Pathway \u2014 surveyors assess for these same clinical signs.", "Supporting Links": "https://www.cms.gov/files/document/cms-20092-hydrationpdf"},
    {"Factor": "Comatose", "MDS Code": "B0100", "Points": 3, "Weight Tier": "High", "Trigger": "B0100 = 1", "Prevalence": "0.2%", "Clinical Reasoning": "A comatose patient cannot self-hydrate at all. Complete dependence on external fluid administration makes this an acute, critical risk factor.", "Primary Source": "Thomas DR et al. JAMDA 2008 \u2014 Clinical dehydration framework: inability to access fluids independently is a primary mechanism.", "Source Link": "https://www.sciencedirect.com/science/article/abs/pii/S1525861008001072", "Supporting Sources": "CMS Hydration Critical Element Pathway \u2014 assesses whether resident can \u2018reach, pour, and drink without assistance.\u2019", "Supporting Links": "https://www.cms.gov/files/document/cms-20092-hydrationpdf"},
    {"Factor": "Swallowing Disorder", "MDS Code": "K0100A-D", "Points": 3, "Weight Tier": "High", "Trigger": "Any of K0100A, K0100B, K0100C, K0100D = 1", "Prevalence": "9.1%", "Clinical Reasoning": "Dysphagia directly impairs the ability to safely consume fluids. Patients with swallowing signs (loss of liquids, coughing/choking, complaints, holding food in mouth) avoid drinking due to aspiration risk, creating a direct pathway to dehydration.", "Primary Source": "Nagae et al. Nutrients 2020 \u2014 Dysphagia Severity Scale used as risk factor; low swallowing function assessed in nursing home dehydration study.", "Source Link": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7709028/", "Supporting Sources": "CMS RAI Manual v1.20.1 Section K \u2014 links swallowing disorders directly to dehydration risk; Thomas DR JAMDA 2008 \u2014 oral/pharyngeal dysphagia listed as primary risk factor.", "Supporting Links": "https://www.sciencedirect.com/science/article/abs/pii/S1525861008001072"},
    {"Factor": "Fever", "MDS Code": "J1550A", "Points": 3, "Weight Tier": "High", "Trigger": "J1550A = 1", "Prevalence": "0.3%", "Clinical Reasoning": "Fever causes insensible fluid losses through sweating and increased metabolic rate. It is one of only two risk factors (with cognitive impairment) confirmed across multiple studies in the Bunn systematic review.", "Primary Source": "Bunn et al. JAMDA 2019 \u2014 Systematic review: fever was one of only two factors consistently associated with dehydration across multiple studies.", "Source Link": "https://pubmed.ncbi.nlm.nih.gov/30056949/", "Supporting Sources": "Thomas DR JAMDA 2008 \u2014 fever listed under acute causes of dehydration; CMS Hydration Pathway \u2014 acute illness as risk factor.", "Supporting Links": "https://www.sciencedirect.com/science/article/abs/pii/S1525861008001072"},
    {"Factor": "Vomiting", "MDS Code": "J1550B", "Points": 3, "Weight Tier": "High", "Trigger": "J1550B = 1", "Prevalence": "0.4%", "Clinical Reasoning": "Vomiting causes direct, rapid fluid and electrolyte loss. It is an acute mechanism of dehydration that compounds with any underlying chronic risk factors.", "Primary Source": "Thomas DR JAMDA 2008 \u2014 Vomiting listed as acute cause of \u2018salt and water loss dehydration\u2019 requiring immediate assessment.", "Source Link": "https://www.sciencedirect.com/science/article/abs/pii/S1525861008001072", "Supporting Sources": "CMS Hydration Critical Element Pathway \u2014 vomiting listed as condition requiring hydration assessment.", "Supporting Links": "https://www.cms.gov/files/document/cms-20092-hydrationpdf"},
    {"Factor": "Hospice", "MDS Code": "O0100K2", "Points": 3, "Weight Tier": "High", "Trigger": "O0100K2 = 1", "Prevalence": "2.2%", "Clinical Reasoning": "Hospice patients are at end-of-life and frequently have reduced oral intake, declining functional status, and multiple co-morbidities that compound dehydration risk. Aggressive hydration decisions in hospice are complex clinical considerations.", "Primary Source": "Thomas DR JAMDA 2008 \u2014 End-of-life hydration decisions discussed as a distinct clinical scenario with high dehydration prevalence.", "Source Link": "https://www.sciencedirect.com/science/article/abs/pii/S1525861008001072", "Supporting Sources": "Clinical consensus \u2014 hospice patients routinely experience declining intake as part of the dying process.", "Supporting Links": ""},
    {"Factor": "Parenteral / IV Fluids", "MDS Code": "K0520A1/A2/A3", "Points": 3, "Weight Tier": "High", "Trigger": "Any of K0520A1, K0520A2, K0520A3 = 1 (any care setting)", "Prevalence": "14.3%", "Clinical Reasoning": "The presence of IV/parenteral fluids indicates the patient already cannot maintain adequate hydration through oral intake alone. This is a marker of existing hydration failure, not just risk. All 3 care-setting columns are checked (in facility, on admission, since admission).", "Primary Source": "CMS RAI Manual v1.20.1, Section K \u2014 K0520A documents parenteral/IV feeding as a nutritional approach indicating inability to meet needs orally.", "Source Link": "https://www.cms.gov/medicare/quality/nursing-home-improvement/resident-assessment-instrument-manual", "Supporting Sources": "CMS Hydration Pathway \u2014 reviews physician orders for IV fluids as part of dehydration assessment.", "Supporting Links": "https://www.cms.gov/files/document/cms-20092-hydrationpdf"},
    {"Factor": "CAA Dehydration Trigger", "MDS Code": "V0200A14A", "Points": 3, "Weight Tier": "High", "Trigger": "V0200A14A = 1 (CMS CAA #14 triggered)", "Prevalence": "22.8%", "Clinical Reasoning": "Care Area Assessment #14 is CMS\u2019s own algorithm built into the MDS that flags patients needing further dehydration assessment. If CMS\u2019s own system triggers a dehydration alert, this is strong evidence of elevated risk. Weighted at 3 pts because it represents CMS\u2019s regulatory determination.", "Primary Source": "CMS MDS 3.0 RAI Manual v1.20.1, Section V \u2014 Care Area Assessment (CAA) triggers. CAA #14 specifically addresses dehydration/fluid maintenance.", "Source Link": "https://www.cms.gov/medicare/quality/nursing-home-improvement/resident-assessment-instrument-manual", "Supporting Sources": "CMS Hydration Critical Element Pathway \u2014 directs surveyors to review CAA triggers.", "Supporting Links": "https://www.cms.gov/files/document/cms-20092-hydrationpdf"},
    {"Factor": "Malnutrition", "MDS Code": "I5600", "Points": 2, "Weight Tier": "Moderate", "Trigger": "I5600 = 1", "Prevalence": "29.7%", "Clinical Reasoning": "Malnutrition and dehydration are closely intertwined \u2014 malnourished patients often have reduced food and fluid intake simultaneously. Food contributes about 20% of daily water intake, so poor nutrition compounds fluid deficits. Initially weighted at 3 pts, moved to 2 pts per VP Operations clinical review.", "Primary Source": "Nagae et al. Nutrients 2020 \u2014 Nutritional status (MNA-SF) assessed as risk factor for chronic dehydration; malnourished residents had higher dehydration prevalence.", "Source Link": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7709028/", "Supporting Sources": "CMS RAI Manual Section V \u2014 CAA #12 (Nutritional Status) directly linked to dehydration CAA #14; Clinical review by S. Sklar, VP Operations \u2014 moved from 3\u21922 pts.", "Supporting Links": "https://www.cms.gov/medicare/quality/nursing-home-improvement/resident-assessment-instrument-manual"},
    {"Factor": "Diabetes Mellitus", "MDS Code": "I0600", "Points": 2, "Weight Tier": "Moderate", "Trigger": "I0600 = 1", "Prevalence": "25.8%", "Clinical Reasoning": "Diabetes causes osmotic diuresis (glucose pulls water into urine), increased urination, and impaired kidney concentrating ability. It is consistently identified as a dehydration risk factor in nursing home studies.", "Primary Source": "Hooper et al. UK DRIE Study 2015 \u2014 Diabetes mellitus was \u2018consistently associated with hydration status\u2019 across analyses.", "Source Link": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5018558/", "Supporting Sources": "Nagae et al. Nutrients 2020 \u2014 Diabetes assessed as comorbidity risk factor; Bunn et al. 2019 systematic review \u2014 diabetes identified in individual studies.", "Supporting Links": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7709028/ | https://pubmed.ncbi.nlm.nih.gov/30056949/"},
    {"Factor": "Heart Failure (CHF)", "MDS Code": "I4000", "Points": 2, "Weight Tier": "Moderate", "Trigger": "I4000 = 1", "Prevalence": "8.5%", "Clinical Reasoning": "CHF patients are frequently on fluid restrictions and diuretics, both of which create a paradoxical dehydration risk. Managing fluid balance in CHF is clinically complex \u2014 too much fluid worsens CHF, too little causes dehydration.", "Primary Source": "Thomas DR JAMDA 2008 \u2014 CHF listed as chronic condition complicating hydration management; diuretic use in CHF patients identified as indirect dehydration mechanism.", "Source Link": "https://www.sciencedirect.com/science/article/abs/pii/S1525861008001072", "Supporting Sources": "CMS Hydration Pathway \u2014 physician orders for fluid restrictions reviewed as part of assessment.", "Supporting Links": "https://www.cms.gov/files/document/cms-20092-hydrationpdf"},
    {"Factor": "Renal Insufficiency", "MDS Code": "I1500", "Points": 2, "Weight Tier": "Moderate", "Trigger": "I1500 = 1 (I1550 excluded \u2014 neurogenic bladder in recent MDS)", "Prevalence": "31.0%", "Clinical Reasoning": "Impaired kidneys lose the ability to concentrate urine, leading to excessive water loss. Renal dysfunction was the most consistently associated factor with serum osmolality in the UK DRIE study. I1550 was excluded because it maps to neurogenic bladder in recent MDS versions, not renal failure.", "Primary Source": "Hooper et al. UK DRIE Study 2015 \u2014 \u2018Renal dysfunction was consistently associated with serum osmolality and odds of dehydration\u2019 \u2014 the strongest association in their analysis.", "Source Link": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5018558/", "Supporting Sources": "Bunn et al. 2019 \u2014 renal impairment identified in systematic review; CMS RAI Manual \u2014 I1500 vs I1550 version change documented.", "Supporting Links": "https://pubmed.ncbi.nlm.nih.gov/30056949/"},
    {"Factor": "Diuretics", "MDS Code": "N0415G1", "Points": 2, "Weight Tier": "Moderate", "Trigger": "N0415G1 = 1 (updated from N0400A which had 0% data)", "Prevalence": "16.9%", "Clinical Reasoning": "Diuretics directly increase urine output, creating fluid deficit. A well-established medication-related dehydration risk factor. The MDS code was corrected from N0400A (old version, 0% prevalence in our data) to N0415G1 (current version) during model validation.", "Primary Source": "Hooper et al. UK DRIE Study 2015 \u2014 \u2018Potassium-sparing diuretics sometimes associated with dehydration.\u2019", "Source Link": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5018558/", "Supporting Sources": "Nagae et al. 2020 \u2014 diuretic use obtained from medical records as risk factor; CMS Hydration Pathway \u2014 medication review includes diuretics.", "Supporting Links": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7709028/ | https://www.cms.gov/files/document/cms-20092-hydrationpdf"},
    {"Factor": "Weight Loss", "MDS Code": "K0300", "Points": 2, "Weight Tier": "Moderate", "Trigger": "K0300 = 1 (5% in 30 days) or 2 (10% in 180 days)", "Prevalence": "7.6%", "Clinical Reasoning": "Significant weight loss indicates declining intake (both food and fluid) and/or wasting conditions. Water constitutes a large portion of body weight, so rapid weight loss often includes fluid loss.", "Primary Source": "CMS RAI Manual v1.20.1, Section K \u2014 K0300 weight loss is a clinical indicator directly linked to nutritional and hydration status assessment.", "Source Link": "https://www.cms.gov/medicare/quality/nursing-home-improvement/resident-assessment-instrument-manual", "Supporting Sources": "CMS Hydration Pathway \u2014 weight monitoring listed as part of hydration assessment; Nagae et al. 2020 \u2014 BMI/nutritional status linked to dehydration.", "Supporting Links": "https://www.cms.gov/files/document/cms-20092-hydrationpdf"},
    {"Factor": "Feeding Tube", "MDS Code": "K0520B1/B2/B3", "Points": 2, "Weight Tier": "Moderate", "Trigger": "Any of K0520B1, K0520B2, K0520B3 = 1 (any care setting)", "Prevalence": "4.8%", "Clinical Reasoning": "Feeding tube presence indicates the patient cannot meet nutritional/fluid needs orally. While the tube itself delivers fluids, it signals severe functional impairment and dependence on precise fluid management. All 3 care-setting columns checked.", "Primary Source": "CMS RAI Manual v1.20.1, Section K \u2014 K0520B documents tube feeding as nutritional approach; CMS Hydration Pathway \u2014 tube feeding reviewed for adequacy of fluid delivery.", "Source Link": "https://www.cms.gov/medicare/quality/nursing-home-improvement/resident-assessment-instrument-manual", "Supporting Sources": "Thomas DR JAMDA 2008 \u2014 tube-fed patients require careful fluid balance monitoring.", "Supporting Links": "https://www.sciencedirect.com/science/article/abs/pii/S1525861008001072"},
    {"Factor": "Eating ADL Dependence", "MDS Code": "G0110H1 / GG0130A1", "Points": 2, "Weight Tier": "Moderate", "Trigger": "G0110H1 = 3 or 4 (old MDS) OR GG0130A1 = 01, 02, 88 (new MDS) \u2014 coalesced", "Prevalence": "8.7%", "Clinical Reasoning": "Patients who cannot feed themselves independently also cannot self-hydrate. Dependence on staff for eating directly correlates with dependence for drinking. Old Section G and new Section GG codes are coalesced to handle the MDS version transition.", "Primary Source": "CMS Hydration Critical Element Pathway \u2014 specifically asks: can the resident \u2018reach, pour, and drink without assistance\u2019? Staff assistance with drinking is a core assessment element.", "Source Link": "https://www.cms.gov/files/document/cms-20092-hydrationpdf", "Supporting Sources": "Nagae et al. 2020 \u2014 ADL (Barthel Index) assessed; lower ADL scores associated with risk; CMS RAI Manual \u2014 G0110H1 to GG0130A1 transition documented.", "Supporting Links": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7709028/"},
    {"Factor": "Delirium Signs", "MDS Code": "C1310A-D", "Points": 2, "Weight Tier": "Moderate", "Trigger": "Any of C1310A=1, C1310B=1/2, C1310C=1/2, C1310D=1/2", "Prevalence": "14.7%", "Clinical Reasoning": "Delirium causes acute confusion, inattention, and altered consciousness \u2014 all of which prevent self-initiated fluid intake. Delirium can also be both a cause and consequence of dehydration, creating a dangerous feedback loop.", "Primary Source": "Thomas DR JAMDA 2008 \u2014 Confusion and altered mental status listed as both risk factor and clinical sign of dehydration.", "Source Link": "https://www.sciencedirect.com/science/article/abs/pii/S1525861008001072", "Supporting Sources": "CMS Hydration Pathway \u2014 confusion listed as sign of altered hydration status; Bunn et al. 2019 \u2014 cognitive status broadly confirmed.", "Supporting Links": "https://www.cms.gov/files/document/cms-20092-hydrationpdf"},
    {"Factor": "UTI", "MDS Code": "I2300", "Points": 2, "Weight Tier": "Moderate", "Trigger": "I2300 = 1", "Prevalence": "3.6%", "Clinical Reasoning": "UTIs cause fever, increased fluid loss, and reduced intake due to malaise. UTIs are also more common in dehydrated patients (concentrated urine promotes bacterial growth), creating a bidirectional risk.", "Primary Source": "Thomas DR JAMDA 2008 \u2014 Infections including UTI listed as acute causes of dehydration through fever and reduced intake.", "Source Link": "https://www.sciencedirect.com/science/article/abs/pii/S1525861008001072", "Supporting Sources": "Clinical consensus \u2014 adequate hydration is a primary UTI prevention strategy in nursing homes.", "Supporting Links": ""},
    {"Factor": "Antipsychotics", "MDS Code": "N0415A1", "Points": 2, "Weight Tier": "Moderate", "Trigger": "N0415A1 = 1", "Prevalence": "11.3%", "Clinical Reasoning": "Antipsychotic medications cause sedation (reducing self-initiated drinking), dry mouth, and can impair swallowing. They also affect thermoregulation, increasing insensible fluid losses. Added per VP Operations clinical review.", "Primary Source": "Thomas DR JAMDA 2008 \u2014 Medications that cause sedation or dry mouth identified as dehydration risk factors.", "Source Link": "https://www.sciencedirect.com/science/article/abs/pii/S1525861008001072", "Supporting Sources": "Added per clinical review by S. Sklar, VP Operations \u2014 identified as commonly prescribed medication with dehydration side effects.", "Supporting Links": ""},
    {"Factor": "CAA Nutritional Trigger", "MDS Code": "V0200A12A", "Points": 2, "Weight Tier": "Moderate", "Trigger": "V0200A12A = 1 (CMS CAA #12 triggered)", "Prevalence": "51.6%", "Clinical Reasoning": "CMS CAA #12 flags patients needing nutritional status assessment. Nutrition and hydration are clinically intertwined \u2014 food provides about 20% of daily water intake, and nutritional decline typically accompanies fluid intake decline.", "Primary Source": "CMS MDS 3.0 RAI Manual v1.20.1, Section V \u2014 CAA #12 (Nutritional Status) is explicitly linked to CAA #14 (Dehydration) in the RAI Manual\u2019s care area assessment guidance.", "Source Link": "https://www.cms.gov/medicare/quality/nursing-home-improvement/resident-assessment-instrument-manual", "Supporting Sources": "CMS Hydration Pathway \u2014 nutritional status review is part of dehydration assessment.", "Supporting Links": "https://www.cms.gov/files/document/cms-20092-hydrationpdf"},
    {"Factor": "Always Incontinent", "MDS Code": "H0300 = 3", "Points": 2, "Weight Tier": "Moderate", "Trigger": "H0300 = 3 (always incontinent)", "Prevalence": "35.9%", "Clinical Reasoning": "Always-incontinent patients may voluntarily restrict fluid intake to reduce incontinence episodes (fluid avoidance behavior). Severe incontinence also indicates functional decline correlating with inability to self-hydrate.", "Primary Source": "Hooper et al. UK DRIE Study 2015 \u2014 \u2018Bladder incontinence sometimes associated with dehydration.\u2019", "Source Link": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5018558/", "Supporting Sources": "CMS RAI Manual \u2014 H0300 coding: 0=continent, 1=occasionally, 2=frequently, 3=always, 9=not rated. Clinical review by S. Sklar \u2014 H0300=3 elevated from 1 pt to 2 pts.", "Supporting Links": ""},
    {"Factor": "Female Sex", "MDS Code": "A0800", "Points": 1, "Weight Tier": "Low", "Trigger": "A0800 = 2", "Prevalence": "50.9%", "Clinical Reasoning": "Women have lower total body water percentage due to higher body fat ratio, making them more susceptible to fluid imbalances.", "Primary Source": "Crea-Arsenio et al. PLOS ONE 2024 \u2014 \u2018Women were significantly more likely to exhibit symptoms of dehydration compared to men.\u2019", "Source Link": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0297588", "Supporting Sources": "Hooper et al. UK DRIE 2015 \u2014 sex \u2018sometimes associated\u2019 with dehydration status.", "Supporting Links": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5018558/"},
    {"Factor": "Depression", "MDS Code": "D0160 / D0600 / I6000", "Points": 1, "Weight Tier": "Low", "Trigger": "PHQ-9 resident (D0160) \u2265 10 OR PHQ-9 staff (D0600) \u2265 10 OR depression dx (I6000) = 1", "Prevalence": "28.3%", "Clinical Reasoning": "Depression reduces motivation for self-care including eating and drinking. Depressed residents may refuse meals and fluids.", "Primary Source": "Thomas DR JAMDA 2008 \u2014 Depression and reduced motivation for self-care identified as contributing factors to inadequate fluid intake.", "Source Link": "https://www.sciencedirect.com/science/article/abs/pii/S1525861008001072", "Supporting Sources": "CMS Hydration Pathway \u2014 behavioral factors affecting fluid intake assessed by surveyors.", "Supporting Links": "https://www.cms.gov/files/document/cms-20092-hydrationpdf"},
    {"Factor": "Pneumonia", "MDS Code": "I2000", "Points": 1, "Weight Tier": "Low", "Trigger": "I2000 = 1", "Prevalence": "2.0%", "Clinical Reasoning": "Pneumonia causes fever (insensible losses), tachypnea (respiratory water loss), reduced oral intake due to malaise, and dysphagia risk from respiratory compromise.", "Primary Source": "Thomas DR JAMDA 2008 \u2014 Acute infections including pneumonia cause dehydration through fever and reduced intake.", "Source Link": "https://www.sciencedirect.com/science/article/abs/pii/S1525861008001072", "Supporting Sources": "Clinical consensus \u2014 pneumonia management guidelines emphasize adequate hydration.", "Supporting Links": ""},
    {"Factor": "Stroke / CVA", "MDS Code": "I4900", "Points": 1, "Weight Tier": "Low", "Trigger": "I4900 = 1", "Prevalence": "8.2%", "Clinical Reasoning": "Stroke can cause dysphagia, cognitive impairment, and functional dependence \u2014 all independent dehydration risk factors. Weighted at 1 pt because its effects are captured indirectly through the specific impairments it causes (which score separately).", "Primary Source": "Thomas DR JAMDA 2008 \u2014 Neurological conditions causing functional impairment listed as dehydration risk factors.", "Source Link": "https://www.sciencedirect.com/science/article/abs/pii/S1525861008001072", "Supporting Sources": "Clinical consensus \u2014 post-stroke dysphagia screening is standard of care precisely because of aspiration and dehydration risk.", "Supporting Links": ""},
    {"Factor": "Dental Problem", "MDS Code": "L0200D", "Points": 1, "Weight Tier": "Low", "Trigger": "L0200D = 1", "Prevalence": "3.5%", "Clinical Reasoning": "Dental problems (broken, loose, or carious teeth, inflamed gums) cause oral pain that reduces both food and fluid intake.", "Primary Source": "CMS Hydration Critical Element Pathway (CMS-20092) \u2014 \u2018Poor oral health and dental problems\u2019 listed as sign indicating altered hydration status.", "Source Link": "https://www.cms.gov/files/document/cms-20092-hydrationpdf", "Supporting Sources": "CMS RAI Manual Section L \u2014 oral/dental status assessment.", "Supporting Links": "https://www.cms.gov/medicare/quality/nursing-home-improvement/resident-assessment-instrument-manual"},
    {"Factor": "Communication Impairment", "MDS Code": "B0700", "Points": 1, "Weight Tier": "Low", "Trigger": "B0700 = 2 or 3 (sometimes/rarely understood)", "Prevalence": "16.4%", "Clinical Reasoning": "Patients who cannot make themselves understood are unable to request fluids, report thirst, or communicate discomfort from dehydration.", "Primary Source": "CMS Hydration Critical Element Pathway \u2014 assesses whether staff can identify and respond to resident\u2019s fluid needs, which requires communication.", "Source Link": "https://www.cms.gov/files/document/cms-20092-hydrationpdf", "Supporting Sources": "Thomas DR JAMDA 2008 \u2014 inability to communicate needs as a mechanism for inadequate intake.", "Supporting Links": "https://www.sciencedirect.com/science/article/abs/pii/S1525861008001072"},
    {"Factor": "Occasionally / Frequently Incontinent", "MDS Code": "H0300 = 1, 2", "Points": 1, "Weight Tier": "Low", "Trigger": "H0300 = 1 (occasionally) or 2 (frequently incontinent)", "Prevalence": "47.2%", "Clinical Reasoning": "Moderate incontinence contributes to fluid avoidance behavior but to a lesser degree than always-incontinent (2 pts). Weighted at 1 pt as a contributing rather than primary risk factor.", "Primary Source": "Hooper et al. UK DRIE Study 2015 \u2014 Bladder incontinence associated with dehydration status.", "Source Link": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5018558/", "Supporting Sources": "CMS RAI Manual \u2014 H0300 coding values; Clinical review \u2014 separated from always-incontinent per MDS coding.", "Supporting Links": ""},
    {"Factor": "Dialysis", "MDS Code": "O0100J2", "Points": 1, "Weight Tier": "Low", "Trigger": "O0100J2 = 1", "Prevalence": "0.7%", "Clinical Reasoning": "Dialysis patients have complex fluid balance needs with strict intake management. While dialysis removes excess fluid, the inter-dialytic period and fluid restrictions create dehydration risk windows.", "Primary Source": "Thomas DR JAMDA 2008 \u2014 Renal conditions and their treatments discussed as dehydration risk factors.", "Source Link": "https://www.sciencedirect.com/science/article/abs/pii/S1525861008001072", "Supporting Sources": "CMS RAI Manual Section O \u2014 special treatments including dialysis.", "Supporting Links": ""},
    {"Factor": "Alzheimer's / Dementia Dx", "MDS Code": "I5250 / I5300", "Points": 1, "Weight Tier": "Low", "Trigger": "I5250 = 1 (Alzheimer\u2019s) or I5300 = 1 (other dementia)", "Prevalence": "38.0%", "Clinical Reasoning": "Dementia diagnosis adds 1 pt as a contributing factor. The primary cognitive impairment effect is already captured through BIMS/C1000 scoring (2-3 pts). This additional point captures the diagnosed condition itself, which may indicate progressive decline beyond what a single cognitive assessment shows.", "Primary Source": "Nagae et al. Nutrients 2020 \u2014 Dementia diagnosis (DSM-V criteria) had OR=6.29 for chronic dehydration \u2014 the strongest single predictor in their study.", "Source Link": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7709028/", "Supporting Sources": "Bunn et al. 2019 \u2014 cognitive impairment consistently confirmed; scored separately from BIMS to capture diagnosed progressive conditions.", "Supporting Links": "https://pubmed.ncbi.nlm.nih.gov/30056949/"},
    {"Factor": "Constipation", "MDS Code": "H0600", "Points": 1, "Weight Tier": "Low", "Trigger": "H0600 = 1", "Prevalence": "21.7%", "Clinical Reasoning": "Constipation is both a cause and effect of inadequate fluid intake. Insufficient hydration reduces bowel motility; constipation can reduce appetite and willingness to eat/drink.", "Primary Source": "Nagae et al. Nutrients 2020 \u2014 \u2018Constipation was assessed based on the resident\u2019s complaint or use of laxatives\u2019 as a factor in the dehydration risk analysis.", "Source Link": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7709028/", "Supporting Sources": "Added per clinical review by S. Sklar, VP Operations.", "Supporting Links": ""},
    {"Factor": "Pressure Ulcers Present", "MDS Code": "M0210", "Points": 1, "Weight Tier": "Low", "Trigger": "M0210 = 1 (unhealed pressure ulcer present)", "Prevalence": "11.6%", "Clinical Reasoning": "Pressure ulcers cause insensible fluid loss through wound exudate and increase metabolic fluid requirements for healing. Dehydration also impairs wound healing, creating a negative cycle.", "Primary Source": "Crea-Arsenio et al. PLOS ONE 2024 \u2014 Studied \u2018Factors associated with pressure ulcer and dehydration in long-term care settings\u2019 \u2014 found both conditions co-occur and share risk factors.", "Source Link": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0297588", "Supporting Sources": "Added per clinical review by S. Sklar, VP Operations \u2014 identified pressure ulcers as a missing risk factor in the initial model.", "Supporting Links": ""},
]

with st.expander("View complete scoring model \u2014 all 33 factors with MDS codes, triggers, and evidence"):
    st.markdown("Each row is one risk factor. All cells wrap text so full content is visible. Links are clickable.")

    _tier_bg = {"High": "#FDF2F2", "Moderate": "#FFF8E1", "Low": "#F0FDF4"}

    def _make_link(url):
        if not url or url != url:
            return ""
        urls = [u.strip() for u in str(url).split("|") if u.strip()]
        return "<br>".join(
            f'<a href="{u}" target="_blank" style="color:#0563C1;word-break:break-all;">{u}</a>'
            for u in urls
        )

    _html_rows = []
    for r in _EVIDENCE_DATA:
        bg = _tier_bg.get(r["Weight Tier"], "#fff")
        _html_rows.append(
            f'<tr style="background:{bg};">'
            f'<td><b>{r["Factor"]}</b></td>'
            f'<td>{r["MDS Code"]}</td>'
            f'<td style="text-align:center;">{r["Points"]}</td>'
            f'<td>{r["Weight Tier"]}</td>'
            f'<td>{r["Trigger"]}</td>'
            f'<td style="text-align:center;">{r["Prevalence"]}</td>'
            f'<td>{r["Clinical Reasoning"]}</td>'
            f'<td>{r["Primary Source"]}</td>'
            f'<td>{_make_link(r["Source Link"])}</td>'
            f'<td>{r["Supporting Sources"]}</td>'
            f'<td>{_make_link(r["Supporting Links"])}</td>'
            f'</tr>'
        )

    _table_html = f"""
    <div style="max-height:800px; overflow:auto; border:1px solid #e0e0e0; border-radius:6px;">
    <table style="border-collapse:collapse; width:100%; font-size:0.8rem; line-height:1.4;">
    <thead>
    <tr style="background:#2C3E50; color:white; position:sticky; top:0; z-index:1;">
        <th style="padding:8px 6px; text-align:left; min-width:140px;">Factor</th>
        <th style="padding:8px 6px; text-align:left; min-width:90px;">MDS Code</th>
        <th style="padding:8px 6px; text-align:center; min-width:30px;">Pts</th>
        <th style="padding:8px 6px; text-align:left; min-width:65px;">Tier</th>
        <th style="padding:8px 6px; text-align:left; min-width:180px;">Trigger</th>
        <th style="padding:8px 6px; text-align:center; min-width:50px;">Prev.</th>
        <th style="padding:8px 6px; text-align:left; min-width:220px;">Clinical Reasoning</th>
        <th style="padding:8px 6px; text-align:left; min-width:200px;">Primary Source</th>
        <th style="padding:8px 6px; text-align:left; min-width:160px;">Source Link</th>
        <th style="padding:8px 6px; text-align:left; min-width:200px;">Supporting Sources</th>
        <th style="padding:8px 6px; text-align:left; min-width:160px;">More Links</th>
    </tr>
    </thead>
    <tbody>
    {"".join(_html_rows)}
    </tbody>
    </table>
    </div>
    <style>
    div[data-testid="stExpander"] table td {{
        padding: 6px;
        vertical-align: top;
        border-bottom: 1px solid #e9ecef;
        word-wrap: break-word;
        white-space: normal;
    }}
    </style>
    """
    st.markdown(_table_html, unsafe_allow_html=True)

st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# 5. DATA EXPLORER
# ═════════════════════════════════════════════════════════════════════════════

st.header("5. Data Explorer")

tab1, tab2 = st.tabs(["Facility Summary", "Per-Assessment Scores"])

with tab1:
    st.markdown("Average per facility per 6-month rolling window. Click column headers to sort.")
    display_cols = [
        "Facility", "#Unique Patients",
        "#Patients Low", "#Patients Moderate", "#Patients High", "#Patients Already",
        "% Moderate Patients", "% High Patients", "% Already Patients",
    ]
    st.dataframe(
        summary[display_cols].style.format({
            "#Unique Patients": "{:.0f}", "#Patients Low": "{:.0f}",
            "#Patients Moderate": "{:.0f}", "#Patients High": "{:.0f}",
            "#Patients Already": "{:.1f}",
            "% Moderate Patients": "{:.1f}%", "% High Patients": "{:.1f}%",
            "% Already Patients": "{:.1f}%",
        }),
        width="stretch", hide_index=True, height=500,
    )

with tab2:
    st.markdown("Individual assessment scores. Use filters below to narrow down.")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_fac = st.multiselect("Facility", sorted(scores["surrogate_facility_id"].unique()), default=[])
    with fc2:
        sel_tier = st.multiselect("Risk Tier", TIER_ORDER, default=[])
    with fc3:
        score_range = st.slider(
            "Score Range", 0, int(scores["risk_score"].max()),
            (0, int(scores["risk_score"].max())),
        )

    filtered = scores.copy()
    if sel_fac:
        filtered = filtered[filtered["surrogate_facility_id"].isin(sel_fac)]
    if sel_tier:
        filtered = filtered[filtered["risk_tier"].isin(sel_tier)]
    filtered = filtered[
        (filtered["risk_score"] >= score_range[0]) & (filtered["risk_score"] <= score_range[1])
    ]

    st.markdown(f"Showing **{len(filtered):,}** of {n_assess:,} assessments")
    show_cols = [
        "surrogate_facility_id", "surrogate_patient_id",
        "risk_score", "risk_tier",
        "item_set_version", "assessment_type_obra",
    ] + pts_cols
    st.dataframe(filtered[show_cols], width="stretch", hide_index=True, height=400)

st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align:center; color:#999; font-size:0.75rem; padding:1rem 0;">
    Dehydration Risk Stratification · MDS 3.0 RAI Manual v1.20.1 · Data: Jan 2022+ · De-identified
</div>
""", unsafe_allow_html=True)
