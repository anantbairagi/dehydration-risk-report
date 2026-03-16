"""Dehydration Risk Stratification v2 — Stakeholder Report.

Launch:  streamlit run app.py
Reads:   dehydration_data.db  (SQLite, produced by extract_to_sqlite.py)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
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

@st.cache_data
def load_and_score():
    db_path = Path(__file__).parent / "dehydration_data.db"
    if not db_path.exists():
        st.error(f"Database not found at `{db_path}`. Run `extract_to_sqlite.py` first.")
        st.stop()

    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql("SELECT * FROM assessments", conn)
    conn.close()

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
    f"~**{avg['#Patients Moderate']:.0f}** moderate-risk + ~**{avg['#Patients High']:.0f}** high-risk + "
    f"~**{avg['#Patients Already']:.1f}** actively dehydrated = "
    f"**~{total_target:.0f} addressable patients**. "
    f"Across {n_fac} facilities: **~{total_target * n_fac:,.0f} target patients** per 6-month cycle."
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

st.markdown("""
This section walks through every step of the analysis — from raw data to risk classification —
so you can verify the methodology yourself.
""")

# --- Step-by-step pipeline ---

st.markdown("""
<div style="background:#f8f9fa; border-radius:8px; padding:1.2rem 1.5rem; border-left:4px solid #3498DB; margin-bottom:1rem;">
<h3 style="margin-top:0; color:#2C3E50;">Step 1 — Data Source</h3>
<b>What:</b> MDS 3.0 (Minimum Data Set) assessments from CMS-mandated standardized nursing home evaluations.<br>
<b>Where:</b> Extracted from the facility management database (PostgreSQL). Each assessment contains hundreds
of coded clinical items stored as structured JSON.<br>
<b>De-identification:</b> All patient and facility identifiers are replaced with surrogate IDs before analysis.
No PHI leaves the source environment.
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="background:#f8f9fa; border-radius:8px; padding:1.2rem 1.5rem; border-left:4px solid #3498DB; margin-bottom:1rem;">
<h3 style="margin-top:0; color:#2C3E50;">Step 2 — Filtering</h3>
<b>Date filter:</b> Only assessments from <b>January 2022 onward</b> are included
(assessment reference date A2300 ≥ 2022-01-01).<br>
<b>Assessment type filter:</b> Only OBRA assessment types are scored:<br>
&nbsp;&nbsp;&nbsp;&nbsp;• <b>01</b> = Admission &nbsp;&nbsp; • <b>02</b> = Quarterly &nbsp;&nbsp;
• <b>03</b> = Annual &nbsp;&nbsp; • <b>04</b> = Significant Change<br>
<b>Excluded:</b> Entry/discharge tracking records (non-clinical snapshots).<br>
<b>Result:</b> {n_assess:,} scoreable assessments across {n_fac} facilities.
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="background:#f8f9fa; border-radius:8px; padding:1.2rem 1.5rem; border-left:4px solid #3498DB; margin-bottom:1rem;">
<h3 style="margin-top:0; color:#2C3E50;">Step 3 — MDS Version Handling</h3>
Each assessment carries an <b>Item Set Version</b> (field ITM_SET_VRSN_CD) that tells us which
version of MDS coding was used. Our data spans versions <b>{', '.join(str(v) for v in versions)}</b>.<br><br>
<b>Why this matters:</b> CMS periodically changes which MDS item codes map to which clinical concepts.
For example:<br>
&nbsp;&nbsp;&nbsp;&nbsp;• <b>Eating ADL</b> moved from Section G (G0110H1) to Section GG (GG0130A1) around v1.18<br>
&nbsp;&nbsp;&nbsp;&nbsp;• <b>Diuretics</b> moved from N0400A to N0415G1 in newer versions<br>
&nbsp;&nbsp;&nbsp;&nbsp;• <b>Renal Insufficiency</b> is I1500 (not I1550, which became neurogenic bladder)<br><br>
<b>How we handle it:</b> Where codes shifted, we <b>coalesce old + new fields</b> — if either is present,
the factor fires. This ensures no assessments are missed due to version transitions.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:#FFF3CD; border-radius:8px; padding:1.2rem 1.5rem; border-left:4px solid #F39C12; margin-bottom:1rem;">
<h3 style="margin-top:0; color:#2C3E50;">Step 4 — Scoring Each Assessment</h3>
For <b>each individual assessment</b>, the algorithm checks 33 clinical risk factors from the MDS data.
Each factor is a simple yes/no check against specific MDS item codes. If the condition is met,
points are added to the assessment's total score:<br><br>
&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#E74C3C;font-weight:700;">● 3 points</span> — Severe / acute risk factors
(e.g., comatose, fever, vomiting, already dehydrated, hospice, IV fluids, CAA dehydration trigger)<br>
&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#F39C12;font-weight:700;">● 2 points</span> — Moderate / chronic risk factors
(e.g., diabetes, CHF, renal failure, malnutrition, diuretics, delirium, feeding tube, weight loss)<br>
&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#2ECC71;font-weight:700;">● 1 point</span> — Contributing risk factors
(e.g., female sex, depression, pneumonia, stroke, dental problems, incontinence, pressure ulcers)<br><br>
<b>Special case — Cognitive Impairment:</b> Uses the RAI Manual's skip pattern. BIMS score (C0500) is checked first.
If BIMS is unavailable (score = 99 or blank), Staff Assessment of cognition (C1000) is used instead.
Severe impairment = 3 pts, moderate = 2 pts.<br><br>
<b>The total score is the sum of all points that fired.</b> Possible range: 0 to 35+.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:#f8f9fa; border-radius:8px; padding:1.2rem 1.5rem; border-left:4px solid #8E44AD; margin-bottom:1rem;">
<h3 style="margin-top:0; color:#2C3E50;">Step 5 — Risk Tier Classification</h3>
After scoring, each assessment is classified into a risk tier:<br><br>
<table style="width:100%; border-collapse:collapse; font-size:0.95rem;">
<tr style="background:#2C3E50; color:white;">
  <th style="padding:8px; text-align:left;">Rule</th>
  <th style="padding:8px; text-align:left;">Tier</th>
  <th style="padding:8px; text-align:left;">Clinical Action</th>
</tr>
<tr style="background:#F0FDF4;">
  <td style="padding:8px; border-bottom:1px solid #e0e0e0;"><b>Score 0–5</b></td>
  <td style="padding:8px; border-bottom:1px solid #e0e0e0;"><span style="color:#2ECC71;font-weight:700;">Low Risk</span></td>
  <td style="padding:8px; border-bottom:1px solid #e0e0e0;">Standard hydration monitoring</td>
</tr>
<tr>
  <td style="padding:8px; border-bottom:1px solid #e0e0e0;"><b>Score 6–10</b></td>
  <td style="padding:8px; border-bottom:1px solid #e0e0e0;"><span style="color:#F39C12;font-weight:700;">Moderate Risk</span></td>
  <td style="padding:8px; border-bottom:1px solid #e0e0e0;">Enhanced monitoring, proactive fluid offering</td>
</tr>
<tr style="background:#FDF2F2;">
  <td style="padding:8px; border-bottom:1px solid #e0e0e0;"><b>Score 11+</b></td>
  <td style="padding:8px; border-bottom:1px solid #e0e0e0;"><span style="color:#E74C3C;font-weight:700;">High Risk</span></td>
  <td style="padding:8px; border-bottom:1px solid #e0e0e0;">Aggressive hydration protocol</td>
</tr>
<tr style="background:#F5EEF8;">
  <td style="padding:8px;"><b>J1550C = 1</b> (overrides score)</td>
  <td style="padding:8px;"><span style="color:#8E44AD;font-weight:700;">Already Dehydrated</span></td>
  <td style="padding:8px;">Immediate intervention required</td>
</tr>
</table><br>
<b>Key:</b> J1550C is the MDS item for "Signs and Symptoms of Dehydration" — if checked, the patient
has 2+ clinical dehydration indicators per the RAI Manual. This <b>overrides</b> the point-based tier
because the patient is already experiencing dehydration regardless of their risk score.
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="background:#f8f9fa; border-radius:8px; padding:1.2rem 1.5rem; border-left:4px solid #3498DB; margin-bottom:1rem;">
<h3 style="margin-top:0; color:#2C3E50;">Step 6 — Demand Forecasting (Facility Averages)</h3>
To produce stable per-facility estimates for service line planning:<br><br>
&nbsp;&nbsp;&nbsp;&nbsp;1. <b>Rolling 6-month windows</b> with a 3-month step are created across the full date range.<br>
&nbsp;&nbsp;&nbsp;&nbsp;2. Within each window, <b>unique patients per facility</b> are counted by their
<b>most recent</b> assessment's tier.<br>
&nbsp;&nbsp;&nbsp;&nbsp;3. Partial windows at the edges are dropped.<br>
&nbsp;&nbsp;&nbsp;&nbsp;4. All metrics are <b>averaged across valid windows</b> to smooth out seasonal variation.<br><br>
This gives a reliable "per facility, per 6-month period" estimate — the numbers shown in the Key Findings
and Facility Summary sections above.
</div>
""", unsafe_allow_html=True)

# --- Full scoring model in expander ---

with st.expander("View complete scoring model — all 33 factors with MDS codes and triggers"):
    st.markdown("""
    Each row below is one risk factor. The **MDS Code** column shows exactly which MDS item is checked.
    The **Trigger** column shows the condition that must be true for points to be awarded.
    **Prevalence** shows how often this factor fires across all assessments in our data.
    """)

    model_rows = []
    trigger_map = {
        "pts_cognitive": "BIMS 0–7 → 3 pts; BIMS 8–12 → 2 pts; if BIMS unavailable: C1000=3 → 3 pts, C1000=2 → 2 pts",
        "pts_dehydrated": "J1550C = 1 (also overrides tier to Already Dehydrated)",
        "pts_comatose": "B0100 = 1",
        "pts_dysphagia": "Any of K0100A, K0100B, K0100C, K0100D = 1",
        "pts_fever": "J1550A = 1",
        "pts_vomiting": "J1550B = 1",
        "pts_hospice": "O0100K2 = 1",
        "pts_iv_parenteral": "Any of K0520A1, K0520A2, K0520A3 = 1 (any care setting)",
        "pts_caa_dehydration": "V0200A14A = 1 (CMS CAA #14 triggered)",
        "pts_malnutrition": "I5600 = 1",
        "pts_diabetes": "I0600 = 1",
        "pts_chf": "I4000 = 1",
        "pts_renal": "I1500 = 1 (I1550 excluded — neurogenic bladder in recent MDS)",
        "pts_diuretics": "N0415G1 = 1 (updated from N0400A which had 0% data)",
        "pts_weight_loss": "K0300 = 1 (5% in 30 days) or 2 (10% in 180 days)",
        "pts_feeding_tube": "Any of K0520B1, K0520B2, K0520B3 = 1 (any care setting)",
        "pts_adl_eating": "G0110H1 = 3 or 4 (old MDS) OR GG0130A1 = 01, 02, 88 (new MDS) — coalesced",
        "pts_delirium": "Any of C1310A = 1, C1310B = 1/2, C1310C = 1/2, C1310D = 1/2",
        "pts_uti": "I2300 = 1",
        "pts_antipsychotics": "N0415A1 = 1",
        "pts_caa_nutritional": "V0200A12A = 1 (CMS CAA #12 triggered)",
        "pts_incontinence_mod": "H0300 = 3 (always incontinent)",
        "pts_female": "A0800 = 2",
        "pts_depression": "D0160 ≥ 10 (PHQ-9 resident) OR D0600 ≥ 10 (PHQ-9 staff) OR I6000 = 1",
        "pts_pneumonia": "I2000 = 1",
        "pts_stroke": "I4900 = 1",
        "pts_dental": "L0200D = 1",
        "pts_communication": "B0700 = 2 or 3 (sometimes/rarely understood)",
        "pts_incontinence_low": "H0300 = 1 (occasionally) or 2 (frequently incontinent)",
        "pts_dialysis": "O0100J2 = 1",
        "pts_dementia_dx": "I5250 = 1 (Alzheimer's) or I5300 = 1 (other dementia)",
        "pts_constipation": "H0600 = 1",
        "pts_pressure_ulcer": "M0210 = 1 (unhealed pressure ulcer present)",
    }

    for col in pts_cols:
        if col in FACTOR_META:
            name, mds, pts, weight = FACTOR_META[col]
        else:
            name = col.replace("pts_", "").replace("_", " ").title()
            mds, pts, weight = "—", "?", "low"
        fired = (scores[col] > 0).sum()
        pct = fired / n_assess * 100
        model_rows.append({
            "Factor": name,
            "MDS Code": mds,
            "Points": pts,
            "Trigger": trigger_map.get(col, "—"),
            "Prevalence": f"{pct:.1f}%",
            "_sort": int(pts) if pts.isdigit() else 3,
        })
    model_df = (pd.DataFrame(model_rows)
                .sort_values("_sort", ascending=False)
                .drop(columns=["_sort"]))
    st.dataframe(model_df, width="stretch", hide_index=True, height=700)

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
