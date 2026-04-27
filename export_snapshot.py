"""Export a de-identified MDS snapshot from VDP Postgres to parquet.

Run this script once (or whenever you want to refresh the snapshot) with the
Fly.io proxy tunnel active on port 16380:

    fly proxy 16380:5432 -a <app-name>
    python export_snapshot.py

Output: dehydration_snapshot.parquet  (committed to the repo; read by app.py
        when DATABASE_URL is not set, e.g. on Streamlit Community Cloud)

No PHI is written -- all data comes from app.mds_assessments which contains
only surrogate IDs and de-identified clinical fields.
"""

from pathlib import Path
import sys
import pandas as pd
import db

# Same query as app.py _LOAD_QUERY
_QUERY = """
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

OUT_PATH = Path(__file__).parent / "dehydration_snapshot.parquet"


def main() -> None:
    print("Connecting to VDP Postgres...")
    try:
        conn = db.get_connection()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Could not connect. Is the Fly tunnel running on port 16380?\n{exc}")
        sys.exit(1)

    print("Querying OBRA assessments (Jan 2022+)...")
    try:
        with conn.cursor() as cur:
            cur.execute(_QUERY)
            cols = [desc[0] for desc in cur.description]
            df = pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        conn.close()

    n = len(df)
    n_fac = df["surrogate_facility_id"].nunique()
    n_pat = df["surrogate_patient_id"].nunique()
    print(f"  {n:,} assessments  |  {n_pat:,} patients  |  {n_fac} facilities")

    df.to_parquet(OUT_PATH, index=False, compression="zstd")
    size_kb = OUT_PATH.stat().st_size // 1024
    print(f"Saved -> {OUT_PATH.name}  ({size_kb:,} KB)")
    print()
    print("Next steps:")
    print("  git add dehydration_snapshot.parquet")
    print("  git commit -m 'snapshot: refresh VDP data'")
    print("  git push")
    print("  Then redeploy (or it auto-deploys) on share.streamlit.io")


if __name__ == "__main__":
    main()
