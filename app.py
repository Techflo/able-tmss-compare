import duckdb
import polars as pl
import pandas as pd
import streamlit as st

st.set_page_config(page_title="TMSS Rate Comparison MVP", layout="wide")

st.title("TMSS Rate Comparison (MVP)")
st.caption("Upload TMSS Rate File exports. Filter by SROID / Origin / Destination / SCAC. Get lane stats + export.")

# === REQUIRED COLUMNS (based on your TMSS extract) ===
REQUIRED_COLS = [
    "SCAC", "SROID", "ORIGIN", "DESTINATION",
    "LINEHAUL", "STORAGEINTRANSIT", "ACCESSORIALS", "UNACCOMPANIEDAIRBAG",
    "CLASSVEHICLE1", "CLASSVEHICLE2", "CLASSVEHICLE3",
]

NUMERIC_COLS = [
    "LINEHAUL", "STORAGEINTRANSIT", "ACCESSORIALS", "UNACCOMPANIEDAIRBAG",
    "CLASSVEHICLE1", "CLASSVEHICLE2", "CLASSVEHICLE3",
]

STRING_COLS = ["SCAC", "SROID", "ORIGIN", "DESTINATION"]


def reset_app():
    for k in ["data", "con", "loaded_filenames", "filter_lists"]:
        if k in st.session_state:
            del st.session_state[k]


def load_with_progress(files) -> pl.DataFrame:
    """
    Load multiple Excel files with a UI progress bar.
    Runs ONLY when user clicks 'Process Files'.
    """
    progress = st.progress(0, text="Starting…")
    status = st.empty()

    frames = []
    total = len(files)

    for i, f in enumerate(files, start=1):
        status.write(f"Reading file {i}/{total}: **{f.name}**")

        pdf = pd.read_excel(f, engine="openpyxl")
        pdf.columns = [c.strip() for c in pdf.columns]

        missing = [c for c in REQUIRED_COLS if c not in pdf.columns]
        if missing:
            raise ValueError(f"{f.name}: Missing required columns: {missing}")

        pdf = pdf[REQUIRED_COLS].copy()
        df = pl.from_pandas(pdf)

        # Clean strings
        for c in STRING_COLS:
            df = df.with_columns(pl.col(c).cast(pl.Utf8).str.strip_chars().alias(c))

        # Cast numerics
        for c in NUMERIC_COLS:
            df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False).alias(c))

        frames.append(df)
        progress.progress(i / total, text=f"Loaded {i}/{total} files…")

    status.write("Stitching & deduplicating…")
    out = pl.concat(frames, how="vertical")
    out = out.unique(subset=["SCAC", "SROID", "ORIGIN", "DESTINATION"], keep="first")

    progress.progress(1.0, text="Done.")
    status.empty()
    return out


def register_in_duckdb(df: pl.DataFrame) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.register("tmss", df.to_arrow())
    return con


# ======================================================
# Upload step (only shown until data is loaded)
# ======================================================
if "data" not in st.session_state:
    uploaded = st.file_uploader(
        "Upload TMSS Rate File export(s) (.xlsx)",
        type=["xlsx"],
        accept_multiple_files=True
    )

    if not uploaded:
        st.info("Upload one or more TMSS exports to begin.")
        st.stop()

    colA, colB = st.columns([1, 1])
    with colA:
        start = st.button("Process Files", type="primary")
    with colB:
        st.button("Clear / Reset", on_click=reset_app)

    if not start:
        st.stop()

    try:
        data = load_with_progress(uploaded)
    except Exception as e:
        st.error(str(e))
        st.stop()

    con = register_in_duckdb(data)

    # Cache data + DB connection in session (prevents re-reading on every rerun)
    st.session_state["data"] = data
    st.session_state["con"] = con
    st.session_state["loaded_filenames"] = [f.name for f in uploaded]

    # Precompute filter lists once
    st.session_state["filter_lists"] = {
        "sroid": [x[0] for x in con.execute("SELECT DISTINCT SROID FROM tmss WHERE SROID IS NOT NULL ORDER BY SROID").fetchall()],
        "origin": [x[0] for x in con.execute("SELECT DISTINCT ORIGIN FROM tmss WHERE ORIGIN IS NOT NULL ORDER BY ORIGIN").fetchall()],
        "dest": [x[0] for x in con.execute("SELECT DISTINCT DESTINATION FROM tmss WHERE DESTINATION IS NOT NULL ORDER BY DESTINATION").fetchall()],
        "scac": [x[0] for x in con.execute("SELECT DISTINCT SCAC FROM tmss WHERE SCAC IS NOT NULL ORDER BY SCAC").fetchall()],
    }

    st.success(f"Loaded {data.height:,} rows (deduplicated).")

# ======================================================
# Tables step (shown after load)
# ======================================================
data = st.session_state["data"]
con = st.session_state["con"]

# Optional: allow changing files
st.button("Change uploaded files", on_click=reset_app)

lists = st.session_state.get("filter_lists", {})
sroid_list = lists.get("sroid", [])
origin_list = lists.get("origin", [])
dest_list = lists.get("dest", [])
scac_list = lists.get("scac", [])

# === Sidebar filters ===
st.sidebar.header("Filters")
sel_sroid = st.sidebar.multiselect("SROID (Agency ID)", sroid_list, default=[])
sel_origin = st.sidebar.multiselect("Origin", origin_list, default=[])
sel_dest = st.sidebar.multiselect("Destination", dest_list, default=[])
sel_scac = st.sidebar.multiselect("SCAC", scac_list, default=[])

# === Factor formatting toggle ===
st.sidebar.header("Factor format")
factor_mode = st.sidebar.radio(
    "How should factors be interpreted?",
    ["Whole percent (85 = 85%)", "Decimal (0.85 = 85%)"],
    index=0
)


def apply_factor_mode(df: pl.DataFrame) -> pl.DataFrame:
    if factor_mode.startswith("Decimal"):
        return df.with_columns([
            (pl.col("LINEHAUL") * 100).alias("LINEHAUL"),
            (pl.col("STORAGEINTRANSIT") * 100).alias("STORAGEINTRANSIT"),
            (pl.col("ACCESSORIALS") * 100).alias("ACCESSORIALS"),
            (pl.col("UNACCOMPANIEDAIRBAG") * 100).alias("UNACCOMPANIEDAIRBAG"),
        ])
    return df


# === Build SQL WHERE clause ===
where = ["1=1"]
params = []


def add_in_filter(col, values):
    if values:
        placeholders = ",".join(["?"] * len(values))
        where.append(f"{col} IN ({placeholders})")
        params.extend(values)


add_in_filter("SROID", sel_sroid)
add_in_filter("ORIGIN", sel_origin)
add_in_filter("DESTINATION", sel_dest)
add_in_filter("SCAC", sel_scac)

where_sql = " AND ".join(where)

# === Query filtered rows (internal use only) ===
query_rows = f"""
SELECT
  SCAC, SROID, ORIGIN, DESTINATION,
  LINEHAUL, STORAGEINTRANSIT, ACCESSORIALS, UNACCOMPANIEDAIRBAG,
  CLASSVEHICLE1, CLASSVEHICLE2, CLASSVEHICLE3
FROM tmss
WHERE {where_sql}
"""
rows_arrow = con.execute(query_rows, params).arrow()
rows = pl.from_arrow(rows_arrow)
rows = apply_factor_mode(rows)

# ======================================================
# Chunk Pivot View (scroll large chunks)
# ======================================================
st.write("## Chunk Pivot View (Scroll lanes, SCACs across top)")

metric_map = {
    "Linehaul %": "LINEHAUL",
    "SIT %": "STORAGEINTRANSIT",
    "Accessorial %": "ACCESSORIALS",
    "UAB %": "UNACCOMPANIEDAIRBAG",
    "Vehicle Cat 1 (flat)": "CLASSVEHICLE1",
    "Vehicle Cat 2 (flat)": "CLASSVEHICLE2",
    "Vehicle Cat 3 (flat)": "CLASSVEHICLE3",
}

metric_label = st.selectbox(
    "Choose metric to compare across SCACs",
    list(metric_map.keys()),
    key="chunk_metric"
)
metric_col = metric_map[metric_label]

chunk_pd = rows.select(["SROID", "ORIGIN", "DESTINATION", "SCAC", metric_col]).to_pandas()

if chunk_pd.empty:
    st.info("No rows available for this chunk.")
else:
    pivot = chunk_pd.pivot_table(
        index=["SROID", "ORIGIN", "DESTINATION"],
        columns="SCAC",
        values=metric_col,
        aggfunc="first"
    ).sort_index()

    # Derived columns across SCAC columns for each lane
    pivot["min_competitor"] = pivot.min(axis=1, numeric_only=True)
    pivot["max_competitor"] = pivot.max(axis=1, numeric_only=True)
    pivot["avg_competitor"] = pivot.mean(axis=1, numeric_only=True)

    # Put derived columns first
    derived_cols = ["avg_competitor", "min_competitor", "max_competitor"]
    scac_cols = [c for c in pivot.columns if c not in derived_cols]
    pivot = pivot[derived_cols + scac_cols]

    st.dataframe(pivot, use_container_width=True, height=650)

    csv_data = pivot.to_csv().encode("utf-8")
    st.download_button(
        f"Download Chunk Pivot (CSV) — {metric_label}",
        data=csv_data,
        file_name=f"tmss_chunk_pivot_{metric_col}.csv",
        mime="text/csv"
    )

# ======================================================
# Lane summary stats
# ======================================================
query_summary = f"""
SELECT
  SROID,
  ORIGIN,
  DESTINATION,
  COUNT(DISTINCT SCAC) AS competitors,
  AVG(LINEHAUL) AS linehaul_avg,
  MIN(LINEHAUL) AS linehaul_min,
  MAX(LINEHAUL) AS linehaul_max,
  AVG(STORAGEINTRANSIT) AS sit_avg,
  MIN(STORAGEINTRANSIT) AS sit_min,
  MAX(STORAGEINTRANSIT) AS sit_max,
  AVG(ACCESSORIALS) AS acc_avg,
  MIN(ACCESSORIALS) AS acc_min,
  MAX(ACCESSORIALS) AS acc_max,
  AVG(UNACCOMPANIEDAIRBAG) AS uab_avg,
  MIN(UNACCOMPANIEDAIRBAG) AS uab_min,
  MAX(UNACCOMPANIEDAIRBAG) AS uab_max,
  AVG(CLASSVEHICLE1) AS v1_avg,
  MIN(CLASSVEHICLE1) AS v1_min,
  MAX(CLASSVEHICLE1) AS v1_max,
  AVG(CLASSVEHICLE2) AS v2_avg,
  MIN(CLASSVEHICLE2) AS v2_min,
  MAX(CLASSVEHICLE2) AS v2_max,
  AVG(CLASSVEHICLE3) AS v3_avg,
  MIN(CLASSVEHICLE3) AS v3_min,
  MAX(CLASSVEHICLE3) AS v3_max
FROM tmss
WHERE {where_sql}
GROUP BY 1,2,3
ORDER BY competitors DESC
"""
summary_arrow = con.execute(query_summary, params).arrow()
summary = pl.from_arrow(summary_arrow)
summary = apply_factor_mode(summary)

st.write("### Lane Summary Stats (SROID + Origin + Destination)")
st.dataframe(summary.to_pandas(), use_container_width=True, height=360)

# === Download lane summary ===
def pl_to_csv_bytes(dframe: pl.DataFrame) -> bytes:
    return dframe.write_csv().encode("utf-8")


st.download_button(
    "Download Lane Summary (CSV)",
    data=pl_to_csv_bytes(summary),
    file_name="tmss_lane_summary.csv",
    mime="text/csv"
)