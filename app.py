import duckdb
import polars as pl
import pandas as pd
import streamlit as st
from streamlit import column_config as cc

st.set_page_config(page_title="TMSS Rate Comparison MVP", layout="wide")

st.title("TMSS Rate Comparison (MVP)")
st.caption("Upload TMSS Rate File exports. Filter by SROID / Origin / Destination / SCAC.")

# ======================================================
# CONFIG
# ======================================================

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

OUR_SCAC = "AMJV"   # change if needed


# ======================================================
# UTILITIES
# ======================================================

def reset_app():
    for k in ["data", "con", "filter_lists"]:
        if k in st.session_state:
            del st.session_state[k]


def load_with_progress(files) -> pl.DataFrame:
    progress = st.progress(0, text="Starting…")
    status = st.empty()

    frames = []
    total = len(files)

    for i, f in enumerate(files, start=1):
        status.write(f"Reading file {i}/{total}: **{f.name}**")

        name = f.name.lower()
        if name.endswith(".csv"):
            pdf = pd.read_csv(f)
        else:
            pdf = pd.read_excel(f, engine="openpyxl")

        pdf.columns = [c.strip() for c in pdf.columns]

        missing = [c for c in REQUIRED_COLS if c not in pdf.columns]
        if missing:
            raise ValueError(f"{f.name}: Missing required columns: {missing}")

        pdf = pdf[REQUIRED_COLS].copy()
        df = pl.from_pandas(pdf)

        for c in STRING_COLS:
            df = df.with_columns(pl.col(c).cast(pl.Utf8).str.strip_chars().alias(c))

        for c in NUMERIC_COLS:
            df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False).alias(c))

        frames.append(df)
        progress.progress(i / total)

    status.write("Stitching & deduplicating…")

    out = pl.concat(frames, how="vertical")
    out = out.unique(subset=["SCAC", "SROID", "ORIGIN", "DESTINATION"], keep="first")

    progress.progress(1.0)
    status.empty()

    return out


def register_in_duckdb(df: pl.DataFrame):
    con = duckdb.connect(database=":memory:")
    con.register("tmss", df.to_arrow())
    return con


# ======================================================
# UPLOAD STEP
# ======================================================

if "data" not in st.session_state:

    uploaded = st.file_uploader(
        "Upload TMSS Rate File export(s) (.xlsx or .csv)",
        type=["xlsx", "csv"],
        accept_multiple_files=True
    )

    if not uploaded:
        st.stop()

    col1, col2 = st.columns(2)
    start = col1.button("Process Files", type="primary")
    col2.button("Reset", on_click=reset_app)

    if not start:
        st.stop()

    try:
        data = load_with_progress(uploaded)
    except Exception as e:
        st.error(str(e))
        st.stop()

    con = register_in_duckdb(data)

    st.session_state["data"] = data
    st.session_state["con"] = con

    st.session_state["filter_lists"] = {
        "sroid": [x[0] for x in con.execute("SELECT DISTINCT SROID FROM tmss WHERE SROID IS NOT NULL ORDER BY SROID").fetchall()],
        "origin": [x[0] for x in con.execute("SELECT DISTINCT ORIGIN FROM tmss WHERE ORIGIN IS NOT NULL ORDER BY ORIGIN").fetchall()],
        "dest": [x[0] for x in con.execute("SELECT DISTINCT DESTINATION FROM tmss WHERE DESTINATION IS NOT NULL ORDER BY DESTINATION").fetchall()],
        "scac": [x[0] for x in con.execute("SELECT DISTINCT SCAC FROM tmss WHERE SCAC IS NOT NULL ORDER BY SCAC").fetchall()],
    }

    st.success(f"Loaded {data.height:,} rows.")


# ======================================================
# TABLES STEP
# ======================================================

data = st.session_state["data"]
con = st.session_state["con"]

st.button("Change uploaded files", on_click=reset_app)

lists = st.session_state["filter_lists"]

# Sidebar filters
st.sidebar.header("Filters")
sel_sroid = st.sidebar.multiselect("SROID", lists["sroid"])
sel_origin = st.sidebar.multiselect("Origin", lists["origin"])
sel_dest = st.sidebar.multiselect("Destination", lists["dest"])
sel_scac = st.sidebar.multiselect("SCAC", lists["scac"])

# Build WHERE clause
where = ["1=1"]
params = []

def add_filter(col, vals):
    if vals:
        placeholders = ",".join(["?"] * len(vals))
        where.append(f"{col} IN ({placeholders})")
        params.extend(vals)

add_filter("SROID", sel_sroid)
add_filter("ORIGIN", sel_origin)
add_filter("DESTINATION", sel_dest)
add_filter("SCAC", sel_scac)

where_sql = " AND ".join(where)

query = f"""
SELECT SCAC, SROID, ORIGIN, DESTINATION,
       LINEHAUL, STORAGEINTRANSIT, ACCESSORIALS, UNACCOMPANIEDAIRBAG,
       CLASSVEHICLE1, CLASSVEHICLE2, CLASSVEHICLE3
FROM tmss
WHERE {where_sql}
"""

rows = pl.from_arrow(con.execute(query, params).arrow())

# ======================================================
# CHUNK PIVOT
# ======================================================

st.write("## Pivot View")
st.caption("Tip: Use trackpad horizontal scroll or Shift + mousewheel.")

metric_map = {
    "HHE %": "LINEHAUL",
    "SIT %": "STORAGEINTRANSIT",
    "Accessorial %": "ACCESSORIALS",
    "UAB %": "UNACCOMPANIEDAIRBAG",
    "Vehicle Cat 1": "CLASSVEHICLE1",
    "Vehicle Cat 2": "CLASSVEHICLE2",
    "Vehicle Cat 3": "CLASSVEHICLE3",
}

metric_label = st.selectbox("Metric", list(metric_map.keys()))
metric_col = metric_map[metric_label]

chunk_pd = rows.select(["SROID", "ORIGIN", "DESTINATION", "SCAC", metric_col]).to_pandas()

if chunk_pd.empty:
    st.stop()

pivot = chunk_pd.pivot_table(
    index=["SROID", "ORIGIN", "DESTINATION"],
    columns="SCAC",
    values=metric_col,
    aggfunc="first"
)

pivot["min_competitor"] = pivot.min(axis=1)
pivot["max_competitor"] = pivot.max(axis=1)
pivot["avg_competitor"] = pivot.mean(axis=1)

pivot[["min_competitor", "max_competitor", "avg_competitor"]] = \
    pivot[["min_competitor", "max_competitor", "avg_competitor"]].round(0)

# Reorder columns
derived = ["avg_competitor", "min_competitor", "max_competitor"]
scacs = [c for c in pivot.columns if c not in derived]

if OUR_SCAC in scacs:
    scacs = [OUR_SCAC] + [c for c in scacs if c != OUR_SCAC]

pivot = pivot[derived + scacs]

# Styling
# Add a visual divider column after the derived metrics
# (Blank strings render as a skinny separator column)
pivot.insert(3, "│", "")

# Make AVG/MIN/MAX whole numbers (already rounded, but ensure int-ish display)
for c in ["avg_competitor", "min_competitor", "max_competitor"]:
    # Keep NaNs safe
    pivot[c] = pivot[c].astype("Int64")

# Rename our SCAC column header to be obvious, without styling
if OUR_SCAC in pivot.columns:
    pivot = pivot.rename(columns={OUR_SCAC: f"{OUR_SCAC} ⭐"})

# Optional: limit rows shown for performance / usability
st.sidebar.header("Display")
max_lanes = st.sidebar.slider("Max lanes to show (rows)", min_value=100, max_value=5000, value=1000, step=100)

# Show the pivot (no Styler)
pivot_to_show = pivot.head(max_lanes)

st.dataframe(pivot_to_show, use_container_width=True, height=650)

# Download full pivot (not truncated)
csv_data = pivot.to_csv().encode("utf-8")
st.download_button(
    "Download Pivot CSV",
    csv_data,
    "tmss_pivot.csv",
    mime="text/csv"
)