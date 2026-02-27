import duckdb
import polars as pl
import pandas as pd
import streamlit as st
from streamlit import column_config as cc

st.set_page_config(page_title="TMSS Rate Comparison MVP", layout="wide")

# --- CSS: wrap headers, keep horizontal scrollbar visible, slightly tighter cells ---
st.markdown(
    """
<style>
/* Wrap column headers */
div[data-testid="stDataFrame"] thead tr th div, 
div[data-testid="stDataEditor"] thead tr th div {
    white-space: normal !important;
    line-height: 1.1 !important;
}
/* Wrap cell text where possible (Origin/Destination benefit) */
div[data-testid="stDataFrame"] tbody tr td div,
div[data-testid="stDataEditor"] tbody tr td div {
    white-space: normal !important;
    line-height: 1.15 !important;
}
/* Make scrollbars more consistently visible */
div[data-testid="stDataFrame"] div[role="grid"] {
    overflow: auto !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("TMSS Rate Comparison (MVP)")
st.caption("Upload TMSS exports (.xlsx/.csv). Filter lanes, pivot SCACs across top, compare actual $ totals.")

# ======================================================
# CONFIG
# ======================================================

REQUIRED_COLS = [
    "SCAC", "SROID", "ORIGIN", "DESTINATION",
    "LINEHAUL", "UNACCOMPANIEDAIRBAG",
    "CLASSVEHICLE2",
]

NUMERIC_COLS = ["LINEHAUL", "UNACCOMPANIEDAIRBAG", "CLASSVEHICLE2"]
STRING_COLS = ["SCAC", "SROID", "ORIGIN", "DESTINATION"]

OUR_SCAC = "AMJV"

# ======================================================
# DEFAULT BASELINES
# ======================================================

DEFAULT_HHG_TABLE = pd.DataFrame(
    [
        {"hhg_weight_break_lbs": "1000–1999 lbs", "hhg_baseline": 174.00, "hhg_actual_net_weight": 1500},
        {"hhg_weight_break_lbs": "2000–3999 lbs", "hhg_baseline": 123.00, "hhg_actual_net_weight": 3000},
        {"hhg_weight_break_lbs": "4000–7999 lbs", "hhg_baseline": 116.00, "hhg_actual_net_weight": 5500},
        {"hhg_weight_break_lbs": "8000–11999 lbs", "hhg_baseline": 107.00, "hhg_actual_net_weight": 10000},
        {"hhg_weight_break_lbs": "12000–15999 lbs", "hhg_baseline": 100.00, "hhg_actual_net_weight": 14000},
        {"hhg_weight_break_lbs": "16000+ lbs", "hhg_baseline": 92.00, "hhg_actual_net_weight": 20000},
    ]
)

DEFAULT_UAB_TABLE = pd.DataFrame(
    [
        {"uab_weight_break_kg": "45–133 kg", "uab_baseline": 1.26, "uab_actual_gross_weight": 113},
        {"uab_weight_break_kg": "134–224 kg", "uab_baseline": 1.14, "uab_actual_gross_weight": 224},
        {"uab_weight_break_kg": "225–314 kg", "uab_baseline": 1.09, "uab_actual_gross_weight": 314},
        {"uab_weight_break_kg": "315–404 kg", "uab_baseline": 1.04, "uab_actual_gross_weight": 404},
        {"uab_weight_break_kg": "405+ kg", "uab_baseline": 0.99, "uab_actual_gross_weight": 1000},
    ]
)

# ======================================================
# UTILITIES
# ======================================================

def reset_app():
    for k in ["data", "con", "filter_lists", "lane_inputs"]:
        if k in st.session_state:
            del st.session_state[k]


def load_with_progress(files) -> pl.DataFrame:
    status = st.empty()
    progress_bar = st.progress(0.0, text="Starting…")

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
            status.empty()
            progress_bar.empty()
            raise ValueError(f"{f.name}: Missing required columns: {missing}")

        pdf = pdf[REQUIRED_COLS].copy()
        df = pl.from_pandas(pdf)

        for c in STRING_COLS:
            df = df.with_columns(pl.col(c).cast(pl.Utf8).str.strip_chars().alias(c))

        for c in NUMERIC_COLS:
            df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False).alias(c))

        frames.append(df)
        progress_bar.progress(i / total, text=f"Loaded {i}/{total} files…")

    status.write("Stitching & deduplicating…")
    out = pl.concat(frames, how="vertical")
    out = out.unique(subset=["SCAC", "SROID", "ORIGIN", "DESTINATION"], keep="first")

    status.empty()
    progress_bar.empty()
    return out


def register_in_duckdb(df: pl.DataFrame):
    con = duckdb.connect(database=":memory:")
    con.register("tmss", df.to_arrow())
    return con


def fmt_money(x: float, decimals: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"${x:,.{decimals}f}"


def build_hhg_dropdown(hhg_df: pd.DataFrame):
    labels = ["N/A"]
    lookup = {"N/A": None}
    for _, r in hhg_df.iterrows():
        label = f"{r['hhg_weight_break_lbs']} | {fmt_money(float(r['hhg_baseline']), 2)}"
        labels.append(label)
        lookup[label] = r["hhg_weight_break_lbs"]
    return labels, lookup


def build_uab_dropdown(uab_df: pd.DataFrame):
    labels = ["N/A"]
    lookup = {"N/A": None}
    for _, r in uab_df.iterrows():
        label = f"{r['uab_weight_break_kg']} | {fmt_money(float(r['uab_baseline']), 2)}"
        labels.append(label)
        lookup[label] = r["uab_weight_break_kg"]
    return labels, lookup


def get_hhg_row(hhg_df: pd.DataFrame, break_label: str) -> dict:
    row = hhg_df.loc[hhg_df["hhg_weight_break_lbs"] == break_label]
    if row.empty:
        raise ValueError(f"HHG baseline row not found for: {break_label}")
    return row.iloc[0].to_dict()


def get_uab_row(uab_df: pd.DataFrame, break_label: str) -> dict:
    row = uab_df.loc[uab_df["uab_weight_break_kg"] == break_label]
    if row.empty:
        raise ValueError(f"UAB baseline row not found for: {break_label}")
    return row.iloc[0].to_dict()


# ======================================================
# SESSION STATE DEFAULTS
# ======================================================

if "hhg_table" not in st.session_state:
    st.session_state["hhg_table"] = DEFAULT_HHG_TABLE.copy()

if "uab_table" not in st.session_state:
    st.session_state["uab_table"] = DEFAULT_UAB_TABLE.copy()

# ======================================================
# TABS
# ======================================================

tab_compare, tab_baselines = st.tabs(["Compare", "Baselines & Assumptions"])

with tab_baselines:
    st.subheader("Baselines & Assumptions (Editable)")

    st.write("### HHG baselines (lbs)")
    st.session_state["hhg_table"] = st.data_editor(
        st.session_state["hhg_table"],
        use_container_width=True,
        height=260,
        num_rows="dynamic",
        column_config={
            "hhg_weight_break_lbs": cc.TextColumn("Weight Break (lbs)", width="medium"),
            "hhg_baseline": cc.NumberColumn("Baseline ($)", format="%.2f", width="small"),
            "hhg_actual_net_weight": cc.NumberColumn("Actual Net Weight (lbs)", format="%.0f", width="small"),
        },
        key="hhg_editor",
    )

    st.write("### UAB baselines (kg)")
    st.session_state["uab_table"] = st.data_editor(
        st.session_state["uab_table"],
        use_container_width=True,
        height=260,
        num_rows="dynamic",
        column_config={
            "uab_weight_break_kg": cc.TextColumn("Weight Break (kg)", width="medium"),
            "uab_baseline": cc.NumberColumn("Baseline ($)", format="%.2f", width="small"),
            "uab_actual_gross_weight": cc.NumberColumn("Actual Gross Weight (kg)", format="%.0f", width="small"),
        },
        key="uab_editor",
    )

with tab_compare:

    if "data" not in st.session_state:
        uploaded = st.file_uploader(
            "Upload TMSS Rate File export(s) (.xlsx or .csv)",
            type=["xlsx", "csv"],
            accept_multiple_files=True
        )

        if not uploaded:
            st.info("Upload one or more TMSS exports to begin.")
            st.stop()

        c1, c2 = st.columns(2)
        start = c1.button("Process Files", type="primary")
        c2.button("Reset", on_click=reset_app)

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

        st.success(f"Loaded {data.height:,} rows (deduplicated).")

    con = st.session_state["con"]
    lists = st.session_state["filter_lists"]

    st.button("Change uploaded files", on_click=reset_app)

    st.sidebar.header("Filters")
    sel_sroid = st.sidebar.multiselect("SROID", lists["sroid"])
    sel_origin = st.sidebar.multiselect("Origin", lists["origin"])
    sel_dest = st.sidebar.multiselect("Destination", lists["dest"])
    sel_scac = st.sidebar.multiselect("SCAC", lists["scac"])

    st.sidebar.header("Display")
    max_lanes = st.sidebar.slider("Max lanes to show (rows)", 100, 5000, 1000, 100)

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
           LINEHAUL, UNACCOMPANIEDAIRBAG, CLASSVEHICLE2
    FROM tmss
    WHERE {where_sql}
    """
    rows = pl.from_arrow(con.execute(query, params).arrow())

    st.write("## Chunk Pivot View")

    hhg_df = st.session_state["hhg_table"].copy()
    uab_df = st.session_state["uab_table"].copy()

    hhg_labels, hhg_lookup = build_hhg_dropdown(hhg_df)
    uab_labels, uab_lookup = build_uab_dropdown(uab_df)

    hhg_choice = st.selectbox("HHG Baseline (Weight break | Baseline)", hhg_labels, index=0)
    uab_choice = st.selectbox("UAB Baseline (Weight break | Baseline)", uab_labels, index=0)

    hhg_break = hhg_lookup.get(hhg_choice)
    uab_break = uab_lookup.get(uab_choice)

    metric_options = [
        "HHE (Actual $)",
        "UAB (Actual $)",
        "POV (Flat $)",
        "TOTAL RELO HHE+UAB",
        "TOTAL RELO HHE+UAB+POV",
    ]
    metric_label = st.selectbox("Metric", metric_options)

    # Lane inputs for AMJV overrides
    lanes_pd = rows.select(["SROID", "ORIGIN", "DESTINATION"]).unique().to_pandas()
    lanes_pd = lanes_pd.sort_values(["SROID", "ORIGIN", "DESTINATION"]).head(max_lanes).reset_index(drop=True)

    if "lane_inputs" not in st.session_state:
        st.session_state["lane_inputs"] = lanes_pd.assign(HHG_RATE=None, UAB_RATE=None, POV_RATE=None)

    existing = st.session_state["lane_inputs"].copy()
    merged = lanes_pd.merge(existing, on=["SROID", "ORIGIN", "DESTINATION"], how="left")
    st.session_state["lane_inputs"] = merged[["SROID", "ORIGIN", "DESTINATION", "HHG_RATE", "UAB_RATE", "POV_RATE"]]

    st.write("### AMJV Lane Inputs (override replaces uploaded AMJV values; clearing inputs reverts to uploaded)")
    edited_inputs = st.data_editor(
        st.session_state["lane_inputs"],
        use_container_width=True,
        height=260,
        num_rows="fixed",
        column_config={
            "SROID": cc.TextColumn("SROID", width="small"),
            "ORIGIN": cc.TextColumn("ORIGIN", width="large"),
            "DESTINATION": cc.TextColumn("DESTINATION", width="large"),
            "HHG_RATE": cc.NumberColumn("HHG Rate (1000% = x10)", format="%.0f", width="small"),
            "UAB_RATE": cc.NumberColumn("UAB Rate (1000% = x10)", format="%.0f", width="small"),
            "POV_RATE": cc.NumberColumn("POV Rate ($ flat)", format="%.0f", width="small"),
        },
        key="lane_inputs_editor",
    )
    st.session_state["lane_inputs"] = edited_inputs

    rows2 = rows.join(pl.from_pandas(st.session_state["lane_inputs"]), on=["SROID", "ORIGIN", "DESTINATION"], how="left")

    # --- Uploaded (absolute) competitor figures ---
    hhg_uploaded = pl.col("LINEHAUL")
    uab_uploaded = pl.col("UNACCOMPANIEDAIRBAG")
    pov_uploaded = pl.col("CLASSVEHICLE2")

    is_ours = (pl.col("SCAC") == OUR_SCAC)
    have_baselines = (hhg_break is not None) & (uab_break is not None)

    if have_baselines:
        hhg_row = get_hhg_row(hhg_df, hhg_break)
        uab_row = get_uab_row(uab_df, uab_break)

        HHG_BASELINE = float(hhg_row["hhg_baseline"])
        HHG_ACTUAL_NET = float(hhg_row["hhg_actual_net_weight"])
        UAB_BASELINE = float(uab_row["uab_baseline"])
        UAB_ACTUAL_GROSS = float(uab_row["uab_actual_gross_weight"])

        # --- AMJV override values (absolute) ---
        hhg_override = pl.lit(HHG_BASELINE) * (pl.col("HHG_RATE") / 100.0) * (pl.lit(HHG_ACTUAL_NET) / 100.0)
        uab_override = pl.lit(UAB_BASELINE) * (pl.col("UAB_RATE") / 100.0) * pl.lit(UAB_ACTUAL_GROSS)
        pov_override = pl.col("POV_RATE")

        # --- Effective values:
        # For AMJV, if a given override input exists, USE override; else revert to uploaded.
        hhg_eff = pl.when(is_ours & pl.col("HHG_RATE").is_not_null()).then(hhg_override).otherwise(hhg_uploaded)
        uab_eff = pl.when(is_ours & pl.col("UAB_RATE").is_not_null()).then(uab_override).otherwise(uab_uploaded)
        pov_eff = pl.when(is_ours & pl.col("POV_RATE").is_not_null()).then(pov_override).otherwise(pov_uploaded)
    else:
        # No baselines chosen -> nobody can be overridden, including AMJV
        hhg_eff, uab_eff, pov_eff = hhg_uploaded, uab_uploaded, pov_uploaded

    # Metric selection
    if metric_label == "HHE (Actual $)":
        metric_expr = hhg_eff
    elif metric_label == "UAB (Actual $)":
        metric_expr = uab_eff
    elif metric_label == "POV (Flat $)":
        metric_expr = pov_eff
    elif metric_label == "TOTAL RELO HHE+UAB":
        metric_expr = hhg_eff + uab_eff
    elif metric_label == "TOTAL RELO HHE+UAB+POV":
        metric_expr = hhg_eff + uab_eff + pov_eff
    else:
        metric_expr = hhg_eff + uab_eff

    rows_metric = rows2.with_columns(metric_expr.alias("METRIC"))

    chunk_pd = rows_metric.select(["SROID", "ORIGIN", "DESTINATION", "SCAC", "METRIC"]).to_pandas()
    if chunk_pd.empty:
        st.info("No rows available for this chunk.")
        st.stop()

    pivot = chunk_pd.pivot_table(
        index=["SROID", "ORIGIN", "DESTINATION"],
        columns="SCAC",
        values="METRIC",
        aggfunc="first"
    ).sort_index()

    pivot["min_competitor"] = pivot.min(axis=1, numeric_only=True)
    pivot["max_competitor"] = pivot.max(axis=1, numeric_only=True)
    pivot["avg_competitor"] = pivot.mean(axis=1, numeric_only=True)

    pivot = pivot.round(0)

    derived = ["avg_competitor", "min_competitor", "max_competitor"]
    scacs = [c for c in pivot.columns if c not in derived]

    if OUR_SCAC in scacs:
        scacs = [OUR_SCAC] + [c for c in scacs if c != OUR_SCAC]

    pivot = pivot[derived + scacs]
    pivot.insert(3, "│", "")

    if OUR_SCAC in pivot.columns:
        pivot = pivot.rename(columns={OUR_SCAC: f"{OUR_SCAC} ⭐"})

    pivot_display = pivot.reset_index().head(max_lanes)

    st.data_editor(
        pivot_display,
        use_container_width=True,
        height=650,
        disabled=True,
        column_config={
            "SROID": cc.TextColumn("SROID", width="small"),
            "ORIGIN": cc.TextColumn("ORIGIN", width="large"),
            "DESTINATION": cc.TextColumn("DESTINATION", width="large"),
            "avg_competitor": cc.NumberColumn("avg_competitor", format="%.0f", width="small"),
            "min_competitor": cc.NumberColumn("min_competitor", format="%.0f", width="small"),
            "max_competitor": cc.NumberColumn("max_competitor", format="%.0f", width="small"),
            "│": cc.TextColumn(" ", width="small"),
        }
    )

    csv_data = pivot.to_csv().encode("utf-8")
    st.download_button(
        "Download Pivot CSV",
        data=csv_data,
        file_name="tmss_pivot.csv",
        mime="text/csv"
    )