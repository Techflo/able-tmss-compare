import duckdb
import polars as pl
import pandas as pd
import streamlit as st
from streamlit import column_config as cc

st.set_page_config(page_title="TMSS Rate Comparison MVP", layout="wide")

st.title("TMSS Rate Comparison (MVP)")
st.caption("Upload TMSS exports (.xlsx/.csv). Filter lanes, pivot SCACs across top, and compare.")

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

OUR_SCAC = "AMJV"  # change if needed

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
        {"uab_weight_break_kg": "45–133 kg", "uab_baseline": 1.26, "uab_actual_gross_weight": 133},
        {"uab_weight_break_kg": "134–224 kg", "uab_baseline": 1.14, "uab_actual_gross_weight": 224},
        {"uab_weight_break_kg": "225–314 kg", "uab_baseline": 1.09, "uab_actual_gross_weight": 314},
        {"uab_weight_break_kg": "315–404 kg", "uab_baseline": 1.04, "uab_actual_gross_weight": 404},
        {"uab_weight_break_kg": "405+ kg", "uab_baseline": 0.99, "uab_actual_gross_weight": 1000},
    ]
)

DEFAULT_POV_TABLE = pd.DataFrame(
    [
        {"category": "CAT 2", "baseline": 21000.0},
    ]
)

# Global TMSS rate (per your rule: 1000% = x10)
DEFAULT_TMSS_RATE = 1000.0

# ======================================================
# UTILITIES
# ======================================================

def reset_app():
    for k in ["data", "con", "filter_lists"]:
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

    # Hide progress UI once done
    status.empty()
    progress_bar.empty()

    return out


def register_in_duckdb(df: pl.DataFrame):
    con = duckdb.connect(database=":memory:")
    con.register("tmss", df.to_arrow())
    return con


def normalize_to_percent_expr(colname: str) -> pl.Expr:
    """
    Normalize factor-like numeric to WHOLE percent (e.g., 85.0 for 85%).
    - decimals: 0.85 -> 85.0
    - whole %: 85 -> 85.0
    - TMSS-style: 850 -> 85.0, 1000 -> 100.0
    """
    return (
        pl.when(pl.col(colname).is_null())
        .then(pl.lit(None))
        .when(pl.col(colname) <= 2.0)
        .then(pl.col(colname) * 100.0)
        .when(pl.col(colname) > 200.0)
        .then(pl.col(colname) / 10.0)
        .otherwise(pl.col(colname))
    )


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
        raise ValueError(f"HHG baseline row not found for weight break: {break_label}")
    return row.iloc[0].to_dict()


def get_uab_row(uab_df: pd.DataFrame, break_label: str) -> dict:
    row = uab_df.loc[uab_df["uab_weight_break_kg"] == break_label]
    if row.empty:
        raise ValueError(f"UAB baseline row not found for weight break: {break_label}")
    return row.iloc[0].to_dict()


def tmss_rate_to_multiplier(tmss_rate_value: float) -> float:
    """
    Your rule: TMSS rate 1000% = x10  => multiplier = tmss_rate / 100
      1000 -> 10.0
      850  -> 8.5
      100  -> 1.0
    """
    if tmss_rate_value is None or pd.isna(tmss_rate_value):
        return 1.0
    return float(tmss_rate_value) / 100.0


# ======================================================
# SESSION STATE DEFAULTS
# ======================================================

if "hhg_table" not in st.session_state:
    st.session_state["hhg_table"] = DEFAULT_HHG_TABLE.copy()

if "uab_table" not in st.session_state:
    st.session_state["uab_table"] = DEFAULT_UAB_TABLE.copy()

if "pov_table" not in st.session_state:
    st.session_state["pov_table"] = DEFAULT_POV_TABLE.copy()

if "tmss_rate_global" not in st.session_state:
    st.session_state["tmss_rate_global"] = DEFAULT_TMSS_RATE


# ======================================================
# TABS
# ======================================================

tab_compare, tab_baselines = st.tabs(["Compare", "Baselines & Assumptions"])


# ======================================================
# BASELINES TAB
# ======================================================

with tab_baselines:
    st.subheader("Baselines & Assumptions (Editable)")

    st.write("### Global TMSS Rate (applies across all calculations)")
    st.session_state["tmss_rate_global"] = st.number_input(
        "TMSS Rate (percent where 1000% = x10)",
        min_value=0.0,
        value=float(st.session_state["tmss_rate_global"]),
        step=50.0,
    )

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

    st.write("### POV baseline (for reference)")
    st.session_state["pov_table"] = st.data_editor(
        st.session_state["pov_table"],
        use_container_width=True,
        height=140,
        num_rows="dynamic",
        column_config={
            "category": cc.TextColumn("Category", width="small"),
            "baseline": cc.NumberColumn("Baseline ($)", format="%.0f", width="small"),
        },
        key="pov_editor",
    )

    st.info("These settings apply when using the TOTAL RELO metrics in the Compare tab.")


# ======================================================
# COMPARE TAB
# ======================================================

with tab_compare:

    # ----------------------------
    # Upload Step
    # ----------------------------
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

    # Sidebar filters
    st.sidebar.header("Filters")
    sel_sroid = st.sidebar.multiselect("SROID", lists["sroid"])
    sel_origin = st.sidebar.multiselect("Origin", lists["origin"])
    sel_dest = st.sidebar.multiselect("Destination", lists["dest"])
    sel_scac = st.sidebar.multiselect("SCAC", lists["scac"])

    # Display controls
    st.sidebar.header("Display")
    max_lanes = st.sidebar.slider("Max lanes to show (rows)", 100, 5000, 1000, 100)

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

    # Normalize percent-like columns to percent (85.0 == 85%)
    rows = rows.with_columns([
        normalize_to_percent_expr("LINEHAUL").alias("LINEHAUL"),
        normalize_to_percent_expr("UNACCOMPANIEDAIRBAG").alias("UNACCOMPANIEDAIRBAG"),
        normalize_to_percent_expr("STORAGEINTRANSIT").alias("STORAGEINTRANSIT"),
        normalize_to_percent_expr("ACCESSORIALS").alias("ACCESSORIALS"),
    ])

    st.write("## Chunk Pivot View")

    metric_options = [
        "HHE %",
        "UAB %",
        "Vehicle Cat 2 (flat)",
        "TOTAL RELO HHE+UAB",
        "TOTAL RELO HHE+UAB+POV",
    ]
    metric_label = st.selectbox("Metric", metric_options, key="metric_select")

    # Pull baseline tables
    hhg_df = st.session_state["hhg_table"].copy()
    uab_df = st.session_state["uab_table"].copy()

    # Only show baseline dropdowns when they matter
    needs_baselines = metric_label in ["TOTAL RELO HHE+UAB", "TOTAL RELO HHE+UAB+POV"]

    hhg_break = None
    uab_break = None

    if needs_baselines:
        hhg_labels, hhg_lookup = build_hhg_dropdown(hhg_df)
        uab_labels, uab_lookup = build_uab_dropdown(uab_df)

        # default N/A as you requested
        hhg_choice = st.selectbox("HHG Baseline (Weight break | Baseline)", hhg_labels, index=0, key="hhg_choice")
        uab_choice = st.selectbox("UAB Baseline (Weight break | Baseline)", uab_labels, index=0, key="uab_choice")

        hhg_break = hhg_lookup.get(hhg_choice)
        uab_break = uab_lookup.get(uab_choice)

        tmss_multiplier = tmss_rate_to_multiplier(st.session_state["tmss_rate_global"])
        st.caption(f"Using: TMSS multiplier **{tmss_multiplier:.2f}x** | HHG break: **{hhg_break or 'N/A'}** | UAB break: **{uab_break or 'N/A'}**")
    else:
        tmss_multiplier = tmss_rate_to_multiplier(st.session_state["tmss_rate_global"])

    # --- compute metric column ---
    if metric_label == "HHE %":
        rows_for_metric = rows
        metric_use = "LINEHAUL"

    elif metric_label == "UAB %":
        rows_for_metric = rows
        metric_use = "UNACCOMPANIEDAIRBAG"

    elif metric_label == "Vehicle Cat 2 (flat)":
        rows_for_metric = rows
        metric_use = "CLASSVEHICLE2"

    elif metric_label in ["TOTAL RELO HHE+UAB", "TOTAL RELO HHE+UAB+POV"]:
        if hhg_break is None or uab_break is None:
            st.warning("Select BOTH HHG and UAB baseline weight breaks (not N/A) to compute TOTAL RELO metrics.")
            st.stop()

        hhg_row = get_hhg_row(hhg_df, hhg_break)
        uab_row = get_uab_row(uab_df, uab_break)

        hhg_baseline = float(hhg_row["hhg_baseline"])
        hhg_actual_net = float(hhg_row["hhg_actual_net_weight"])

        uab_baseline = float(uab_row["uab_baseline"])
        uab_actual_gross = float(uab_row["uab_actual_gross_weight"])

        # Calculated Net Chargeable Weight = baseline * TMSS multiplier
        hhg_calc_ncw = hhg_baseline * tmss_multiplier
        uab_calc_ncw = uab_baseline * tmss_multiplier

        # HHG billable = NCW * (Actual Net Weight / 100) * (HHE factor %)
        hhg_billable = (hhg_calc_ncw * (hhg_actual_net / 100.0)) * (pl.col("LINEHAUL") / 100.0)

        # UAB billable = NCW * (Actual Gross Weight / 100) * (UAB factor %)
        uab_billable = (uab_calc_ncw * (uab_actual_gross / 100.0)) * (pl.col("UNACCOMPANIEDAIRBAG") / 100.0)

        total_expr = hhg_billable + uab_billable

        if metric_label == "TOTAL RELO HHE+UAB+POV":
            # CONFIRMED: CLASSVEHICLE2 is a FLAT dollar value → add directly
            total_expr = total_expr + pl.col("CLASSVEHICLE2")

        rows_for_metric = rows.with_columns([total_expr.alias("TOTAL_RELO")])
        metric_use = "TOTAL_RELO"

    else:
        rows_for_metric = rows
        metric_use = "LINEHAUL"

    chunk_pd = rows_for_metric.select(["SROID", "ORIGIN", "DESTINATION", "SCAC", metric_use]).to_pandas()

    if chunk_pd.empty:
        st.info("No rows available for this chunk.")
        st.stop()

    pivot = chunk_pd.pivot_table(
        index=["SROID", "ORIGIN", "DESTINATION"],
        columns="SCAC",
        values=metric_use,
        aggfunc="first"
    ).sort_index()

    pivot["min_competitor"] = pivot.min(axis=1, numeric_only=True)
    pivot["max_competitor"] = pivot.max(axis=1, numeric_only=True)
    pivot["avg_competitor"] = pivot.mean(axis=1, numeric_only=True)

    pivot[["min_competitor", "max_competitor", "avg_competitor"]] = (
        pivot[["min_competitor", "max_competitor", "avg_competitor"]].round(0)
    )

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