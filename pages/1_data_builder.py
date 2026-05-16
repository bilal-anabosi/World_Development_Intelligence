"""
pages/1_data_builder.py — World Data Builder

Supports:
1. Live World Bank API mode
2. Local CSV mode from world_database.csv

No derived indicators are created here.
All indicators are either from the local CSV or real World Bank API indicators.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import time
import io
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import (
    CSS,
    render_sidebar,
    init_state,
    page_header,
    step_bar,
    stat_row,
    geo_layout,
)


# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Build World Data · World Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CSS, unsafe_allow_html=True)
init_state()
render_sidebar()
step_bar(0)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_label(name: str) -> str:
    return str(name).replace("_", " ").title()


def info_box(text: str):
    st.markdown(
        f"""
        <div class="info-strip">{text}</div>
        """,
        unsafe_allow_html=True,
    )


def ok_box(text: str):
    st.markdown(
        f"""
        <div class="ok-strip">{text}</div>
        """,
        unsafe_allow_html=True,
    )


def warn_box(text: str):
    st.markdown(
        f"""
        <div class="warn-strip">{text}</div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_local_world():
    path = os.path.join(os.path.dirname(__file__), "..", "world_database.csv")
    return pd.read_csv(path)


def normalize_country_code(x):
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Real World Bank indicators
# Format:
# output_column: (World Bank indicator code, readable name, domain)
# ─────────────────────────────────────────────────────────────────────────────

WB_INDICATORS = {
    # Economic
    "gdp_per_capita": ("NY.GDP.PCAP.CD", "GDP per capita", "Economic"),
    "gdp": ("NY.GDP.MKTP.CD", "GDP", "Economic"),
    "gdp_growth": ("NY.GDP.MKTP.KD.ZG", "GDP growth", "Economic"),
    "inflation": ("FP.CPI.TOTL.ZG", "Inflation", "Economic"),
    "unemployment_rate": ("SL.UEM.TOTL.ZS", "Unemployment rate", "Economic"),
    "trade_pct_gdp": ("NE.TRD.GNFS.ZS", "Trade as % of GDP", "Economic"),
    "exports_pct_gdp": ("NE.EXP.GNFS.ZS", "Exports as % of GDP", "Economic"),
    "imports_pct_gdp": ("NE.IMP.GNFS.ZS", "Imports as % of GDP", "Economic"),
    "foreign_direct_investment": ("BX.KLT.DINV.WD.GD.ZS", "Foreign direct investment", "Economic"),
    "population": ("SP.POP.TOTL", "Population", "Economic"),
    "population_growth": ("SP.POP.GROW", "Population growth", "Economic"),

    # Human development / health
    "life_expectancy": ("SP.DYN.LE00.IN", "Life expectancy", "Human development"),
    "fertility_rate": ("SP.DYN.TFRT.IN", "Fertility rate", "Human development"),
    "infant_mortality": ("SP.DYN.IMRT.IN", "Infant mortality", "Human development"),
    "under_5_mortality": ("SH.DYN.MORT", "Under-5 mortality", "Human development"),
    "literacy_rate": ("SE.ADT.LITR.ZS", "Literacy rate", "Human development"),
    "school_enrollment_primary": ("SE.PRM.ENRR", "Primary school enrollment", "Human development"),
    "school_enrollment_secondary": ("SE.SEC.ENRR", "Secondary school enrollment", "Human development"),
    "school_enrollment_tertiary": ("SE.TER.ENRR", "Tertiary school enrollment", "Human development"),
    "health_spending_pct": ("SH.XPD.CHEX.GD.ZS", "Health expenditure as % of GDP", "Human development"),
    "hospital_beds": ("SH.MED.BEDS.ZS", "Hospital beds per 1,000 people", "Human development"),
    "physicians": ("SH.MED.PHYS.ZS", "Physicians per 1,000 people", "Human development"),

    # Technology / infrastructure
    "internet_usage_pct": ("IT.NET.USER.ZS", "Internet usage", "Technology & infrastructure"),
    "mobile_subscriptions": ("IT.CEL.SETS.P2", "Mobile subscriptions per 100 people", "Technology & infrastructure"),
    "fixed_broadband": ("IT.NET.BBND.P2", "Fixed broadband subscriptions", "Technology & infrastructure"),
    "electricity_access": ("EG.ELC.ACCS.ZS", "Access to electricity", "Technology & infrastructure"),
    "electric_power_consumption": ("EG.USE.ELEC.KH.PC", "Electric power consumption", "Technology & infrastructure"),
    "urbanization_rate": ("SP.URB.TOTL.IN.ZS", "Urban population %", "Technology & infrastructure"),

    # Social / inequality
    "gini_index": ("SI.POV.GINI", "Gini index", "Social"),
    "poverty_headcount": ("SI.POV.DDAY", "Extreme poverty headcount", "Social"),
    "education_spending_pct": ("SE.XPD.TOTL.GD.ZS", "Government education spending", "Social"),

    # Governance
    "political_stability": ("PV.EST", "Political stability", "Governance"),
    "rule_of_law": ("RL.EST", "Rule of law", "Governance"),
    "control_of_corruption": ("CC.EST", "Control of corruption", "Governance"),
    "government_effectiveness": ("GE.EST", "Government effectiveness", "Governance"),

    # Environment
    "co2_per_capita": ("EN.ATM.CO2E.PC", "CO2 emissions per capita", "Environment"),
    "renewable_energy_pct": ("EG.FEC.RNEW.ZS", "Renewable energy consumption", "Environment"),
    "forest_area_pct": ("AG.LND.FRST.ZS", "Forest area", "Environment"),
    "access_to_water_pct": ("SH.H2O.BASW.ZS", "Access to basic water", "Environment"),
}


INDICATOR_GROUPS = {
    "💰 Economic": [
        "gdp_per_capita",
        "gdp",
        "gdp_growth",
        "inflation",
        "unemployment_rate",
        "trade_pct_gdp",
        "exports_pct_gdp",
        "imports_pct_gdp",
        "foreign_direct_investment",
        "population",
        "population_growth",
    ],
    "🌱 Human development": [
        "life_expectancy",
        "fertility_rate",
        "infant_mortality",
        "under_5_mortality",
        "literacy_rate",
        "school_enrollment_primary",
        "school_enrollment_secondary",
        "school_enrollment_tertiary",
        "health_spending_pct",
        "hospital_beds",
        "physicians",
    ],
    "💻 Technology & infrastructure": [
        "internet_usage_pct",
        "mobile_subscriptions",
        "fixed_broadband",
        "electricity_access",
        "electric_power_consumption",
        "urbanization_rate",
    ],
    "🏙️ Social": [
        "gini_index",
        "poverty_headcount",
        "education_spending_pct",
    ],
    "⚖️ Governance": [
        "political_stability",
        "rule_of_law",
        "control_of_corruption",
        "government_effectiveness",
    ],
    "🌿 Environment": [
        "co2_per_capita",
        "renewable_energy_pct",
        "forest_area_pct",
        "access_to_water_pct",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# World Bank API fetcher — fixed reliable version
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_wb(codes, selected_indicators, start_year, end_year):
    """
    Fetches World Bank indicators safely.

    Fixes:
    - Uses requests instead of urllib for more reliable handling.
    - Batches countries to avoid long URLs.
    - Uses pagination.
    - Retries failed requests.
    - Continues if one indicator or country batch fails.
    - Keeps partial data instead of failing the whole build.
    - Drops rows where all selected indicators are missing.
    """

    codes = [normalize_country_code(c) for c in codes]
    codes = [c for c in codes if c]

    if not codes:
        return pd.DataFrame(), []

    all_frames = []
    failed = []

    for out_col in selected_indicators:
        if out_col not in WB_INDICATORS:
            failed.append(out_col)
            continue

        wb_code, readable_name, domain = WB_INDICATORS[out_col]
        rows = []

        for batch in chunks(codes, 20):
            country_part = ";".join(batch)

            page = 1
            total_pages = 1

            while page <= total_pages:
                url = f"https://api.worldbank.org/v2/country/{country_part}/indicator/{wb_code}"

                params = {
                    "format": "json",
                    "date": f"{int(start_year)}:{int(end_year)}",
                    "per_page": 20000,
                    "page": page,
                }

                response_json = None
                success = False

                for attempt in range(3):
                    try:
                        response = requests.get(
                            url,
                            params=params,
                            timeout=25,
                            headers={
                                "User-Agent": "Mozilla/5.0 WorldDashboardDataBuilder/1.0"
                            },
                        )

                        if response.status_code == 200:
                            response_json = response.json()
                            success = True
                            break

                    except Exception:
                        time.sleep(1 + attempt)

                if not success:
                    break

                if not isinstance(response_json, list) or len(response_json) < 2:
                    break

                metadata = response_json[0] or {}
                records = response_json[1]

                if not records:
                    break

                total_pages = int(metadata.get("pages", 1))

                for r in records:
                    value = safe_float(r.get("value"))

                    if value is None:
                        continue

                    country_obj = r.get("country", {}) or {}

                    country_name = country_obj.get("value")
                    code = normalize_country_code(r.get("countryiso3code"))
                    year = r.get("date")

                    if not country_name or not code or year is None:
                        continue

                    rows.append(
                        {
                            "country": country_name,
                            "code": code,
                            "year": int(year),
                            out_col: value,
                        }
                    )

                page += 1

            time.sleep(0.05)

        if rows:
            frame = pd.DataFrame(rows)
            frame[out_col] = pd.to_numeric(frame[out_col], errors="coerce")
            frame = frame.dropna(subset=[out_col])

            if not frame.empty:
                frame = frame[["country", "code", "year", out_col]]
                all_frames.append(frame)
            else:
                failed.append(out_col)
        else:
            failed.append(out_col)

    if not all_frames:
        return pd.DataFrame(), failed

    merged = all_frames[0]

    for frame in all_frames[1:]:
        merged = merged.merge(frame, on=["country", "code", "year"], how="outer")

    indicator_cols = [c for c in selected_indicators if c in merged.columns]

    if indicator_cols:
        merged = merged.dropna(subset=indicator_cols, how="all")

    merged = merged.drop_duplicates(subset=["country", "code", "year"])

    return merged.reset_index(drop=True), failed


# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────

page_header(
    "📦",
    "Build World Data",
    "Build a country development dataset either from the live World Bank API or from the included local CSV.",
    tags=[
        ("World Bank API", "blue"),
        ("Local CSV", "green"),
        ("Countries", "purple"),
        ("Years", "amber"),
    ],
    accent="linear-gradient(90deg,#38BDF8,#818CF8)",
)


# ─────────────────────────────────────────────────────────────────────────────
# Load local reference data
# ─────────────────────────────────────────────────────────────────────────────

try:
    world = load_local_world()
except Exception as e:
    st.error(f"Could not load world_database.csv: {e}")
    st.stop()

if "code" in world.columns:
    world["code"] = world["code"].apply(normalize_country_code)

if "year" in world.columns:
    world["year"] = pd.to_numeric(world["year"], errors="coerce")

all_countries = (
    sorted(world["country"].dropna().unique().tolist())
    if "country" in world.columns
    else []
)

all_regions = (
    sorted(world["region"].dropna().unique().tolist())
    if "region" in world.columns
    else []
)

all_years = (
    sorted(world["year"].dropna().astype(int).unique().tolist())
    if "year" in world.columns and world["year"].notna().any()
    else list(range(2000, 2024))
)

country_to_code = (
    world.dropna(subset=["country", "code"])
    .drop_duplicates("country")
    .set_index("country")["code"]
    .to_dict()
    if "country" in world.columns and "code" in world.columns
    else {}
)

code_to_region = (
    world.dropna(subset=["code", "region"])
    .drop_duplicates("code")
    .set_index("code")["region"]
    .to_dict()
    if "code" in world.columns and "region" in world.columns
    else {}
)

country_to_region = (
    world.dropna(subset=["country", "region"])
    .drop_duplicates("country")
    .set_index("country")["region"]
    .to_dict()
    if "country" in world.columns and "region" in world.columns
    else {}
)


# ─────────────────────────────────────────────────────────────────────────────
# Data source
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="sec-header">Dataset Source</div>', unsafe_allow_html=True)

source = st.radio(
    "Choose how to build the dataset",
    ["Live World Bank API", "Local built-in CSV"],
    horizontal=True,
)

if source == "Live World Bank API":
    info_box(
        "🌐 Live API mode fetches real World Bank indicators. "
        "This version uses batching, retries, and pagination to avoid the API failure you had before."
    )
else:
    info_box(
        "💾 Local CSV mode is faster and safer for the demo. "
        "It uses the included world_database.csv file."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Selection layout
# ─────────────────────────────────────────────────────────────────────────────

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown(
        '<div class="g-card"><div class="g-card-title">🌍 Countries and Years</div>',
        unsafe_allow_html=True,
    )

    select_mode = st.radio(
        "Select countries",
        ["By Region", "By Country"],
        horizontal=True,
    )

    if select_mode == "By Region":
        default_regions = all_regions if all_regions else []

        selected_regions = st.multiselect(
            "Regions",
            all_regions,
            default=default_regions,
        )

        if selected_regions and "region" in world.columns:
            selected_countries = (
                world[world["region"].isin(selected_regions)]["country"]
                .dropna()
                .unique()
                .tolist()
            )
        else:
            selected_countries = []

        st.caption(
            f"{len(selected_countries)} countries selected from {len(selected_regions)} region(s)."
        )

    else:
        default_countries = [
            c for c in [
                "Palestine",
                "Jordan",
                "Egypt",
                "Lebanon",
                "Turkey",
                "Germany",
                "France",
                "United States",
                "China",
                "India",
                "Brazil",
                "Nigeria",
                "South Africa",
            ]
            if c in all_countries
        ]

        selected_countries = st.multiselect(
            "Countries",
            all_countries,
            default=default_countries if default_countries else all_countries[:10],
        )

    min_year = int(min(all_years)) if all_years else 2000
    max_year = int(max(all_years)) if all_years else 2023

    default_start = max(min_year, 2010)
    default_end = max_year

    year_range = st.slider(
        "Year range",
        min_year,
        max_year,
        (default_start, default_end),
    )

    aggregation = st.selectbox(
        "Aggregation",
        [
            "Latest year only",
            "All years panel",
            "Average over range",
        ],
    )

    st.markdown("</div>", unsafe_allow_html=True)


with right:
    st.markdown(
        '<div class="g-card"><div class="g-card-title">📊 Real Indicators</div>',
        unsafe_allow_html=True,
    )

    selected_indicators = []

    for group_name, cols in INDICATOR_GROUPS.items():
        available = []

        for col in cols:
            if source == "Live World Bank API":
                if col in WB_INDICATORS:
                    available.append(col)
            else:
                if col in world.columns:
                    available.append(col)

        if not available:
            continue

        default = available[: min(3, len(available))]

        chosen = st.multiselect(
            group_name,
            available,
            default=default,
            format_func=lambda x: WB_INDICATORS[x][1] if x in WB_INDICATORS else safe_label(x),
        )

        selected_indicators.extend(chosen)

    selected_indicators = list(dict.fromkeys(selected_indicators))

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Build dataset
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="sec-header">Build Dataset</div>', unsafe_allow_html=True)

build = st.button("🚀 Build and Load Dataset", type="primary")

if build:
    if not selected_countries:
        st.error("Please select at least one country.")
        st.stop()

    if not selected_indicators:
        st.error("Please select at least one indicator.")
        st.stop()

    if source == "Live World Bank API":
        selected_codes = [
            country_to_code.get(c)
            for c in selected_countries
            if country_to_code.get(c)
        ]

        selected_codes = [normalize_country_code(c) for c in selected_codes if c]

        if not selected_codes:
            st.error("No valid ISO country codes found for selected countries.")
            st.stop()

        with st.spinner("Fetching data from World Bank API..."):
            built, failed_indicators = fetch_wb(
                selected_codes,
                selected_indicators,
                int(year_range[0]),
                int(year_range[1]),
            )

        if built.empty:
            st.error(
                "Live API returned no usable data. Try fewer countries, fewer indicators, "
                "a different year range, or use Local CSV mode for the demo."
            )
            st.stop()

        built["region"] = built["code"].map(code_to_region).fillna("Unknown")

        # Keep only selected countries
        built = built[built["code"].isin(selected_codes)].copy()

        # Prefer local CSV country names if available
        code_to_country = {
            v: k for k, v in country_to_code.items()
            if isinstance(v, str) and v.strip()
        }

        built["country"] = built["code"].map(code_to_country).fillna(built["country"])

        if failed_indicators:
            failed_readable = [
                WB_INDICATORS[x][1] if x in WB_INDICATORS else x
                for x in failed_indicators
            ]

            warn_box(
                "⚠ Some indicators returned no data from the API and were skipped: "
                + ", ".join(failed_readable[:8])
                + ("..." if len(failed_readable) > 8 else "")
            )

    else:
        sub = world[
            world["country"].isin(selected_countries)
            & (world["year"].astype(int) >= int(year_range[0]))
            & (world["year"].astype(int) <= int(year_range[1]))
        ].copy()

        keep_cols = ["country", "code", "region", "year"] + [
            c for c in selected_indicators if c in sub.columns
        ]

        built = sub[keep_cols].copy()

    # ─────────────────────────────────────────────────────────────────────────
    # Aggregation
    # ─────────────────────────────────────────────────────────────────────────

    indicator_cols = [c for c in selected_indicators if c in built.columns]

    if aggregation == "Latest year only":
        # Latest available year per country, not one global year.
        built = (
            built.sort_values("year")
            .groupby("code", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )

    elif aggregation == "Average over range":
        group_cols = ["country", "code", "region"]

        built = (
            built.groupby(group_cols, dropna=False)[indicator_cols]
            .mean(numeric_only=True)
            .reset_index()
        )

        built["year"] = f"{year_range[0]}–{year_range[1]} avg"

        ordered_cols = ["country", "code", "region", "year"] + indicator_cols
        built = built[[c for c in ordered_cols if c in built.columns]]

    else:
        # All years panel
        built = built.sort_values(["country", "year"]).reset_index(drop=True)

    # Clean final dataset
    built = built.drop_duplicates().reset_index(drop=True)

    # Store for the next pages
    st.session_state.raw_df = built
    st.session_state.clean_df = None
    st.session_state.dataset_name = (
        "World Bank API Dataset"
        if source == "Live World Bank API"
        else "Local World Dataset"
    )

    ok_box("✅ Dataset built and loaded. Go to Prepare Data next.")

    stat_row(
        [
            ("Rows", f"{len(built):,}", "records", "#7C3AED"),
            ("Columns", f"{len(built.columns):,}", "total columns", "#2563EB"),
            (
                "Countries",
                f"{built['code'].nunique():,}" if "code" in built.columns else "—",
                "unique countries",
                "#059669",
            ),
            ("Indicators", f"{len(indicator_cols):,}", "selected indicators", "#D97706"),
        ]
    )

    st.dataframe(built.head(30), use_container_width=True)

    buffer = io.StringIO()
    built.to_csv(buffer, index=False)

    st.download_button(
        "⬇️ Download Built Dataset CSV",
        buffer.getvalue(),
        file_name="built_world_dataset.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Preview loaded dataset
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.raw_df is not None:
    df_preview = st.session_state.raw_df.copy()

    st.markdown('<div class="sec-header">Quick Map Preview</div>', unsafe_allow_html=True)

    num_cols = df_preview.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "year"]

    if "code" in df_preview.columns and num_cols:
        map_col = st.selectbox(
            "Indicator to map",
            num_cols,
            format_func=lambda x: WB_INDICATORS[x][1] if x in WB_INDICATORS else safe_label(x),
        )

        map_df = df_preview.copy()

        if "year" in map_df.columns and pd.api.types.is_numeric_dtype(map_df["year"]):
            map_df = (
                map_df.sort_values("year")
                .groupby("code", as_index=False)
                .tail(1)
            )

        fig = px.choropleth(
            map_df,
            locations="code",
            color=map_col,
            hover_name="country" if "country" in map_df.columns else "code",
            color_continuous_scale="Viridis",
            labels={
                map_col: WB_INDICATORS[map_col][1] if map_col in WB_INDICATORS else safe_label(map_col)
            },
        )

        # Palestine visibility marker
        if "PSE" in map_df["code"].astype(str).values:
            fig.add_scattergeo(
                lon=[35.2332],
                lat=[31.9522],
                text=["Palestine"],
                mode="markers+text",
                marker=dict(size=10, color="red", symbol="circle"),
                textposition="top center",
                name="Palestine",
            )

        fig.update_layout(
            **geo_layout(),
            title=f"{WB_INDICATORS[map_col][1] if map_col in WB_INDICATORS else safe_label(map_col)} — World Map",
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Map preview requires a code column and at least one numeric indicator.")