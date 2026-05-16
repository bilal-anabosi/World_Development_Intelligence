import os, io, math, urllib.parse, urllib.request, json
import streamlit as st
import os, json, textwrap
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

PAL = ["#7C3AED", "#2563EB", "#059669", "#D97706", "#DC2626", "#0891B2", "#9333EA", "#0EA5E9", "#65A30D", "#EA580C"]

FEATURE_LABELS = {
    "gdp_per_capita":"GDP per capita", "gdp_growth":"GDP growth", "population":"Population",
    "hdi":"Human Development Index", "life_expectancy":"Life expectancy", "literacy_rate":"Literacy rate",
    "internet_usage_pct":"Internet usage", "co2_per_capita":"CO₂ per capita", "fertility_rate":"Fertility rate",
    "infant_mortality":"Infant mortality", "maternal_mortality":"Maternal mortality", "education_spending_pct":"Education spending",
    "health_spending_pct":"Health spending", "gini_index":"Gini inequality index", "political_stability":"Political stability",
    "urbanization_rate":"Urbanization", "unemployment_rate":"Unemployment", "renewable_energy_pct":"Renewable energy",
    "access_to_water_pct":"Access to basic water", "inflation_pct":"Inflation", "trade_pct_gdp":"Trade (% of GDP)",
    "electricity_access_pct":"Electricity access", "secondary_enrollment_pct":"Secondary school enrollment", "forest_area_pct":"Forest area",
    "voice_accountability":"Voice and accountability", "government_effectiveness":"Government effectiveness", "rule_of_law":"Rule of law",
    "control_corruption":"Control of corruption", "regulatory_quality":"Regulatory quality", "military_expenditure_pct":"Military expenditure"
}

DOMAIN_MAP = {
    "Economic": ["gdp_per_capita","gdp_growth","unemployment_rate","gini_index","inflation_pct","trade_pct_gdp"],
    "Human development": ["hdi","life_expectancy","literacy_rate","infant_mortality","maternal_mortality","secondary_enrollment_pct"],
    "Technology & infrastructure": ["internet_usage_pct","electricity_access_pct","urbanization_rate"],
    "Social": ["fertility_rate","education_spending_pct","health_spending_pct","access_to_water_pct"],
    "Governance": ["political_stability","voice_accountability","government_effectiveness","rule_of_law","control_corruption","regulatory_quality"],
    "Environment": ["co2_per_capita","renewable_energy_pct","forest_area_pct"],
}

WB_INDICATORS = {
    "gdp_per_capita": ("NY.GDP.PCAP.CD", "GDP per capita", "Economic"),
    "gdp_growth": ("NY.GDP.MKTP.KD.ZG", "GDP growth", "Economic"),
    "population": ("SP.POP.TOTL", "Population", "Economic"),
    "inflation_pct": ("FP.CPI.TOTL.ZG", "Inflation", "Economic"),
    "trade_pct_gdp": ("NE.TRD.GNFS.ZS", "Trade (% of GDP)", "Economic"),
    "unemployment_rate": ("SL.UEM.TOTL.ZS", "Unemployment", "Economic"),
    "gini_index": ("SI.POV.GINI", "Gini inequality index", "Economic"),
    "life_expectancy": ("SP.DYN.LE00.IN", "Life expectancy", "Human development"),
    "literacy_rate": ("SE.ADT.LITR.ZS", "Literacy rate", "Human development"),
    "infant_mortality": ("SP.DYN.IMRT.IN", "Infant mortality", "Human development"),
    "maternal_mortality": ("SH.STA.MMRT", "Maternal mortality", "Human development"),
    "secondary_enrollment_pct": ("SE.SEC.ENRR", "Secondary school enrollment", "Human development"),
    "internet_usage_pct": ("IT.NET.USER.ZS", "Internet usage", "Technology & infrastructure"),
    "electricity_access_pct": ("EG.ELC.ACCS.ZS", "Electricity access", "Technology & infrastructure"),
    "urbanization_rate": ("SP.URB.TOTL.IN.ZS", "Urbanization", "Technology & infrastructure"),
    "education_spending_pct": ("SE.XPD.TOTL.GD.ZS", "Education spending", "Social"),
    "health_spending_pct": ("SH.XPD.CHEX.GD.ZS", "Health spending", "Social"),
    "fertility_rate": ("SP.DYN.TFRT.IN", "Fertility rate", "Social"),
    "access_to_water_pct": ("SH.H2O.BASW.ZS", "Access to basic water", "Social"),
    "political_stability": ("PV.EST", "Political stability", "Governance"),
    "voice_accountability": ("VA.EST", "Voice and accountability", "Governance"),
    "government_effectiveness": ("GE.EST", "Government effectiveness", "Governance"),
    "rule_of_law": ("RL.EST", "Rule of law", "Governance"),
    "control_corruption": ("CC.EST", "Control of corruption", "Governance"),
    "regulatory_quality": ("RQ.EST", "Regulatory quality", "Governance"),
    "co2_per_capita": ("EN.ATM.CO2E.PC", "CO₂ per capita", "Environment"),
    "renewable_energy_pct": ("EG.FEC.RNEW.ZS", "Renewable energy", "Environment"),
    "forest_area_pct": ("AG.LND.FRST.ZS", "Forest area", "Environment"),
    "military_expenditure_pct": ("MS.MIL.XPND.GD.ZS", "Military expenditure", "Governance"),
}

CSS = """
<style>
html, body, [data-testid="stAppViewContainer"]{background:#0B0F17!important;color:#E2E8F0!important;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}.block-container{padding-top:1rem!important;max-width:1400px}header{background:transparent!important}[data-testid="stSidebarNav"]{display:none!important}[data-testid="stSidebar"]{background:#080B12!important;border-right:1px solid #263149!important}[data-testid="stSidebar"] *{color:#CBD5E1!important}[data-testid="stSidebar"] a{color:#CBD5E1!important;font-weight:700!important;text-decoration:none!important}[data-testid="stSidebar"] a:hover{color:#fff!important;background:#1E293B!important;border-radius:10px!important}.sidebar-logo{padding:1.2rem .9rem .8rem!important;border-bottom:none!important;margin-bottom:.7rem!important}.sidebar-logo .mark{color:#F8FAFC!important;background:none!important;-webkit-text-fill-color:#F8FAFC!important;font-size:1.25rem!important;font-weight:850!important}.sidebar-logo .tagline{color:#94A3B8!important;font-size:.75rem!important}.ds-pill{margin:.5rem .75rem 1rem!important;background:#111827!important;border:1px solid #334155!important;border-radius:16px!important;padding:1rem!important}.ds-pill .ds-label{color:#60A5FA!important;font-size:.72rem!important;font-weight:850!important}.ds-pill .ds-name{color:#F8FAFC!important;font-size:.98rem!important;font-weight:800!important}.ds-pill .ds-meta{color:#94A3B8!important;font-size:.78rem!important}.side-footer{margin:1.2rem .75rem .5rem;color:#94A3B8!important;font-size:.78rem;line-height:1.65}.page-header{padding:1.5rem 1.7rem;background:linear-gradient(135deg,#12172A 0%,#0F1117 100%);border:1px solid #263149;border-radius:18px;margin-bottom:1.3rem;border-top:3px solid #38BDF8}.ph-title{font-size:1.55rem;font-weight:850;color:#F8FAFC}.ph-sub{font-size:.92rem;color:#94A3B8;margin-top:.35rem;line-height:1.55}.g-card{background:#12172A;border:1px solid #263149;border-radius:16px;padding:1.2rem 1.35rem;margin-bottom:1rem}.sec-header{font-size:1rem;font-weight:850;color:#F8FAFC;border-bottom:1px solid #263149;padding-bottom:.45rem;margin:1.25rem 0 .8rem}.info-strip{background:#0F1E3D;border-left:4px solid #2563EB;border-radius:0 10px 10px 0;padding:.75rem 1rem;color:#A7C7FF;margin:.5rem 0;line-height:1.6}.warn-strip{background:#2D1A06;border-left:4px solid #D97706;border-radius:0 10px 10px 0;padding:.75rem 1rem;color:#FCD34D;margin:.5rem 0;line-height:1.6}.ok-strip{background:#052E1C;border-left:4px solid #059669;border-radius:0 10px 10px 0;padding:.75rem 1rem;color:#6EE7B7;margin:.5rem 0;line-height:1.6}.stat-row{display:flex;gap:.75rem;flex-wrap:wrap;margin-bottom:1rem}.stat-card{flex:1;min-width:150px;background:#12172A;border:1px solid #263149;border-radius:14px;padding:1rem}.s-label{font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;font-weight:800;color:var(--sc,#60A5FA)}.s-value{font-size:1.45rem;font-weight:850;color:#F8FAFC}.s-sub{font-size:.75rem;color:#94A3B8;margin-top:.2rem}.chip{display:inline-block;background:#1E293B;border:1px solid #334155;color:#CBD5E1;border-radius:20px;padding:.22rem .65rem;font-size:.76rem;margin:.15rem}.stTabs [data-baseweb="tab-list"]{background:#111827;border:1px solid #263149;border-radius:12px;padding:.35rem}.stTabs [data-baseweb="tab"]{color:#94A3B8;font-weight:700}.stTabs [aria-selected="true"]{background:#7C3AED!important;color:white!important;border-radius:9px}.stDataFrame{border-radius:12px!important}
</style>
"""

def init_state():
    defaults = dict(raw_df=None, clean_df=None, dataset_name="", pp_num_cols=[], pp_label_col=None, pp_id_col=None, pp_scaling="StandardScaler")
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v

def label(col):
    return FEATURE_LABELS.get(str(col), str(col).replace('_',' ').title())

def labels(cols):
    return [label(c) for c in cols]


def collapse_country_year_panel(df, num_cols, id_col=None, label_col=None):
    """
    Return a country-level version of a panel dataset for non-time-series analysis.

    Why: clustering, PCA, association rules, and anomaly detection should usually compare
    each country once. If the user built an "All years panel" dataset, this averages the
    selected numeric indicators across years per country, while keeping Time Series untouched.

    Returns: (analysis_df, analysis_num_cols, was_collapsed)
    """
    if df is None or len(df) == 0:
        return df, [], False

    out = df.copy()
    num_cols = [c for c in num_cols if c in out.columns]

    # Year is useful for time series, but it should not act as a mining feature here.
    analysis_num_cols = [c for c in num_cols if str(c).lower() != "year"]

    group_col = None
    if id_col and id_col in out.columns:
        group_col = id_col
    elif "country" in out.columns:
        group_col = "country"

    has_year = "year" in out.columns
    repeated_entities = bool(group_col and out[group_col].duplicated().any())

    if not (has_year and repeated_entities and group_col):
        return out, analysis_num_cols, False

    # Average only the real numeric analysis indicators.
    numeric_to_avg = [c for c in analysis_num_cols if c != group_col and pd.api.types.is_numeric_dtype(out[c])]
    if not numeric_to_avg:
        return out.drop_duplicates(subset=[group_col]).reset_index(drop=True), analysis_num_cols, True

    agg = {c: "mean" for c in numeric_to_avg}

    # Keep useful descriptive columns once per country.
    keep_first = []
    for c in ["code", "region", label_col]:
        if c and c in out.columns and c != group_col and c not in agg and c not in keep_first:
            keep_first.append(c)
    for c in keep_first:
        agg[c] = "first"

    analysis_df = out.groupby(group_col, as_index=False).agg(agg)

    # Optional readable period label for hover/tables; not included in numeric features.
    yrs = pd.to_numeric(out["year"], errors="coerce")
    if yrs.notna().any():
        y_min, y_max = int(yrs.min()), int(yrs.max())
        analysis_df["period"] = f"{y_min}–{y_max} average"

    # Keep a clean, predictable column order.
    ordered = [group_col]
    for c in ["code", "region", label_col, "period"]:
        if c and c in analysis_df.columns and c not in ordered:
            ordered.append(c)
    ordered += [c for c in numeric_to_avg if c in analysis_df.columns and c not in ordered]
    ordered += [c for c in analysis_df.columns if c not in ordered]
    analysis_df = analysis_df[ordered].reset_index(drop=True)

    return analysis_df, [c for c in numeric_to_avg if c in analysis_df.columns], True

def html(s):
    st.markdown(str(s).replace("\n", ""), unsafe_allow_html=True)

def page_header(icon, title, subtitle, *args, **kwargs):
    html(f'<div class="page-header"><div style="font-size:2rem">{icon}</div><div class="ph-title">{title}</div><div class="ph-sub">{subtitle}</div></div>')

def stat_row(items):
    out='<div class="stat-row">'
    for lab,val,sub,color in items:
        out += f'<div class="stat-card" style="--sc:{color};"><div class="s-label">{lab}</div><div class="s-value">{val}</div><div class="s-sub">{sub}</div></div>'
    out+='</div>'; html(out)

def fig_layout(height=420, margin=None):
    return dict(paper_bgcolor="#12172A", plot_bgcolor="#0B0F17", font=dict(color="#CBD5E1", size=11), height=height, margin=margin or dict(l=20,r=20,t=45,b=20), xaxis=dict(gridcolor="#263149", zerolinecolor="#263149"), yaxis=dict(gridcolor="#263149", zerolinecolor="#263149"), legend=dict(bgcolor="#111827", bordercolor="#263149", borderwidth=1))

def geo_layout(height=420):
    return dict(paper_bgcolor="#12172A", height=height, margin=dict(l=0,r=0,t=45,b=0), geo=dict(bgcolor="#0B0F17", showframe=False, showcoastlines=True, coastlinecolor="#334155", showland=True, landcolor="#111827", showocean=True, oceancolor="#07111F", projection_type="natural earth"), legend=dict(bgcolor="#111827", font=dict(color="#CBD5E1")))

def render_sidebar():
    init_state()
    with st.sidebar:
        html('<div class="sidebar-logo"><div class="mark">🌍 World Dashboard</div><div class="tagline">Data mining for country development patterns</div></div>')
        raw, clean = st.session_state.raw_df, st.session_state.clean_df
        if raw is None:
            html('<div class="ds-pill"><div class="ds-label">DATASET</div><div class="ds-name">No data loaded</div><div class="ds-meta">Start from Data Builder</div></div>')
        else:
            status = "Ready for analysis" if clean is not None else "Needs preprocessing"
            html(f'<div class="ds-pill"><div class="ds-label">DATASET</div><div class="ds-name">{st.session_state.dataset_name or "Dataset"}</div><div class="ds-meta">{len(raw):,} rows · {len(raw.columns)} cols · {status}</div></div>')
        nav=[("🏠  Home","app.py"),("📦  Build World Data","pages/1_data_builder.py"),("📥  Import Data","pages/2_upload.py"),("🧹  Prepare Data","pages/3_preprocessing.py"),("🧩  Country Similarity","pages/4_clustering.py"),("🗺️  Development Map","pages/5_pca.py"),("🔗  Pattern Rules","pages/6_association_rules.py"),("📈  Trends Over Time","pages/8_time_series.py"),("🚨  Unusual Countries","pages/7_anomaly_detection.py"),("🤖  AI Country Report","pages/9_country_report.py")]
        for lab,path in nav:
            try: st.page_link(path, label=lab)
            except Exception: pass
        html('<div class="side-footer">Bilal Anabosi · Yaqoob Hanbali · Mohammad Sayeh</div>')

def step_bar(current):
    pass

def needs_data():
    if st.session_state.raw_df is None:
        st.warning("Load or build a dataset first."); st.stop()

def needs_preprocess():
    if st.session_state.clean_df is None:
        st.warning("Run preprocessing first."); st.stop()

def add_palestine_marker(fig, df, lat=31.95, lon=35.23):
    try:
        codes = set(df.get('code', pd.Series(dtype=str)).astype(str))
        names = set(df.get('country', pd.Series(dtype=str)).astype(str))
        if 'PSE' in codes or 'Palestine' in names:
            fig.add_trace(go.Scattergeo(lat=[lat], lon=[lon], text=["Palestine"], mode="markers+text", marker=dict(size=9,color="#F59E0B",line=dict(width=1,color="white")), textfont=dict(color="#FCD34D", size=11), textposition="top center", name="Palestine"))
    except Exception: pass
    return fig

def domain_summary(df, feats):
    parts=[]
    for dom, cols in DOMAIN_MAP.items():
        use=[c for c in cols if c in feats and c in df.columns]
        if use: parts.append((dom, use))
    return parts



# ─────────────────────────────────────────────────────────────────────────────
# Lightweight Groq / AI helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_local_env_once():
    """Load .env values without requiring python-dotenv."""
    if st.session_state.get("_local_env_loaded"):
        return
    candidates = [os.path.join(os.getcwd(), ".env"), os.path.join(os.path.dirname(__file__), ".env")]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass
    st.session_state["_local_env_loaded"] = True


def get_groq_api_key():
    """Accept several key names so the project works with the user's existing .env."""
    _load_local_env_once()
    for name in ["GROQ_API_KEY", "GROQ_KEY", "GQ_API_KEY", "gq", "GQ", "qroqgq"]:
        val = os.environ.get(name)
        if val:
            return val
    return None


def call_groq_json(system_prompt, user_payload, fallback=None, max_tokens=900):
    """Call Groq's OpenAI-compatible chat endpoint and parse a JSON response.

    Returns (data, error). If the call fails or JSON parsing fails, data=fallback.
    """
    api_key = get_groq_api_key()
    if not api_key:
        return fallback, "No Groq API key found. Add GROQ_API_KEY=... or gq=... to your .env file."

    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "temperature": 0.25,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=25)
        if r.status_code >= 400:
            return fallback, f"Groq API error {r.status_code}: {r.text[:220]}"
        content = r.json()["choices"][0]["message"]["content"]
        return json.loads(content), None
    except Exception as e:
        return fallback, f"AI generation failed: {e}"


def _compact_feature_name(name):
    return str(name).replace("_", " ").title()


def build_cluster_ai_context(df, labels, features, id_col):
    global_mean = df[features].mean(numeric_only=True)
    global_std = df[features].std(numeric_only=True).replace(0, 1)
    contexts = []
    for cl in sorted([c for c in set(labels) if c != -1]):
        mask = labels == cl
        sub = df.loc[mask]
        means = sub[features].mean(numeric_only=True)
        z = ((means - global_mean) / global_std).sort_values(key=lambda x: x.abs(), ascending=False)
        high = [{"feature": _compact_feature_name(f), "cluster_mean": round(float(means[f]), 3), "global_mean": round(float(global_mean[f]), 3)} for f in z.index if z[f] > 0][:4]
        low = [{"feature": _compact_feature_name(f), "cluster_mean": round(float(means[f]), 3), "global_mean": round(float(global_mean[f]), 3)} for f in z.index if z[f] < 0][:4]
        members = sub[id_col].astype(str).tolist() if id_col and id_col in sub.columns else [f"row_{i}" for i in sub.index]
        contexts.append({"cluster": int(cl), "size": int(mask.sum()), "members": members[:18], "high_indicators": high, "low_indicators": low})
    return contexts


def fallback_cluster_ai(contexts):
    out = {}
    for ctx in contexts:
        highs = [x["feature"] for x in ctx.get("high_indicators", [])[:2]]
        lows = [x["feature"] for x in ctx.get("low_indicators", [])[:2]]
        title_bits = highs or lows or ["Mixed Development"]
        name = " & ".join(title_bits) + " Group"
        reason = []
        if highs:
            reason.append("higher " + ", ".join(highs))
        if lows:
            reason.append("lower " + ", ".join(lows))
        explanation = "These countries were grouped together because their selected indicators form a similar profile, especially " + (" and ".join(reason) if reason else "their overall distances in feature space") + "."
        out[str(ctx["cluster"])] = {"name": name[:45], "explanation": explanation, "facts": "Use the indicator values shown in the card and radar chart to validate this interpretation."}
    return out


def generate_kmeans_cluster_ai(df, labels, features, id_col):
    contexts = build_cluster_ai_context(df, labels, features, id_col)
    fallback = fallback_cluster_ai(contexts)
    system = """
You are an analyst explaining K-Means clusters in a world development dashboard.
Return JSON only. For each cluster, create a short human-readable name and a concise explanation.
Use the provided country names and indicator patterns. Do not invent exact statistics beyond the values provided.
Mention partial real-world context carefully, such as development profile, income structure, health/education access, or population dynamics, but avoid unsupported claims.
Schema: {"clusters": {"0": {"name": "...", "explanation": "...", "facts": "..."}}}
""".strip()
    data, err = call_groq_json(system, {"clusters": contexts}, fallback={"clusters": fallback}, max_tokens=1200)
    clusters = data.get("clusters", fallback) if isinstance(data, dict) else fallback
    return clusters, err


def build_anomaly_ai_context(df, features, id_col, is_anom, scores, X, max_items=8):
    if not np.any(is_anom):
        return []
    normal_idx = np.where(~is_anom)[0]
    anom_idx = np.where(is_anom)[0]
    # More extreme first. For IF lower is more anomalous; for LOF higher can be more anomalous.
    # Use the displayed order by score when possible, but keep it simple and stable.
    anom_idx = anom_idx[:max_items]
    contexts = []
    global_mean = df[features].mean(numeric_only=True)
    global_std = df[features].std(numeric_only=True).replace(0, 1)
    for i in anom_idx:
        row = df.iloc[i]
        name = str(row[id_col]) if id_col and id_col in df.columns else f"row_{i}"
        if len(normal_idx) > 0:
            dists = [(j, float(np.linalg.norm(X[j] - X[i]))) for j in normal_idx]
            nearest = sorted(dists, key=lambda x: x[1])[:3]
        else:
            nearest = []
        vals = row[features]
        z = ((vals - global_mean) / global_std).sort_values(key=lambda x: x.abs(), ascending=False)
        unusual = []
        for f in z.index[:5]:
            unusual.append({"feature": _compact_feature_name(f), "value": round(float(vals[f]), 3), "average": round(float(global_mean[f]), 3), "direction": "high" if z[f] > 0 else "low"})
        contexts.append({
            "country": name,
            "score": round(float(scores[i]), 4),
            "most_unusual_indicators": unusual,
            "nearest_normal_countries": [str(df.iloc[j][id_col]) if id_col and id_col in df.columns else f"row_{j}" for j, _ in nearest],
        })
    return contexts


def fallback_anomaly_ai(contexts):
    out = {}
    for ctx in contexts:
        parts = [f"{x['direction']} {x['feature']}" for x in ctx.get("most_unusual_indicators", [])[:3]]
        sim = ", ".join(ctx.get("nearest_normal_countries", [])[:3]) or "the closest normal countries"
        out[ctx["country"]] = {
            "why": "This country is flagged because its profile is unusual compared with the dataset, especially " + (", ".join(parts) if parts else "the selected indicators") + ".",
            "similar_to": sim,
            "fact_context": "Check whether this reflects a real development pattern or a data quality issue before drawing conclusions."
        }
    return out


def generate_anomaly_ai(df, features, id_col, is_anom, scores, X):
    contexts = build_anomaly_ai_context(df, features, id_col, is_anom, scores, X)
    fallback = fallback_anomaly_ai(contexts)
    system = """
You are explaining anomaly detection results for a world development dashboard.
Return JSON only. Explain why each anomalous country is unusual using the provided indicators and nearest normal countries.
Do not invent precise facts or statistics not provided. You may add cautious real-world context, but phrase it as interpretation, not certainty.
Schema: {"anomalies": {"Country": {"why": "...", "similar_to": "...", "fact_context": "..."}}}
""".strip()
    data, err = call_groq_json(system, {"anomalies": contexts}, fallback={"anomalies": fallback}, max_tokens=1400)
    anomalies = data.get("anomalies", fallback) if isinstance(data, dict) else fallback
    return anomalies, err


def fallback_country_report(context):
    country = context.get("country", "Selected country")
    highs = context.get("highest_vs_average", [])[:4]
    lows = context.get("lowest_vs_average", [])[:4]
    similar = context.get("similar_countries", [])[:5]
    anomaly = context.get("anomaly", {})
    trend = context.get("time_trends", [])[:4]
    lines = [f"# Country Intelligence Report: {country}", ""]
    lines.append("## Executive summary")
    lines.append(f"{country} was analysed using its averaged indicator profile, nearest-country similarity, PCA position, anomaly detection status, and available time trends.")
    lines.append("")
    lines.append("## Development profile")
    if highs:
        lines.append("Strongest above-average indicators: " + ", ".join([x["feature"] for x in highs]) + ".")
    if lows:
        lines.append("Most below-average indicators: " + ", ".join([x["feature"] for x in lows]) + ".")
    lines.append("")
    lines.append("## Similar countries")
    if similar:
        lines.append("Closest countries in the selected feature space: " + ", ".join([x["country"] for x in similar]) + ".")
    else:
        lines.append("No similar countries could be calculated from the current dataset.")
    lines.append("")
    lines.append("## Anomaly interpretation")
    lines.append(f"Status: {'Anomaly' if anomaly.get('is_anomaly') else 'Normal'}.")
    lines.append(anomaly.get("reason", "The country was judged from its overall distance and indicator profile."))
    lines.append("")
    if trend:
        lines.append("## Time-series notes")
        for t in trend:
            lines.append(f"- {t['feature']}: {t['first_year']} → {t['last_year']} changed by {t['change']}.")
        lines.append("")
    lines.append("## Recommendation")
    lines.append("Use this report as an interpretation layer. Confirm important conclusions using the charts and original indicator values.")
    return "\n".join(lines)


def generate_country_report_ai(context):
    fallback = {"report_markdown": fallback_country_report(context)}
    system = """
You are generating a concise country intelligence report for a world development data-mining dashboard.
Return JSON only with key report_markdown.
Use the provided computed context only. Do not invent exact numbers. You may add careful, general real-world interpretation, but avoid unsupported claims.
The report should include: executive summary, development profile, similar countries, PCA/anomaly interpretation, time-series notes if available, and practical conclusion.
""".strip()
    data, err = call_groq_json(system, context, fallback=fallback, max_tokens=1600)
    if isinstance(data, dict) and data.get("report_markdown"):
        return data["report_markdown"], err
    return fallback["report_markdown"], err

# ─────────────────────────────────────────────────────────────────────────────
# AI FIX v2 — better JSON handling, useful fallback names, detailed reports
# ─────────────────────────────────────────────────────────────────────────────
def _json_safe(obj):
    """Convert pandas/numpy objects into clean JSON-safe Python values."""
    try:
        import pandas as _pd
        import numpy as _np
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            if not _np.isfinite(obj):
                return None
            return float(obj)
        if isinstance(obj, (_np.ndarray,)):
            return [_json_safe(x) for x in obj.tolist()]
        if isinstance(obj, (_pd.Series,)):
            return {str(k): _json_safe(v) for k, v in obj.to_dict().items()}
        if isinstance(obj, (_pd.DataFrame,)):
            return [_json_safe(r) for r in obj.to_dict(orient="records")]
        if _pd.isna(obj):
            return None
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _extract_json_object(text):
    """Parse JSON even if the model adds text around it."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            return None
    return None


def call_groq_json(system_prompt, user_payload, fallback=None, max_tokens=900):
    """Call Groq and safely parse JSON. Returns (data, error_message_or_None)."""
    api_key = get_groq_api_key()
    if not api_key:
        return fallback, "No Groq API key found. Add GROQ_API_KEY=... or gq=... to your .env file."

    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = _json_safe(user_payload)
    body = {
        "model": model,
        "temperature": 0.15,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt + "\n\nReturn valid JSON only. No markdown. No code fences."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }

    # Groq supports JSON mode for many chat models, but keeping it optional makes
    # the app more robust if a model/key behaves differently.
    if os.environ.get("GROQ_FORCE_JSON", "0") != "0":
        body["response_format"] = {"type": "json_object"}

    try:
        r = requests.post(url, headers=headers, json=body, timeout=35)
        if r.status_code >= 400:
            # Never print the giant prompt/body into the Streamlit page.
            return fallback, f"Groq API returned {r.status_code}. Check GROQ_API_KEY and GROQ_MODEL."
        content = r.json()["choices"][0]["message"]["content"]
        parsed = _extract_json_object(content)
        if parsed is None:
            return fallback, "Groq returned text that was not valid JSON."
        return parsed, None
    except Exception as e:
        return fallback, f"AI generation failed: {type(e).__name__}: {str(e)[:120]}"


_POSITIVE_DEV = [
    "gdp", "life expectancy", "internet", "electricity", "education", "school", "literacy",
    "hdi", "water", "urban", "government", "rule of law", "stability", "regulatory", "corruption"
]
_NEGATIVE_DEV = [
    "fertility", "infant", "maternal", "mortality", "unemployment", "inflation", "gini", "inequality"
]
_WEALTH = ["gdp", "internet", "electricity", "urban", "co2", "trade"]
_DEMOGRAPHIC_PRESSURE = ["fertility", "infant", "maternal", "population", "unemployment"]


def _indicator_score(items, keys):
    score = 0
    for it in items or []:
        name = str(it.get("feature", "")).lower()
        if any(k in name for k in keys):
            score += 1
    return score


def _nice_cluster_name(ctx):
    highs = ctx.get("high_indicators", [])
    lows = ctx.get("low_indicators", [])
    high_pos = _indicator_score(highs, _POSITIVE_DEV)
    low_pos = _indicator_score(lows, _POSITIVE_DEV)
    high_neg = _indicator_score(highs, _NEGATIVE_DEV)
    low_neg = _indicator_score(lows, _NEGATIVE_DEV)
    wealth = _indicator_score(highs, _WEALTH)
    pressure = _indicator_score(highs, _DEMOGRAPHIC_PRESSURE)
    low_pressure = _indicator_score(lows, _DEMOGRAPHIC_PRESSURE)
    members = ", ".join(ctx.get("members", [])[:4])

    if high_pos >= 2 and low_neg >= 1:
        return "High Development Countries"
    if wealth >= 2 and low_pressure >= 1:
        return "Advanced Wealthy Economies"
    if high_neg >= 2 and low_pos >= 1:
        return "Struggling Development Countries"
    if pressure >= 2 and low_pos >= 1:
        return "High Pressure Developing Countries"
    if high_pos >= 1 and high_neg >= 1:
        return "Uneven Transition Economies"
    if low_pos >= 2:
        return "Low Development Countries"
    if "Bahrain" in members or "Qatar" in members or "UAE" in members:
        return "Resource-Rich High Income Countries"
    return "Middle Development Countries"


def fallback_cluster_ai(contexts):
    out = {}
    for ctx in contexts:
        name = _nice_cluster_name(ctx)
        highs = ctx.get("high_indicators", [])[:3]
        lows = ctx.get("low_indicators", [])[:3]
        members = ctx.get("members", [])[:6]

        high_txt = ", ".join([f"{x['feature']} above average" for x in highs]) or "some indicators above average"
        low_txt = ", ".join([f"{x['feature']} below average" for x in lows]) or "some indicators below average"
        mem_txt = ", ".join(members)
        explanation = (
            f"This cluster looks like **{name.lower()}** because countries such as {mem_txt} share a similar indicator pattern: "
            f"{high_txt}, while {low_txt}. The grouping is based on distance across the selected features, so countries do not need to be geographically close to end up together."
        )
        facts = (
            "Use the card values and radar/heatmap to validate the label. The name is an interpretation of the data pattern, not an official country classification."
        )
        out[str(ctx["cluster"])] = {"name": name, "explanation": explanation, "facts": facts}
    return out


def _valid_cluster_output(data, fallback):
    if not isinstance(data, dict):
        return fallback
    clusters = data.get("clusters", data)
    if not isinstance(clusters, dict):
        return fallback
    cleaned = {}
    for k, fb in fallback.items():
        item = clusters.get(str(k), clusters.get(k, {}))
        if not isinstance(item, dict):
            item = {}
        name = str(item.get("name", "")).strip()
        exp = str(item.get("explanation", "")).strip()
        facts = str(item.get("facts", "")).strip()
        # Reject useless names like Cluster 0 or empty output.
        if not name or name.lower().startswith("cluster") or len(name) < 6:
            name = fb["name"]
        if not exp or len(exp) < 40:
            exp = fb["explanation"]
        if not facts or len(facts) < 20:
            facts = fb["facts"]
        cleaned[str(k)] = {"name": name[:60], "explanation": exp, "facts": facts}
    return cleaned


def generate_kmeans_cluster_ai(df, labels, features, id_col):
    contexts = build_cluster_ai_context(df, labels, features, id_col)
    fallback = fallback_cluster_ai(contexts)
    system = """
You are an expert analyst for a world development dashboard.
Task: name K-Means clusters with meaningful archetype names and explain them.

Rules:
- Return JSON exactly: {"clusters":{"0":{"name":"...","explanation":"...","facts":"..."}}}
- The name must be a readable development archetype, NOT "Cluster 0".
- Good name examples: "High Development Countries", "Advanced Wealthy Economies", "Struggling Development Countries", "High Pressure Developing Countries", "Uneven Transition Economies", "Resource-Rich High Income Countries".
- Use ONLY the countries and indicator patterns provided.
- Explanation should compare high and low indicators and mention example member countries.
- Facts must add careful real-world context, but do not invent exact statistics.
""".strip()
    data, err = call_groq_json(system, {"clusters": contexts}, fallback={"clusters": fallback}, max_tokens=1700)
    return _valid_cluster_output(data, fallback), err


def _valid_anomaly_output(data, fallback):
    if not isinstance(data, dict):
        return fallback
    anomalies = data.get("anomalies", data)
    if not isinstance(anomalies, dict):
        return fallback
    cleaned = {}
    for country, fb in fallback.items():
        item = anomalies.get(country, {})
        if not isinstance(item, dict):
            item = {}
        why = str(item.get("why", "")).strip() or fb["why"]
        similar = str(item.get("similar_to", "")).strip() or fb["similar_to"]
        fact = str(item.get("fact_context", "")).strip() or fb["fact_context"]
        if len(why) < 40:
            why = fb["why"]
        cleaned[country] = {"why": why, "similar_to": similar, "fact_context": fact}
    return cleaned


def fallback_anomaly_ai(contexts):
    out = {}
    for ctx in contexts:
        parts = [f"{x['direction']} {x['feature']} ({x['value']} vs avg {x['average']})" for x in ctx.get("most_unusual_indicators", [])[:4]]
        sim = ", ".join(ctx.get("nearest_normal_countries", [])[:3]) or "the closest normal countries"
        country = ctx["country"]
        out[country] = {
            "why": f"{country} is flagged because its combined profile is far from most countries in the selected feature space, especially: " + "; ".join(parts) + ".",
            "similar_to": f"Its nearest non-anomalous comparison countries are {sim}. This means they are closest numerically, even if the flagged country is still extreme overall.",
            "fact_context": "Interpret this with care: an anomaly can represent real exceptional development conditions, a resource-rich/small-state effect, a conflict or institutional shock, or missing/noisy data depending on the country and indicators."
        }
    return out


def generate_anomaly_ai(df, features, id_col, is_anom, scores, X):
    contexts = build_anomaly_ai_context(df, features, id_col, is_anom, scores, X)
    fallback = fallback_anomaly_ai(contexts)
    system = """
You explain anomaly detection results in a world development dashboard.
Return JSON exactly: {"anomalies":{"Country":{"why":"...","similar_to":"...","fact_context":"..."}}}

For each country:
- Explain the exact unusual indicator pattern using provided values vs averages.
- Mention the nearest normal countries and what the comparison means.
- Add careful outside context in general terms only: e.g., small rich economy, post-conflict pressure, oil/resource structure, high-income city-state, population pressure, governance/institutional issues.
- Do not output the input JSON. Do not use generic repeated wording.
""".strip()
    data, err = call_groq_json(system, {"anomalies": contexts}, fallback={"anomalies": fallback}, max_tokens=1900)
    return _valid_anomaly_output(data, fallback), err


def _feature_sentence(items, limit=6):
    bits = []
    for x in (items or [])[:limit]:
        bits.append(f"{x.get('feature')} = {x.get('value')} (dataset avg {x.get('dataset_average')})")
    return "; ".join(bits)


def fallback_country_report(context):
    country = context.get("country", "Selected country")
    highs = context.get("highest_vs_average", [])[:6]
    lows = context.get("lowest_vs_average", [])[:6]
    similar = context.get("similar_countries", [])[:5]
    comp = context.get("similar_country_comparison", [])[:5]
    anomaly = context.get("anomaly", {})
    trend = context.get("time_trends", [])[:6]

    lines = [f"# Country Intelligence Report: {country}", ""]
    lines.append("## 1. Executive summary")
    lines.append(f"{country} was analysed using the selected development indicators, averaged country profile, nearest-country similarity, PCA position, anomaly detection, and available time-series movement. The report should be read as a decision-support interpretation, not as an official statistical classification.")
    lines.append("")

    lines.append("## 2. Development profile compared with the dataset")
    if highs:
        lines.append("**Above-average signals:** " + _feature_sentence(highs) + ".")
    if lows:
        lines.append("**Below-average signals:** " + _feature_sentence(lows) + ".")
    lines.append("These values show which dimensions pull the country away from the dataset average. A country can be strong in one dimension and weak in another, so the overall story is the combination, not a single indicator.")
    lines.append("")

    lines.append("## 3. Similar-country comparison")
    if similar:
        lines.append("The closest countries in the selected feature space are: " + ", ".join([x["country"] for x in similar]) + ".")
    if comp:
        for c in comp[:3]:
            diffs = "; ".join(c.get("closest_matching_indicators", [])[:4])
            lines.append(f"- Compared with **{c.get('country')}**, the closest matching indicators are: {diffs}.")
    else:
        lines.append("No detailed similar-country comparison was available.")
    lines.append("")

    lines.append("## 4. PCA and anomaly interpretation")
    pca = context.get("pca_position", {})
    lines.append(f"PCA position: PC1={pca.get('pc1')}, PC2={pca.get('pc2')}. The first two components explain approximately {pca.get('explained_variance')} of the selected-feature variation.")
    lines.append(f"Anomaly status: **{'Anomaly' if anomaly.get('is_anomaly') else 'Normal'}** with score {anomaly.get('score')}. {anomaly.get('reason', '')}")
    lines.append("")

    if trend:
        lines.append("## 5. Time-series movement")
        for t in trend:
            direction = "increased" if t.get("change", 0) > 0 else "decreased" if t.get("change", 0) < 0 else "stayed almost unchanged"
            lines.append(f"- **{t['feature']}** {direction} from {t['first_value']} in {t['first_year']} to {t['last_value']} in {t['last_year']} (change {t['change']}).")
        lines.append("")

    lines.append("## 6. Practical conclusion")
    lines.append("The most useful interpretation is to compare this country with its nearest neighbours and then inspect the indicators where it differs most from the dataset average. This helps explain whether the country behaves like a low-development profile, a high-development profile, a transition economy, or a special outlier.")
    return "\n".join(lines)


def generate_country_report_ai(context):
    fallback = {"report_markdown": fallback_country_report(context)}
    system = """
You are a senior development-data analyst writing a rich country intelligence report for a data-mining dashboard.
Return JSON exactly: {"report_markdown":"...markdown..."}

Write a detailed, impressive report, not a short generic summary.
Requirements:
- Use headings and clear paragraphs.
- Use the actual provided values against dataset averages.
- Compare the selected country with its nearest similar countries, explaining what is similar and what differs.
- Use PCA/anomaly results as interpretation, not as magic.
- Use the time-series changes when available.
- Add careful outside/general context where reasonable: conflict, institutions, resource-rich economies, geography, population pressure, post-war recovery, high-income city-state effects, etc.
- Do not invent exact outside statistics or pretend facts are in the data if they are not.
- End with practical insights: what a policymaker, researcher, or NGO would learn from the profile.
""".strip()
    data, err = call_groq_json(system, context, fallback=fallback, max_tokens=3200)
    if isinstance(data, dict) and data.get("report_markdown"):
        report = str(data["report_markdown"]).strip()
        if len(report) > 700:
            return report, err
    return fallback["report_markdown"], err
