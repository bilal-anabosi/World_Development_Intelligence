"""
pages/9_country_report.py — AI Country Intelligence Report
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import io, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    CSS, render_sidebar, init_state, page_header, step_bar, stat_row,
    fig_layout, PAL, collapse_country_year_panel, generate_country_report_ai
)

st.set_page_config(page_title="AI Country Report · DataMine", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)
init_state()
render_sidebar()
step_bar(7)

page_header("🤖", "AI Country Intelligence Report",
            "Generate one readable country report using the dataset, similarity, PCA, anomaly detection, and time trends.",
            tags=[("AI Report", "purple"), ("Similarity", "blue"), ("Anomaly", "red"), ("Download", "green")],
            accent="linear-gradient(90deg,#7C3AED,#0EA5E9)")

if st.session_state.raw_df is None:
    st.markdown('<div class="warn-strip">⚠ No dataset loaded. Go to Data Builder or Upload first.</div>', unsafe_allow_html=True)
    st.stop()
if st.session_state.clean_df is None:
    st.markdown('<div class="warn-strip">⚠ Data not preprocessed yet. Go to Preprocessing first.</div>', unsafe_allow_html=True)
    st.stop()

clean_df = st.session_state.clean_df.copy()
original_df = st.session_state.raw_df.copy() if st.session_state.raw_df is not None else clean_df.copy()
num_cols = [c for c in st.session_state.pp_num_cols if c in clean_df.columns]
label_col = st.session_state.pp_label_col
id_col = st.session_state.pp_id_col or ("country" if "country" in clean_df.columns else None)

if not id_col or id_col not in clean_df.columns:
    st.warning("A country/name ID column is needed to generate country reports.")
    st.stop()

analysis_df, analysis_num_cols, collapsed = collapse_country_year_panel(clean_df, num_cols, id_col, label_col)
if collapsed:
    st.markdown('<div class="info-strip">📌 This report uses one averaged profile per country for similarity, PCA, and anomaly detection. Time-series notes still use the original yearly rows.</div>', unsafe_allow_html=True)

if len(analysis_num_cols) < 2:
    st.warning("At least two numeric indicators are needed for the report.")
    st.stop()

countries = sorted(analysis_df[id_col].astype(str).dropna().unique().tolist())
c1, c2 = st.columns([1, 2])
with c1:
    country = st.selectbox("Choose a country", countries)
with c2:
    features = st.multiselect("Indicators to use in the report", analysis_num_cols, default=analysis_num_cols[:min(10, len(analysis_num_cols))], format_func=lambda x: x.replace("_", " ").title())

if len(features) < 2:
    st.warning("Select at least 2 indicators.")
    st.stop()

work = analysis_df.dropna(subset=features).reset_index(drop=True)
if country not in work[id_col].astype(str).values:
    st.warning("This country has missing values for the selected indicators. Try fewer indicators.")
    st.stop()

X = work[features].values
country_idx = work.index[work[id_col].astype(str) == country][0]
row = work.iloc[country_idx]

# Build a raw/original-value table for the report text.
# Similarity/PCA/anomaly should keep using the preprocessed values above,
# but the human-readable report should not show scaled z-score values when
# the original indicator values are available.
raw_features = [c for c in features if c in original_df.columns and pd.api.types.is_numeric_dtype(original_df[c])]
raw_value_df = None
if id_col in original_df.columns and raw_features:
    raw_value_df, _, _ = collapse_country_year_panel(original_df.copy(), raw_features, id_col, label_col)
    raw_value_df = raw_value_df.dropna(subset=raw_features).reset_index(drop=True)

value_work = work.copy()
value_features = features
if raw_value_df is not None and country in raw_value_df[id_col].astype(str).values:
    # Keep only countries that also exist in the analysis table so comparisons align.
    keep_countries = set(work[id_col].astype(str))
    value_work = raw_value_df[raw_value_df[id_col].astype(str).isin(keep_countries)].reset_index(drop=True)
    value_features = raw_features

value_row = value_work[value_work[id_col].astype(str) == country].iloc[0] if country in value_work[id_col].astype(str).values else row

# Similarity
similar = []
for i in range(len(work)):
    if i == country_idx:
        continue
    d = float(np.linalg.norm(X[i] - X[country_idx]))
    similar.append((i, d))
similar = sorted(similar, key=lambda x: x[1])[:5]

# Detailed comparison with nearest countries. This gives the LLM real evidence
# instead of only country names and distances.
similar_country_comparison = []
for i, d in similar:
    other_country = str(work.iloc[i][id_col])
    diffs = []
    other_value_rows = value_work[value_work[id_col].astype(str) == other_country]
    if len(other_value_rows) > 0:
        other_value_row = other_value_rows.iloc[0]
        for f in value_features:
            if f in value_row.index and f in other_value_row.index:
                a = float(value_row[f])
                b = float(other_value_row[f])
                diffs.append((f, abs(a - b), a, b))
    if not diffs:
        for f in features:
            a = float(work.iloc[country_idx][f])
            b = float(work.iloc[i][f])
            diffs.append((f, abs(a - b), a, b))
    diffs_sorted = sorted(diffs, key=lambda t: t[1])
    closest_matches = [
        f"{f.replace('_',' ').title()}: {a:.3f} vs {b:.3f}"
        for f, _, a, b in diffs_sorted[:5]
    ]
    biggest_gaps = [
        f"{f.replace('_',' ').title()}: {a:.3f} vs {b:.3f}"
        for f, _, a, b in sorted(diffs, key=lambda t: t[1], reverse=True)[:4]
    ]
    similar_country_comparison.append({
        "country": other_country,
        "distance": round(float(d), 3),
        "closest_matching_indicators": closest_matches,
        "biggest_differences": biggest_gaps,
    })

# PCA
pca = PCA(n_components=2, random_state=42)
X2d = pca.fit_transform(X)
pca_position = {"pc1": round(float(X2d[country_idx, 0]), 4), "pc2": round(float(X2d[country_idx, 1]), 4), "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_]}

# Anomaly
iso = IsolationForest(contamination=min(0.15, max(0.05, 3 / max(len(work), 1))), random_state=42)
preds = iso.fit_predict(X)
scores = iso.decision_function(X)
is_anomaly = bool(preds[country_idx] == -1)

# Above/below average for the report, using raw/original values when available.
means = value_work[value_features].mean(numeric_only=True)
stds = value_work[value_features].std(numeric_only=True).replace(0, 1)
z = ((value_row[value_features] - means) / stds).sort_values(key=lambda x: x.abs(), ascending=False)
highs, lows = [], []
for f in z.index:
    item = {"feature": f.replace("_", " ").title(), "value": round(float(value_row[f]), 3), "dataset_average": round(float(means[f]), 3)}
    if z[f] > 0:
        highs.append(item)
    else:
        lows.append(item)

# Time trends from original yearly data
trends = []
if "year" in original_df.columns and id_col in original_df.columns:
    crows = original_df[original_df[id_col].astype(str) == country].copy()
    crows["year"] = pd.to_numeric(crows["year"], errors="coerce")
    crows = crows.dropna(subset=["year"]).sort_values("year")
    if len(crows) >= 2:
        for f in features[:6]:
            if f in crows.columns and pd.api.types.is_numeric_dtype(crows[f]):
                valid = crows.dropna(subset=[f])
                if len(valid) >= 2:
                    first, last = valid.iloc[0], valid.iloc[-1]
                    trends.append({
                        "feature": f.replace("_", " ").title(),
                        "first_year": int(first["year"]),
                        "last_year": int(last["year"]),
                        "first_value": round(float(first[f]), 3),
                        "last_value": round(float(last[f]), 3),
                        "change": round(float(last[f] - first[f]), 3),
                    })

context = {
    "country": country,
    "dataset_name": st.session_state.dataset_name or "Current dataset",
    "profile_period": str(row.get("period", "latest/available profile")),
    "highest_vs_average": highs[:6],
    "lowest_vs_average": lows[:6],
    "similar_countries": [{"country": str(work.iloc[i][id_col]), "distance": round(d, 3)} for i, d in similar],
    "similar_country_comparison": similar_country_comparison,
    "selected_country_values": {f.replace("_", " ").title(): round(float(value_row[f]), 3) for f in value_features if f in value_row.index},
    "value_note": "Human-readable values use original raw indicators when available; similarity/PCA/anomaly calculations still use the preprocessed analysis matrix.",
    "pca_position": pca_position,
    "anomaly": {"is_anomaly": is_anomaly, "score": round(float(scores[country_idx]), 4), "reason": "Isolation Forest was applied to the same selected indicators."},
    "time_trends": trends[:6],
}

stat_row([
    ("Country", country[:18], "selected", "#7C3AED"),
    ("Similar Countries", str(len(similar)), "nearest profiles", "#2563EB"),
    ("Anomaly Status", "Anomaly" if is_anomaly else "Normal", "Isolation Forest", "#DC2626" if is_anomaly else "#059669"),
    ("Indicators", str(len(features)), "used", "#D97706"),
])

left, right = st.columns([1.05, 1])
with left:
    st.markdown('<div class="sec-header">📌 Computed Evidence</div>', unsafe_allow_html=True)
    st.markdown("**Most above-average indicators**")
    st.dataframe(pd.DataFrame(highs[:6]), use_container_width=True, hide_index=True)
    st.markdown("**Most below-average indicators**")
    st.dataframe(pd.DataFrame(lows[:6]), use_container_width=True, hide_index=True)
    st.markdown("**Closest countries by selected indicators**")
    st.dataframe(pd.DataFrame(context["similar_countries"]), use_container_width=True, hide_index=True)
    with st.expander("Show detailed nearest-country comparison evidence"):
        st.dataframe(pd.DataFrame(similar_country_comparison), use_container_width=True, hide_index=True)

with right:
    plot_df = pd.DataFrame({"PC1": X2d[:,0], "PC2": X2d[:,1], "Country": work[id_col].astype(str), "Selected": ["Selected" if i == country_idx else "Other" for i in range(len(work))]})
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="Selected", hover_name="Country", color_discrete_sequence=[PAL[0], PAL[4]])
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="#0F1117")))
    fig.update_layout(**fig_layout(410), title="Country Position on PCA Map")
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="sec-header">🤖 Generated AI Report</div>', unsafe_allow_html=True)

report_sig = f"report-{country}-{','.join(features)}-{len(work)}"
if "country_report_sig" not in st.session_state:
    st.session_state.country_report_sig = None
if "country_report_text" not in st.session_state:
    st.session_state.country_report_text = None

run_report = st.button("🤖 Generate detailed AI country report", key="run_country_report_ai")
if run_report:
    with st.spinner("Generating the detailed country report after the evidence is ready..."):
        try:
            report, err = generate_country_report_ai(context)
            st.session_state.country_report_sig = report_sig
            st.session_state.country_report_text = report
            if err:
                st.caption("AI note: Groq could not generate a clean report, so the app used the detailed local fallback report.")
            else:
                st.success("AI country report generated.")
        except Exception as e:
            st.session_state.country_report_sig = report_sig
            st.session_state.country_report_text = None
            st.caption(f"AI note: report generation failed safely ({type(e).__name__}).")

report = st.session_state.country_report_text if st.session_state.country_report_sig == report_sig else None
if report is None:
    st.info("Click the AI button after reviewing the computed evidence. The report will use raw/original indicator values when available, plus the computed similarity, PCA, anomaly, and time-series evidence.")
else:
    st.markdown(report)
    st.download_button("⬇️ Download AI Country Report", report.encode("utf-8"), file_name=f"{country.replace(' ', '_')}_country_report.md", mime="text/markdown")
