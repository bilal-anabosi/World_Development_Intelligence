"""
pages/7_anomaly_detection.py — Anomaly Detection
Isolation Forest · Local Outlier Factor · Consensus
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import io, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import CSS, render_sidebar, init_state, page_header, step_bar, stat_row, fig_layout, geo_layout, PAL, collapse_country_year_panel, generate_anomaly_ai

st.set_page_config(page_title="Anomaly Detection · DataMine", page_icon="🚨",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)
init_state()
render_sidebar()
step_bar(6)

page_header("🚨", "Anomaly Detection",
            "Automatically flag countries that deviate significantly from all others.",
            tags=[("Isolation Forest","red"),("LOF","amber"),("Consensus","purple"),("Outliers","blue")],
            accent="linear-gradient(90deg,#DC2626,#D97706)")

with st.expander("🧠 What is Anomaly Detection? — Read this if you're new", expanded=False):
    st.markdown("""
**Anomaly Detection** automatically identifies countries (or rows) that are
**significantly different** from everyone else — outliers that don't fit the general pattern.

**Think of it like this:** If you plot 150 countries on a map of economic and social indicators,
most will cluster together in groups. An anomaly detection algorithm finds the countries that sit
far away from any group — the ones that are genuinely unusual.

**Why is this useful?**
- **Discovering exceptional cases** — e.g. a country with very high GDP but very low life
  expectancy (which would be unusual and worth investigating)
- **Finding data quality issues** — an anomaly might be a data entry error
- **Identifying countries on unusual development paths** — either much better or much worse
  than their economic peers

**The two algorithms used here:**

| Algorithm | How it works |
|-----------|-------------|
| **Isolation Forest** | Builds many random decision trees. Easy-to-isolate points (reached quickly by the trees) are anomalies. Works well for high-dimensional data. |
| **Local Outlier Factor (LOF)** | Compares each point's local density to its neighbours. If a country is in a sparse neighbourhood compared to its neighbours, it's an outlier. Good at finding local outliers. |

**Consensus mode:** A country is flagged as a **definite anomaly** only if BOTH algorithms flag it.
This reduces false positives and gives you higher confidence in the results.

**The contamination parameter** is your estimate of what fraction of the data are genuine anomalies.
Start with 10% and adjust based on what you find.
    """)

if st.session_state.raw_df is None:
    st.markdown('<div class="warn-strip">⚠ No dataset. Go to Data Builder or Upload first.</div>',
                unsafe_allow_html=True); st.stop()
if st.session_state.clean_df is None:
    st.markdown('<div class="warn-strip">⚠ Data not preprocessed. Go to Preprocessing first.</div>',
                unsafe_allow_html=True); st.stop()

df        = st.session_state.clean_df.copy()
num_cols  = [c for c in st.session_state.pp_num_cols if c in df.columns]
label_col = st.session_state.pp_label_col
id_col    = st.session_state.pp_id_col

df, num_cols, _collapsed_panel = collapse_country_year_panel(df, num_cols, id_col, label_col)
if _collapsed_panel:
    st.markdown('<div class="info-strip">📌 Anomaly Detection is using one averaged profile per country, so a country is judged by its overall period profile instead of appearing once per year. The original yearly data is still kept for Time Series.</div>', unsafe_allow_html=True)

#Config
st.markdown('<div class="g-card"><div class="g-card-title">⚙️ Configuration</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    an_feats = st.multiselect("Features to analyse", num_cols, default=num_cols,
                               format_func=lambda x: x.replace("_"," ").title())
    algo_an  = st.selectbox("Algorithm",
                             ["Isolation Forest", "Local Outlier Factor (LOF)", "Consensus (both must agree)"],
                             help="Consensus is most reliable — only flags items both algorithms agree on.")
with c2:
    contamination = st.slider("Expected anomaly rate (%)", 2, 30, 10) / 100
    n_est = st.slider("n_estimators (Isolation Forest)", 50, 300, 100, 50,
                       help="More trees = more accurate but slower.")
with c3:
    n_nbrs = st.slider("n_neighbors (LOF)", 5, 50, 20,
                        help="Number of neighbours to compare density against.")
    st.markdown("""<div class="info-strip" style="margin-top:.5rem;">
    💡 <b>Contamination</b> is how many anomalies you expect. 10% means "I think ~10% of
    countries are genuinely unusual." Adjust until results feel right.
    </div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if len(an_feats) < 2:
    st.warning("Select at least 2 features."); st.stop()

Xa      = df[an_feats].dropna().values
idx_a   = df[an_feats].dropna().index
base_a  = df.loc[idx_a].reset_index(drop=True)

# PCA 2D for visualisation
pca_a   = PCA(n_components=2, random_state=42)
X2d_a   = pca_a.fit_transform(Xa)
ev_a    = pca_a.explained_variance_ratio_

# Run detectors 
iso       = IsolationForest(contamination=contamination, n_estimators=n_est, random_state=42)
p_iso     = iso.fit_predict(Xa)
s_iso     = iso.decision_function(Xa)

lof_m     = LocalOutlierFactor(contamination=contamination, n_neighbors=n_nbrs)
p_lof     = lof_m.fit_predict(Xa)
s_lof     = -lof_m.negative_outlier_factor_

if "Isolation Forest" in algo_an:
    final_preds  = p_iso
    final_scores = s_iso
    algo_label   = "Isolation Forest"
elif "LOF" in algo_an:
    final_preds  = p_lof
    final_scores = s_lof
    algo_label   = "LOF"
else:
    votes = (p_iso == -1).astype(int) + (p_lof == -1).astype(int)
    final_preds  = np.where(votes >= 2, -1, 1)
    # Combine normalised scores
    s_iso_n = (s_iso - s_iso.min()) / (s_iso.max()-s_iso.min()+1e-9)
    s_lof_n = (s_lof - s_lof.min()) / (s_lof.max()-s_lof.min()+1e-9)
    final_scores = (s_iso_n + s_lof_n) / 2
    algo_label   = "Consensus (IF + LOF)"

is_anom  = final_preds == -1
n_anom   = int(is_anom.sum())
n_normal = int((~is_anom).sum())

# Unique anomaly names (no duplicates)
def unique_ids(mask):
    if id_col and id_col in base_a.columns:
        raw = base_a.loc[mask, id_col].astype(str).tolist()
        return list(dict.fromkeys(raw))
    return [f"row_{i}" for i, m in enumerate(mask) if m]

anom_ids   = unique_ids(is_anom)
normal_ids = unique_ids(~is_anom)

stat_row([
    ("Total Records",   f"{len(Xa):,}",   "analysed",                    "#4B5563"),
    ("Anomalies Found", str(n_anom),      f"{n_anom/len(Xa)*100:.1f}% of data","#DC2626"),
    ("Normal Records",  str(n_normal),    "fit the pattern",              "#059669"),
    ("Algorithm",       algo_label[:20],  "used",                         "#7C3AED"),
])

if n_anom > 0:
    chips = "".join(f'<span class="chip" style="border-color:#DC2626;color:#FCA5A5;">{n}</span>'
                    for n in anom_ids[:20])
    if len(anom_ids) > 20:
        chips += f'<span class="chip">+{len(anom_ids)-20} more</span>'
    st.markdown(f"""
    <div style="background:#1A0A0A;border:1px solid #3D1515;border-left:4px solid #DC2626;
                border-radius:0 12px 12px 0;padding:.9rem 1.2rem;margin:.5rem 0;">
        <div style="font-size:.82rem;font-weight:700;color:#FCA5A5;margin-bottom:.5rem;">
            🚨 {n_anom} anomalous items detected by {algo_label}
        </div>
        <div class="chip-grid">{chips}</div>
    </div>""", unsafe_allow_html=True)

# Tabs 
t1, t2, t3, t_ai, t4, t5 = st.tabs([
    "🔵 PCA Map",
    "🗺️ World Map",
    "📋 Anomaly Records",
    "🤖 AI Explanations",
    "📊 Feature Analysis",
    "🆚 Algorithm Comparison",
])

with t1:
    id_vals  = unique_ids(np.ones(len(base_a), dtype=bool))
    # deduplicated per-row ids (allow repeats in panel data but show correct name)
    id_list  = (base_a[id_col].astype(str).tolist()
                if id_col and id_col in base_a.columns
                else [f"row_{i}" for i in range(len(base_a))])
    grp_list = (base_a[label_col].astype(str).tolist()
                if label_col and label_col in base_a.columns
                else ["—"]*len(base_a))

    plot_a = pd.DataFrame({
        "PC1":    X2d_a[:,0],
        "PC2":    X2d_a[:,1],
        "Status": np.where(is_anom,"Anomaly","Normal"),
        "Score":  final_scores,
        "ID":     id_list,
        "Group":  grp_list,
    })
    fig_an = px.scatter(plot_a, x="PC1", y="PC2", color="Status",
                         hover_name="ID",
                         hover_data={"Group":True,"Score":":.4f",
                                     "PC1":":.3f","PC2":":.3f"},
                         color_discrete_map={"Anomaly":"#DC2626","Normal":"#7C3AED"},
                         symbol="Status",
                         symbol_map={"Anomaly":"x","Normal":"circle"},
                         labels={"PC1":f"PC1 ({ev_a[0]*100:.1f}%)",
                                 "PC2":f"PC2 ({ev_a[1]*100:.1f}%)"})
    fig_an.update_traces(marker=dict(size=10, line=dict(width=1,color="#0F1117")))

    # Annotate top anomalies
    anom_rows = plot_a[plot_a["Status"]=="Anomaly"].head(15)
    for _, row in anom_rows.iterrows():
        fig_an.add_annotation(
            x=row["PC1"], y=row["PC2"],
            text=str(row["ID"])[:12], showarrow=True,
            arrowhead=2, arrowsize=.8, arrowcolor="#DC2626",
            font=dict(size=8, color="#DC2626"),
            bgcolor="rgba(15,17,23,0.85)",
            bordercolor="#DC2626", borderwidth=1, borderpad=2)

    fig_an.update_layout(**fig_layout(500), title=f"Anomaly Detection — {algo_label}")
    st.plotly_chart(fig_an, use_container_width=True)
    st.markdown(f"""<div class="info-strip">
    📌 <b>Red ✕ markers = anomalies</b> — countries that sit far from all clusters.
    Blue circles = normal countries. <b>Anomalies are NOT necessarily bad</b> — they are
    simply unusual. A country could be an anomaly because it's exceptionally well-developed
    (like Singapore or Norway) or because it faces extreme challenges (like Somalia or Yemen).
    Always investigate <i>why</i> a country is flagged.
    </div>""", unsafe_allow_html=True)

with t2:
    if "code" in base_a.columns:
        map_df = base_a.copy()
        map_df["Status"] = np.where(is_anom,"Anomaly","Normal")
        # Deduplicate: if panel, take "Anomaly" if any row for that country is anomaly
        if id_col and id_col in map_df.columns:
            status_by_code = (map_df.groupby("code")["Status"]
                              .apply(lambda x: "Anomaly" if "Anomaly" in x.values else "Normal")
                              .reset_index())
            hover_col = id_col if id_col in map_df.columns else "code"
            first_info = map_df.groupby("code")[[hover_col]].first().reset_index()
            map_df_u = status_by_code.merge(first_info, on="code", how="left")
        else:
            map_df_u = map_df[["code","Status"]].drop_duplicates(subset="code")
            hover_col = "code"

        fig_map = px.choropleth(map_df_u, locations="code", color="Status",
                                 hover_name=hover_col,
                                 color_discrete_map={"Anomaly":"#DC2626","Normal":"#1E2D4E"})
        fig_map.update_layout(**geo_layout(), title=f"Anomalies — World Map ({algo_label})")
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown(f"""<div class="info-strip">
        📌 <b>Red countries</b> are anomalies — they have an unusual combination of indicators
        compared to their peers. These countries are worth studying individually to understand
        what makes them structurally different.
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Add a 'code' column (ISO 3-letter country code) to enable the world map.")

with t3:
    anom_df = base_a[is_anom].copy()
    anom_df["Anomaly Score"] = final_scores[is_anom].round(4)
    anom_df = anom_df.sort_values("Anomaly Score")

    display_cols = ([id_col]    if id_col    and id_col    in anom_df.columns else []) + \
                   ([label_col] if label_col and label_col in anom_df.columns else []) + \
                   an_feats[:8] + ["Anomaly Score"]
    display_cols = list(dict.fromkeys(c for c in display_cols if c in anom_df.columns))

    st.markdown(f"**{n_anom} anomalous records:**")
    st.dataframe(anom_df[display_cols].round(3), use_container_width=True)
    buf = io.StringIO(); anom_df[display_cols].to_csv(buf, index=False)
    st.download_button("⬇️ Download Anomaly Records", buf.getvalue(),
                        "anomalies.csv", "text/csv")
    st.markdown("""<div class="info-strip">
    💡 Download this table to investigate each anomalous country in detail.
    The "Anomaly Score" is lower for more extreme outliers (Isolation Forest convention).
    </div>""", unsafe_allow_html=True)

with t_ai:
    st.markdown('<div class="sec-header">🤖 AI Explanation for Anomalous Countries</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-strip">The AI explains only the countries flagged as anomalies. It uses the selected indicators, anomaly score, and closest normal countries as context.</div>', unsafe_allow_html=True)

    if n_anom == 0:
        st.info("No anomalies were detected with the current settings.")
    else:
        ai_sig = f"anom-{len(base_a)}-{','.join(an_feats)}-{algo_an}-{contamination}-{hash(tuple(map(bool, is_anom)))}"
        if "anomaly_ai_sig" not in st.session_state:
            st.session_state.anomaly_ai_sig = None
        if "anomaly_ai_cards" not in st.session_state:
            st.session_state.anomaly_ai_cards = None

        run_ai = st.button("🤖 Generate AI anomaly explanations", key="run_anomaly_ai")
        if run_ai:
            with st.spinner("Generating anomaly explanations after the anomaly results are ready..."):
                try:
                    ai_anoms, ai_err = generate_anomaly_ai(base_a, list(an_feats), id_col, np.array(is_anom, dtype=bool), np.array(final_scores, dtype=float), np.array(Xa, dtype=float))
                    st.session_state.anomaly_ai_sig = ai_sig
                    st.session_state.anomaly_ai_cards = ai_anoms
                    if ai_err:
                        st.caption("AI note: Groq could not generate a clean response, so the app used smart local fallback explanations.")
                    else:
                        st.success("AI anomaly explanations generated.")
                except Exception as e:
                    st.session_state.anomaly_ai_sig = ai_sig
                    st.session_state.anomaly_ai_cards = None
                    st.caption(f"AI note: generation failed safely ({type(e).__name__}). Showing basic anomaly cards.")

        ai_anoms = st.session_state.anomaly_ai_cards if st.session_state.anomaly_ai_sig == ai_sig else None
        if ai_anoms is None:
            st.info("Click the AI button after anomalies are detected to generate country-specific explanations. Until then, basic anomaly cards are shown.")
            ai_anoms = {}

        # Display cards in the same order as the anomaly table.
        shown = 0
        for _, row in anom_df.head(8).iterrows():
            country = str(row[id_col]) if id_col and id_col in anom_df.columns else str(row.name)
            item = ai_anoms.get(country, {}) if isinstance(ai_anoms, dict) else {}
            why = item.get("why", "This country has an unusual combination of selected indicators compared with the rest of the dataset.")
            similar = item.get("similar_to", "See nearest normal countries in the generated context.")
            fact = item.get("fact_context", "Validate with the feature analysis before making conclusions.")
            st.markdown(f"""
            <div style="background:#12172A;border:1px solid #3D1515;border-left:4px solid #DC2626;border-radius:0 14px 14px 0;padding:1rem 1.15rem;margin:.75rem 0;">
                <div style="font-weight:850;color:#FCA5A5;font-size:1rem;">🚨 {country}</div>
                <div style="font-size:.78rem;color:#94A3B8;margin:.15rem 0 .55rem 0;">Anomaly score: {row.get('Anomaly Score', 0):.4f}</div>
                <div style="font-size:.85rem;color:#E2E8F0;line-height:1.55;"><b>Why unusual:</b> {why}</div>
                <div style="font-size:.82rem;color:#CBD5E1;line-height:1.55;margin-top:.35rem;"><b>Closest normal comparison:</b> {similar}</div>
                <div style="font-size:.8rem;color:#93C5FD;line-height:1.55;margin-top:.35rem;"><b>Context:</b> {fact}</div>
            </div>""", unsafe_allow_html=True)
            shown += 1
        if n_anom > shown:
            st.caption(f"Showing AI explanations for the first {shown} anomalies to keep the page fast.")

with t4:
    st.markdown('<div class="sec-header">📊 How Do Anomalies Differ From Normal Countries?</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="info-strip">
    These box plots show the distribution of each feature for <b>normal countries (purple)</b>
    vs <b>anomalies (red)</b>. If the boxes barely overlap, that feature is a strong reason
    why the anomalies are unusual.
    </div>""", unsafe_allow_html=True)

    n_show = min(len(an_feats), 8)
    cols_f = st.columns(2)
    for i, feat in enumerate(an_feats[:n_show]):
        with cols_f[i % 2]:
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=base_a.loc[~is_anom, feat].dropna(), name="Normal",
                marker_color="#7C3AED", line_color="#7C3AED", fillcolor="rgba(124,58,237,0.2)"))
            fig_box.add_trace(go.Box(
                y=base_a.loc[is_anom,  feat].dropna(), name="Anomaly",
                marker_color="#DC2626", line_color="#DC2626", fillcolor="rgba(220,38,38,0.2)"))
            fig_box.update_layout(**fig_layout(280),
                                   title=feat.replace("_"," ").title(),
                                   showlegend=(i == 0))
            st.plotly_chart(fig_box, use_container_width=True)

    # Parallel coordinates plot
    if n_anom > 0 and len(an_feats) >= 3:
        st.markdown('<div class="sec-header">🔀 Parallel Coordinates — Full Feature View</div>',
                    unsafe_allow_html=True)
        st.markdown("""<div class="info-strip">
        Each line is a country. <b>Red lines = anomalies</b>, blue = normal.
        Follow a red line across all features to see its complete profile.
        A line that zigzags a lot relative to blue lines confirms the anomaly.
        </div>""", unsafe_allow_html=True)
        para_df = base_a[an_feats[:8]].copy()
        para_df["_anom"] = is_anom.astype(int)
        dims = [dict(range=[para_df[c].min(), para_df[c].max()],
                     label=c.replace("_"," ").title()[:18], values=para_df[c])
                for c in an_feats[:8]]
        fig_par = go.Figure(go.Parcoords(
            line=dict(color=para_df["_anom"],
                      colorscale=[[0,"#7C3AED"],[1,"#DC2626"]],
                      showscale=False),
            dimensions=dims))
        fig_par.update_layout(paper_bgcolor="#12172A", plot_bgcolor="#0F1117",
                               font=dict(family="Manrope", size=10, color="#64748B"),
                               height=400, margin=dict(l=60,r=20,t=40,b=20))
        st.plotly_chart(fig_par, use_container_width=True)

with t5:
    st.markdown('<div class="sec-header">🆚 Isolation Forest vs LOF Agreement</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="info-strip">
    💡 The two algorithms often disagree on edge cases. Items both flag are the most reliable
    anomalies. Items only one flags may depend on the algorithm's assumptions.
    </div>""", unsafe_allow_html=True)

    votes2 = (p_iso==-1).astype(int) + (p_lof==-1).astype(int)
    vote_df = base_a.copy()
    vote_df["IF"]   = (p_iso==-1)
    vote_df["LOF"]  = (p_lof==-1)
    vote_df["Votes"] = votes2
    vote_df["Status"] = vote_df["Votes"].map(
        {0:"Normal (neither)", 1:"One algorithm only", 2:"Both agree — strong anomaly"})

    if id_col and id_col in vote_df.columns:
        vote_df["ID"] = vote_df[id_col].astype(str)
    else:
        vote_df["ID"] = [f"row_{i}" for i in range(len(vote_df))]

    # Deduplicate by ID
    vote_summary = vote_df.groupby("ID").agg(
        IF=("IF","max"), LOF=("LOF","max"), Votes=("Votes","max"),
        Status=("Status","first")).reset_index()
    vote_summary = vote_summary[vote_summary["Votes"]>0].sort_values("Votes",ascending=False)

    if len(vote_summary):
        st.dataframe(vote_summary, use_container_width=True, hide_index=True)

    
    vcounts = vote_df.drop_duplicates("ID")["Status"].value_counts().reset_index()
    vcounts.columns = ["Status","Count"]
    color_map = {"Normal (neither)":"#1E2D4E",
                 "One algorithm only":"#D97706",
                 "Both agree — strong anomaly":"#DC2626"}
    fig_v = px.bar(vcounts, x="Status", y="Count",
                    color="Status", color_discrete_map=color_map, text="Count")
    fig_v.update_traces(textposition="outside", textfont=dict(color="#E2E8F0"))
    fig_v.update_layout(**fig_layout(340), showlegend=False,
                         title="Anomaly Agreement Between Algorithms",
                         xaxis_title="Category", yaxis_title="Count")
    st.plotly_chart(fig_v, use_container_width=True)

    strong_anom = vote_summary[vote_summary["Votes"]==2]["ID"].tolist()
    if strong_anom:
        st.markdown(f"""<div class="info-strip" style="border-color:#DC2626;background:#1A0A0A;">
        🚨 <b>{len(strong_anom)} items flagged by BOTH algorithms</b>
        (highest confidence anomalies):<br>
        {", ".join(f"<b>{n}</b>" for n in strong_anom[:20])}
        {'…' if len(strong_anom)>20 else ''}
        </div>""", unsafe_allow_html=True)
