"""
pages/5_pca.py — PCA Dimensionality Reduction Analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import io, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import CSS, render_sidebar, init_state, page_header, step_bar, stat_row, fig_layout, geo_layout, PAL, collapse_country_year_panel

st.set_page_config(page_title="PCA · DataMine", page_icon="📉",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)
init_state()
render_sidebar()
step_bar(4)

page_header("📉", "PCA — Principal Component Analysis",
            "Reduce many indicators into a few dimensions and visualise hidden structure.",
            tags=[("Dimensionality Reduction","purple"),("Biplot","blue"),("Loadings","green"),("2D / 3D","amber")],
            accent="linear-gradient(90deg,#2563EB,#0891B2)")

with st.expander("🧠 What is PCA? — Read this if you're new to data mining", expanded=False):
    st.markdown("""
**PCA** (Principal Component Analysis) is a way to take many indicators and compress them into
a small number of "summary dimensions" called **Principal Components (PCs)**.

**Think of it like this:** Instead of describing a country with 15 separate numbers
(GDP, HDI, literacy, internet usage, fertility rate…), PCA finds 2 or 3 "super-scores"
that capture most of what matters — like an overall *development score* (PC1) and a
*demographic structure score* (PC2).

**Why is this useful?**
- You can **plot all countries on a 2D map** even though you have 15+ indicators
- You can see which countries are **truly similar** across all dimensions at once
- You can understand which **indicators drive the differences** between countries

**Key concepts:**
- **Explained Variance:** How much of the original information is kept in each PC.
  PC1 always captures the most, PC2 the second most, etc.
  If PC1+PC2 together explain 80%, your 2D map preserves 80% of all your data's structure.
- **Loadings:** How much each original feature contributes to a PC.
  A high positive loading on GDP and HDI in PC1 means PC1 is essentially a "wealth score."
- **Biplot:** Shows countries AND feature arrows together. Countries in the direction of an
  arrow score high on that feature.
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
    st.markdown('<div class="info-strip">📌 PCA is using one averaged profile per country, so repeated yearly rows do not distort the map. The yearly panel remains available in Time Series.</div>', unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
st.markdown('<div class="g-card"><div class="g-card-title">⚙️ PCA Configuration</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    pca_feats = st.multiselect("Features for PCA", num_cols, default=num_cols,
                                format_func=lambda x: x.replace("_"," ").title())
with c2:
    viz_dims  = st.radio("Projection", ["2D","3D"], horizontal=True)
with c3:
    color_opts = ["None"] + ([label_col] if label_col else [])
    cat_cols   = df.select_dtypes(exclude=[np.number]).columns.tolist()
    color_opts += [c for c in cat_cols if c != label_col]
    color_by   = st.selectbox("Colour points by", color_opts)
st.markdown('</div>', unsafe_allow_html=True)

if len(pca_feats) < 2:
    st.warning("Select at least 2 features."); st.stop()

X_pca  = df[pca_feats].dropna().values
idx_p  = df[pca_feats].dropna().index
base   = df.loc[idx_p].reset_index(drop=True)

n_max  = min(len(pca_feats), len(X_pca))
pca_f  = PCA(n_components=n_max, random_state=42)
pca_f.fit(X_pca)
ev     = pca_f.explained_variance_ratio_
ev_cum = np.cumsum(ev)

n_90 = int(np.argmax(ev_cum >= .90)) + 1
n_95 = int(np.argmax(ev_cum >= .95)) + 1

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "📈 Explained Variance",
    "🔵 Scatter Map",
    "🌊 Loadings & Biplot",
    "🔥 Contribution Heatmap",
    "📋 Download Scores",
])

# ── Tab 1: Variance ───────────────────────────────────────────────────────────
with t1:
    stat_row([
        ("PC1 + PC2",  f"{(ev[0]+ev[1])*100:.1f}%", "variance explained",    "#7C3AED"),
        ("For 90%",    f"{n_90} PCs",                "components needed",     "#2563EB"),
        ("For 95%",    f"{n_95} PCs",                "components needed",     "#059669"),
        ("Total PCs",  f"{n_max}",                   "available",             "#D97706"),
    ])

    c1, c2 = st.columns(2)
    with c1:
        fig_ev = go.Figure()
        fig_ev.add_trace(go.Bar(
            x=[f"PC{i+1}" for i in range(len(ev))],
            y=ev*100, name="Individual",
            marker_color="#7C3AED", opacity=.85))
        fig_ev.add_trace(go.Scatter(
            x=[f"PC{i+1}" for i in range(len(ev))],
            y=ev_cum*100, mode="lines+markers",
            name="Cumulative",
            line=dict(color="#D97706", width=2.5),
            marker=dict(size=7)))
        fig_ev.add_hline(y=90, line_dash="dash", line_color="#059669",
                          annotation_text="90%", annotation_font_color="#059669")
        fig_ev.add_hline(y=95, line_dash="dash", line_color="#2563EB",
                          annotation_text="95%", annotation_font_color="#2563EB")
        fig_ev.update_layout(**fig_layout(400),
                              title="Scree Plot — how much information each PC keeps",
                              xaxis_title="Principal Component", yaxis_title="Variance Explained (%)")
        st.plotly_chart(fig_ev, use_container_width=True)

    with c2:
        var_tbl = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(len(ev))],
            "Variance %": (ev*100).round(2),
            "Cumulative %": (ev_cum*100).round(2),
            "Eigenvalue": pca_f.explained_variance_.round(4),
        })
        st.dataframe(var_tbl, use_container_width=True, hide_index=True)

    st.markdown(f"""<div class="info-strip">
    🧠 <b>Plain English:</b> PC1 captures the biggest pattern in your data
    ({ev[0]*100:.1f}% of all variation). PC2 captures the next biggest ({ev[1]*100:.1f}%).
    Together they explain {(ev[0]+ev[1])*100:.1f}% — so a 2D plot of PC1 vs PC2
    shows you {(ev[0]+ev[1])*100:.1f}% of all the structure in your full dataset.
    To reach 90% you need {n_90} components, meaning your data has about {n_90}
    truly independent patterns driving differences between countries.
    </div>""", unsafe_allow_html=True)

# ── Tab 2: Scatter ────────────────────────────────────────────────────────────
with t2:
    n_vis    = min(3 if viz_dims=="3D" else 2, n_max)
    pca_vis  = PCA(n_components=n_vis, random_state=42)
    scores   = pca_vis.fit_transform(X_pca)
    ev_vis   = pca_vis.explained_variance_ratio_

    plot_df  = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_vis)])
    id_vals  = (base[id_col].astype(str).tolist()
                if id_col and id_col in base.columns
                else [f"row_{i}" for i in range(len(base))])
    plot_df["ID"] = id_vals

    col_arg  = None
    if color_by != "None" and color_by in base.columns:
        plot_df["Color"] = base[color_by].astype(str).values
        col_arg = "Color"

    kw = dict(hover_name="ID",
              labels={f"PC{i+1}": f"PC{i+1} ({ev_vis[i]*100:.1f}%)"
                      for i in range(n_vis)})
    if col_arg:
        kw["color"] = col_arg
        n_unique = plot_df[col_arg].nunique()
        if n_unique <= 20:
            kw["color_discrete_sequence"] = PAL
        else:
            kw["color_continuous_scale"] = "Viridis"

    if viz_dims == "3D" and n_vis == 3:
        fig_p = px.scatter_3d(plot_df, x="PC1", y="PC2", z="PC3", **kw)
        fig_p.update_traces(marker=dict(size=5, line=dict(width=.3,color="#0F1117")))
    else:
        fig_p = px.scatter(plot_df, x="PC1", y="PC2", **kw)
        fig_p.update_traces(marker=dict(size=10, line=dict(width=1,color="#0F1117")))

    fig_p.update_layout(**fig_layout(500 if viz_dims=="3D" else 480),
                         title=f"PCA {viz_dims} — Country Map")
    st.plotly_chart(fig_p, use_container_width=True)

    st.markdown(f"""<div class="info-strip">
    📌 <b>How to read this map:</b> Every dot is a country.
    <b>Countries close together have similar profiles</b> across all your chosen indicators.
    Countries far apart are very different. The axes are not individual indicators — they are
    "summary dimensions" that combine multiple indicators into one score.
    PC1 ({ev_vis[0]*100:.1f}%) is usually the overall development level.
    PC2 ({ev_vis[1]*100:.1f}%) often separates countries by demographic or geographic patterns.
    </div>""", unsafe_allow_html=True)

    # World map coloured by PC1
    if "code" in base.columns:
        base_map = base.copy()
        base_map["PC1_score"] = scores[:, 0]
        # deduplicate by code (take mean if panel)
        map_df = base_map.groupby("code").agg(
            PC1_score=("PC1_score","mean"),
            **({id_col: (id_col,"first")} if id_col and id_col in base_map.columns else {})
        ).reset_index()
        fig_map = px.choropleth(map_df, locations="code", color="PC1_score",
                                hover_name=id_col if id_col and id_col in map_df.columns else "code",
                                color_continuous_scale="RdYlBu",
                                labels={"PC1_score": f"PC1 Score ({ev_vis[0]*100:.1f}%)"})
        fig_map.update_layout(**geo_layout(), title="PC1 Score — World Map")
        st.plotly_chart(fig_map, use_container_width=True)
        st.markdown(f"""<div class="info-strip">
        📌 PC1 score on the map: countries in <b>blue are high on PC1</b>
        (typically high development, high GDP, high internet usage…),
        while countries in <b>red are low on PC1</b>.
        </div>""", unsafe_allow_html=True)

# ── Tab 3: Loadings & Biplot ──────────────────────────────────────────────────
with t3:
    n_load = min(4, n_max)
    loadings = pd.DataFrame(pca_f.components_[:n_load].T,
                             index=pca_feats,
                             columns=[f"PC{i+1}" for i in range(n_load)])

    st.markdown('<div class="sec-header">📊 Loadings — Which features drive each PC?</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="info-strip">
    🧠 <b>Plain English:</b> A <b>loading</b> tells you how much each original indicator
    contributes to a Principal Component. A high positive loading means "countries with high
    PC1 tend to have high values of this feature." Negative means the opposite.
    Features with near-zero loading barely influence that component.
    </div>""", unsafe_allow_html=True)

    cols_load = st.columns(min(n_load, 4))
    for i in range(n_load):
        with cols_load[i]:
            sorted_l = loadings[f"PC{i+1}"].sort_values()
            colors   = ["#DC2626" if v < 0 else "#7C3AED" for v in sorted_l]
            fig_l    = go.Figure(go.Bar(
                x=sorted_l.values,
                y=[f.replace("_"," ").title()[:22] for f in sorted_l.index],
                orientation="h", marker_color=colors
            ))
            exp_pct = pca_f.explained_variance_ratio_[i]*100
            fig_l.update_layout(**fig_layout(max(200, len(pca_feats)*26+50), margin=dict(l=0,r=0,t=38,b=10)),
                                 title=f"PC{i+1} ({exp_pct:.1f}% variance)",
                                 xaxis_title="Loading (−1 to +1)")
            st.plotly_chart(fig_l, use_container_width=True)

    # Biplot
    st.markdown('<div class="sec-header">🎯 Biplot — Countries + Feature Arrows Together</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="info-strip">
    📌 Dots = countries in PC1×PC2 space. <b>Arrows = original features.</b>
    A country in the direction of an arrow scores high on that feature.
    Two arrows pointing in similar directions mean those features are correlated.
    Opposite directions = negatively correlated (e.g. fertility rate vs life expectancy).
    </div>""", unsafe_allow_html=True)

    pca2b  = PCA(n_components=2, random_state=42)
    sc2    = pca2b.fit_transform(X_pca)
    ld2    = pca2b.components_.T  # shape: n_feats × 2
    ev2b   = pca2b.explained_variance_ratio_

    scale  = np.std(sc2, axis=0) * 2.5
    fig_bp = go.Figure()

    # Points (subsample if large)
    N_SHOW = min(400, len(sc2))
    idx_s  = np.random.choice(len(sc2), N_SHOW, replace=False)
    hover  = (base.loc[idx_s, id_col].astype(str).tolist()
              if id_col and id_col in base.columns
              else [f"r{i}" for i in idx_s])
    c_vals = (base.loc[idx_s, color_by].astype(str).tolist()
              if color_by != "None" and color_by in base.columns else None)

    fig_bp.add_trace(go.Scatter(
        x=sc2[idx_s, 0], y=sc2[idx_s, 1], mode="markers",
        marker=dict(size=8, color=c_vals if c_vals else "#4B5563",
                    opacity=.7, line=dict(width=.5, color="#0F1117")),
        hovertext=hover, hoverinfo="text", name="Countries", showlegend=False))

    for j, feat in enumerate(pca_feats):
        vx, vy = ld2[j,0]*scale[0], ld2[j,1]*scale[1]
        fig_bp.add_annotation(x=vx, y=vy, ax=0, ay=0,
                               xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowsize=1.4,
                               arrowwidth=2, arrowcolor="#D97706")
        fig_bp.add_annotation(x=vx*1.13, y=vy*1.13,
                               text=feat.replace("_"," ").title()[:16],
                               showarrow=False,
                               font=dict(size=9, color="#D97706"))

    fig_bp.update_layout(**fig_layout(520),
                          title=f"PCA Biplot — Countries + Feature Vectors",
                          xaxis_title=f"PC1 ({ev2b[0]*100:.1f}%)",
                          yaxis_title=f"PC2 ({ev2b[1]*100:.1f}%)")
    st.plotly_chart(fig_bp, use_container_width=True)

# ── Tab 4: Contribution Heatmap ───────────────────────────────────────────────
with t4:
    st.markdown("""<div class="info-strip">
    📌 This heatmap shows the <b>absolute contribution</b> of each feature to each PC.
    Darker blue = stronger contribution. Features with a strong contribution to PC1 are the
    main drivers of the biggest differences between countries.
    </div>""", unsafe_allow_html=True)

    n_h    = min(6, n_max)
    ld_abs = pd.DataFrame(np.abs(pca_f.components_[:n_h].T),
                           index=pca_feats,
                           columns=[f"PC{i+1}" for i in range(n_h)])
    fig_h  = px.imshow(ld_abs, color_continuous_scale="Blues",
                        text_auto=".2f", aspect="auto",
                        labels={"color":"|Loading|"})
    fig_h.update_traces(textfont=dict(size=9))
    fig_h.update_layout(paper_bgcolor="#12172A",
                         font=dict(family="Manrope", size=10, color="#64748B"),
                         height=max(320, len(pca_feats)*38+70),
                         margin=dict(l=0,r=0,t=36,b=0),
                         title=dict(text="Feature Contribution to Each PC",
                                    font=dict(size=13, color="#E2E8F0")))
    st.plotly_chart(fig_h, use_container_width=True)

    # Top contributors to PC1
    top_pc1 = pd.Series(np.abs(pca_f.components_[0]), index=pca_feats).sort_values(ascending=False)
    fig_fi  = px.bar(x=[f.replace("_"," ").title() for f in top_pc1.index],
                      y=top_pc1.values,
                      color=top_pc1.values, color_continuous_scale="Blues",
                      labels={"x":"Feature","y":"|PC1 Loading|"})
    fig_fi.update_layout(**fig_layout(320), title="Feature Importance in PC1",
                          xaxis_tickangle=-35, showlegend=False)
    st.plotly_chart(fig_fi, use_container_width=True)
    st.markdown(f"""<div class="info-strip">
    📌 The tallest bar is <b>{top_pc1.index[0].replace("_"," ").title()}</b> —
    this is the single most important indicator driving the main axis of variation between
    countries in your dataset. Countries scoring high on PC1 tend to have high
    {top_pc1.index[0].replace("_"," ")} (and also high
    {top_pc1.index[1].replace("_"," ")} and {top_pc1.index[2].replace("_"," ")}).
    </div>""", unsafe_allow_html=True)

# ── Tab 5: Download ───────────────────────────────────────────────────────────
with t5:
    n_dl   = st.slider("Components to include", 2, min(8,n_max), min(4,n_max))
    pca_dl = PCA(n_components=n_dl, random_state=42)
    sc_dl  = pca_dl.fit_transform(X_pca)
    sc_df  = pd.DataFrame(sc_dl, columns=[f"PC{i+1}" for i in range(n_dl)]).round(4)

    if id_col and id_col in base.columns:
        sc_df.insert(0, id_col, base[id_col].values)
    if label_col and label_col in base.columns:
        sc_df.insert(1 if id_col else 0, label_col, base[label_col].values)

    st.dataframe(sc_df, use_container_width=True)
    buf = io.StringIO()
    sc_df.to_csv(buf, index=False)
    st.download_button("⬇️ Download PCA Scores CSV", buf.getvalue(), "pca_scores.csv", "text/csv")
    st.markdown("""<div class="info-strip">
    💡 You can use these PC scores as reduced features in other tools, or import them
    into Excel / R / Python for further analysis.
    </div>""", unsafe_allow_html=True)
