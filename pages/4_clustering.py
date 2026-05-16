"""
pages/4_clustering.py — Clustering Analysis
K-Means · DBSCAN · Hierarchical
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage as sc_linkage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io, sys, os
import re
from html import escape, unescape
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import CSS, render_sidebar, init_state, page_header, step_bar, stat_row, fig_layout, geo_layout, PAL, collapse_country_year_panel, generate_kmeans_cluster_ai

st.set_page_config(page_title="Clustering · DataMine", page_icon="🔵",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)
init_state()
render_sidebar()
step_bar(3)

page_header("🔵", "Clustering Analysis",
            "Group similar countries (or rows) together automatically — no labels needed.",
            tags=[("K-Means","purple"),("DBSCAN","blue"),("Hierarchical","green"),("Comparison","amber")],
            accent="linear-gradient(90deg,#7C3AED,#2563EB)")

# ── What is clustering? ───────────────────────────────────────────────────────
with st.expander("🧠 What is Clustering? — Read this if you're new to data mining", expanded=False):
    st.markdown("""
**Clustering** is an algorithm that automatically groups your data into clusters of similar items —
without you telling it what the groups should be. It discovers natural structure on its own.

**Think of it like this:** Imagine 150 countries on a map of economic and social indicators.
Clustering automatically draws circles around countries that look similar to each other — not
geographically, but in terms of their GDP, life expectancy, education levels, and so on.

**What makes two countries "similar" here?**
Countries are similar if they share patterns across the indicators you selected in Preprocessing —
things like economic development, social wellbeing, internet access, fertility rates, etc.
The algorithm finds groups that are similar economically, socially, culturally, and in terms of
human development — all at the same time.

**The three algorithms available:**
- 🔵 **K-Means** — You choose how many groups (K). The algorithm finds the K tightest groups.
  Best for roughly equal-sized, round-ish clusters. *Most commonly used.*
- 🟠 **DBSCAN** — You don't choose K. It finds dense regions and labels sparse points as "noise".
  Good for irregular-shaped clusters. Can find outlier countries automatically.
- 🟢 **Hierarchical** — Builds a family tree (dendrogram) of all countries merging step by step.
  Great for understanding how groups relate to each other.

**How to read the results:**
- *PCA Scatter*: A 2D map of all countries. Closer = more similar. Same colour = same cluster.
- *Radar Chart*: Shows what makes each cluster distinctive across all features.
- *World Map*: See which regions of the world fall into the same clusters.
- *Cluster Cards*: The countries in each group and their average feature values.
- *Silhouette Score*: How well-defined the clusters are (closer to 1 = better).
    """)

if st.session_state.raw_df is None:
    st.markdown('<div class="warn-strip">⚠ No dataset loaded. Go to Data Builder or Upload first.</div>',
                unsafe_allow_html=True); st.stop()
if st.session_state.clean_df is None:
    st.markdown('<div class="warn-strip">⚠ Data not preprocessed yet. Go to Preprocessing first.</div>',
                unsafe_allow_html=True); st.stop()

df        = st.session_state.clean_df.copy()
num_cols  = [c for c in st.session_state.pp_num_cols if c in df.columns]
label_col = st.session_state.pp_label_col
id_col    = st.session_state.pp_id_col

df, num_cols, _collapsed_panel = collapse_country_year_panel(df, num_cols, id_col, label_col)
if _collapsed_panel:
    st.markdown('<div class="info-strip">📌 This page is using one row per country by averaging each selected indicator across the available years. The original yearly data is still kept for the Time Series page.</div>', unsafe_allow_html=True)

# ── Feature selector ──────────────────────────────────────────────────────────
st.markdown('<div class="g-card"><div class="g-card-title">⚙️ Feature Selection</div>', unsafe_allow_html=True)
st.markdown("""<div class="info-strip">
Select which indicators to use for grouping. More features = richer comparison,
but start with 4–8 key ones for clearest results.
</div>""", unsafe_allow_html=True)
sel_feats = st.multiselect("Features to cluster on", num_cols, default=num_cols,
                            format_func=lambda x: x.replace("_"," ").title())
st.markdown('</div>', unsafe_allow_html=True)

if len(sel_feats) < 2:
    st.warning("Select at least 2 features."); st.stop()

X         = df[sel_feats].dropna().values
kept_idx  = df[sel_feats].dropna().index
df_sub    = df.loc[kept_idx].reset_index(drop=True)

# PCA 2D for all scatter plots
pca2      = PCA(n_components=2, random_state=42)
X2d       = pca2.fit_transform(X)
ev        = pca2.explained_variance_ratio_

# ── Shared helpers ────────────────────────────────────────────────────────────
def get_ids():
    """Return a deduplicated list of ID strings aligned with df_sub rows."""
    if id_col and id_col in df_sub.columns:
        raw = df_sub[id_col].astype(str).tolist()
    else:
        raw = [f"row_{i}" for i in range(len(df_sub))]
    # Deduplicate: if panel data, each country appears N times.
    # Keep only one label per unique value so chips don't repeat.
    seen  = {}
    dedup = []
    for i, v in enumerate(raw):
        if v not in seen:
            seen[v] = i
        dedup.append(v)
    return dedup

def get_unique_members(mask, raw_ids):
    """Unique member IDs for a cluster mask (no duplicates)."""
    return list(dict.fromkeys(r for r, m in zip(raw_ids, mask) if m))

def pca_scatter(labels, title):
    lbl_str = ["Noise" if l == -1 else f"Cluster {l}" for l in labels]
    raw_ids = get_ids()
    hover_id = [raw_ids[i] for i in range(len(X2d))]
    grp_vals = (df_sub[label_col].astype(str).tolist()
                if label_col and label_col in df_sub.columns
                else ["—"]*len(X2d))
    plot_df = pd.DataFrame({
        "PC1": X2d[:,0], "PC2": X2d[:,1],
        "Cluster": lbl_str, "ID": hover_id, "Group": grp_vals,
    })
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster",
                     hover_name="ID",
                     hover_data={"Group": True, "Cluster": True,
                                 "PC1":":.3f","PC2":":.3f"},
                     color_discrete_sequence=PAL,
                     labels={"PC1":f"PC1 ({ev[0]*100:.1f}%)",
                             "PC2":f"PC2 ({ev[1]*100:.1f}%)"})
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="#0F1117")))
    fig.update_layout(**fig_layout(450), title=title)
    return fig

def world_map(labels, title):
    if "code" not in df_sub.columns:
        return None
    tmp = df_sub.copy()
    tmp["Cluster"] = ["Noise" if l==-1 else f"Cluster {l}" for l in labels]
    hover_name = id_col if id_col and id_col in tmp.columns else "code"
    fig = px.choropleth(tmp, locations="code", color="Cluster",
                        hover_name=hover_name,
                        color_discrete_sequence=PAL)
    fig.update_layout(**geo_layout(), title=title)
    return fig

def radar_chart(labels, title):
    tmp = df_sub.copy()
    tmp["_cl"] = labels
    profile = tmp.groupby("_cl")[sel_feats].mean()
    # Normalise to 0-1 for fair visual comparison
    pn = profile.copy()
    for c in sel_feats:
        mn, mx = pn[c].min(), pn[c].max()
        if mx > mn: pn[c] = (pn[c]-mn)/(mx-mn)
    theta = [f.replace("_"," ").title() for f in sel_feats]
    fig = go.Figure()
    for i, cl in enumerate(pn.index):
        if cl == -1: continue
        v = pn.loc[cl, sel_feats].tolist()
        v_closed = v + [v[0]]
        t_closed = theta + [theta[0]]
        r, g, b = (int(PAL[i%len(PAL)][1:3],16),
                   int(PAL[i%len(PAL)][3:5],16),
                   int(PAL[i%len(PAL)][5:7],16))
        fig.add_trace(go.Scatterpolar(
            r=v_closed, theta=t_closed, fill="toself",
            name=f"Cluster {cl}",
            line=dict(color=PAL[i%len(PAL)], width=2),
            fillcolor=f"rgba({r},{g},{b},0.12)"))
    fig.update_layout(
        polar=dict(bgcolor="#0F1117",
                   radialaxis=dict(visible=True, gridcolor="#1E2333", color="#4B5563"),
                   angularaxis=dict(gridcolor="#1E2333", color="#64748B")),
        paper_bgcolor="#12172A", height=430,
        legend=dict(bgcolor="#12172A", bordercolor="#1E2333", borderwidth=1,
                    font=dict(color="#94A3B8")),
        title=dict(text=title, font=dict(size=13, color="#E2E8F0")))
    return fig

def feature_heatmap(labels, title):
    tmp = df_sub.copy()
    tmp["_cl"] = labels
    profile = tmp[tmp["_cl"]!=-1].groupby("_cl")[sel_feats].mean().round(3)
    profile.index = [f"Cluster {i}" for i in profile.index]
    fig = px.imshow(profile, text_auto=".2f", color_continuous_scale="RdYlBu",
                    aspect="auto")
    fig.update_traces(textfont=dict(size=9))
    fig.update_layout(paper_bgcolor="#12172A", plot_bgcolor="#0F1117",
                       font=dict(family="Manrope", size=10, color="#64748B"),
                       height=max(180, len(profile)*45+70),
                       margin=dict(l=0,r=0,t=36,b=0),
                       title=dict(text=title, font=dict(size=13, color="#E2E8F0")))
    return fig


def clean_ai_card_text(text, max_chars=550):
    """
    Very defensive cleaner for AI text.
    If the model/session state contains HTML card code, remove it completely
    instead of showing <div style=...> inside the card.
    """
    if text is None:
        return ""

    # Some Groq/fallback outputs may accidentally store nested objects.
    if isinstance(text, dict):
        text = " ".join(str(v) for v in text.values() if v is not None)
    elif isinstance(text, (list, tuple, set)):
        text = " ".join(str(v) for v in text if v is not None)
    else:
        text = str(text)

    text = unescape(text)
    lower_original = text.lower()

    # If old session state contains the HTML card itself, do not display it.
    # This is what caused the visible <div style=...> block.
    html_card_signals = [
        "<div style=", "&lt;div style=", "font-size:",
        "margin-top:.75rem", "background:#0f1e3d", "unsafe_allow_html"
    ]
    if any(sig in lower_original for sig in html_card_signals):
        return ""

    # Remove markdown/code fences.
    text = re.sub(r"```(?:html|json|python)?", " ", text, flags=re.IGNORECASE)
    text = text.replace("```", " ")

    # Remove any remaining HTML tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove JSON-ish wrappers if they leak.
    text = re.sub(r"^[\s\{\[\"']+", "", text)
    text = re.sub(r"[\s\}\]\"']+$", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    # Keep only normal readable text.
    if not text or len(text) < 4:
        return ""

    return text[:max_chars].strip()

def cluster_cards_fixed(labels, title="", ai_cards=None):
    """Render cluster profile cards — no duplicate country names."""
    raw_ids = get_ids()
    unique_cls = sorted([c for c in set(labels) if c != -1])
    if not unique_cls:
        st.info("No clusters to display."); return
    cols = st.columns(min(len(unique_cls), 4))
    for i, cl in enumerate(unique_cls):
        mask = np.array(labels) == cl
        members = get_unique_members(mask, raw_ids)
        n_total = len(members)
        display  = members[:10]
        chips = "".join(f'<span class="chip">{m}</span>' for m in display)
        if n_total > 10:
            chips += f'<span class="chip">+{n_total-10} more</span>'

        stats = ""
        for f in sel_feats[:5]:
            if f in df_sub.columns:
                v = df_sub.loc[mask, f].mean()
                stats += (f'<div style="font-size:.74rem;color:#64748B;margin-top:.22rem;">'
                          f'<b style="color:#94A3B8">{f.replace("_"," ").title()[:22]}:</b>'
                          f' {v:.3f}</div>')

        ai = (ai_cards or {}).get(str(cl), {}) if ai_cards else {}
        ai_name = clean_ai_card_text(ai.get("name", f"Cluster {cl}"), max_chars=70)
        ai_exp = clean_ai_card_text(ai.get("explanation", ""), max_chars=420)
        ai_fact = clean_ai_card_text(ai.get("facts", ""), max_chars=360)

        ai_name = escape(ai_name or f"Cluster {cl}")
        ai_exp = escape(ai_exp)
        ai_fact = escape(ai_fact)

        # Do NOT put AI text inside the raw HTML card.
        # Streamlit renders it safely below the card inside the same column.
        ai_block = ""

        color = PAL[i % len(PAL)]
        with cols[i % 4]:
            st.markdown(f"""
            <div class="cl-card" style="--cc:{color};background:#12172A;border:1px solid #263149;border-left:4px solid {color};border-radius:14px;padding:1rem;margin-bottom:.75rem;">
                <div class="cl-title" style="font-weight:850;color:#F8FAFC;">{ai_name}
                    <span style="font-size:.7rem;font-weight:500;
                                 color:#64748B;margin-left:.4rem;">
                        Cluster {cl} · {n_total} countries
                    </span>
                </div>
                <div class="chip-grid" style="margin-top:.45rem;">{chips}</div>
                {stats}
            </div>""", unsafe_allow_html=True)

            if ai_exp or ai_fact:
                st.markdown(
                    f"""
                    <div style="margin-top:-.45rem;margin-bottom:.75rem;padding:.65rem .75rem;background:#0F1E3D;border:1px solid #1D4ED8;border-radius:10px;overflow-wrap:break-word;">
                        <div style="font-size:.72rem;font-weight:850;color:#93C5FD;margin-bottom:.25rem;">🤖 AI explanation</div>
                        <div style="font-size:.75rem;color:#C7D2FE;line-height:1.45;white-space:normal;">{ai_exp}</div>
                        <div style="font-size:.72rem;color:#93C5FD;line-height:1.45;margin-top:.35rem;white-space:normal;">{ai_fact}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

def size_bar(labels, title="Cluster Sizes"):
    uniq, cnts = np.unique([l for l in labels if l != -1], return_counts=True)
    fig = px.bar(x=[f"Cluster {u}" for u in uniq], y=cnts,
                 color=[f"Cluster {u}" for u in uniq],
                 color_discrete_sequence=PAL,
                 labels={"x":"Cluster","y":"Count","color":"Cluster"},
                 text=cnts)
    fig.update_traces(textposition="outside", textfont=dict(color="#E2E8F0"))
    fig.update_layout(**fig_layout(300), showlegend=False, title=title)
    return fig

def explain_metrics(sil, db, ch):
    st.markdown(f"""
    <div class="info-strip" style="margin-top:.5rem;">
    📊 <b>Cluster quality summary:</b><br>
    <b>Silhouette = {sil:.3f}</b> (higher means points are closer to their own cluster than to other clusters).<br>
    <b>Davies-Bouldin = {db:.3f}</b> (lower means compact and separated clusters).<br>
    <b>Calinski-Harabasz = {ch:.1f}</b> (higher means stronger separation).<br>
    Treat these as guidance, not as absolute truth. A useful clustering must also make domain sense.
    </div>""", unsafe_allow_html=True)

def estimate_elbow_k(inertias, ks):
    # maximum distance from line joining first and last point
    x=np.array(ks, dtype=float); y=np.array(inertias, dtype=float)
    if len(x)<3: return int(ks[0])
    p1=np.array([x[0], y[0]]); p2=np.array([x[-1], y[-1]])
    d=[]
    for xi, yi in zip(x,y):
        p=np.array([xi,yi]); d.append(abs(np.cross(p2-p1, p1-p))/np.linalg.norm(p2-p1))
    return int(ks[int(np.argmax(d))])

def similarity_explorer(labels, title):
    if not (id_col and id_col in df_sub.columns):
        return
    ids=df_sub[id_col].astype(str).tolist()
    choice=st.selectbox(f"Choose a country/item to explain similarity — {title}", sorted(list(dict.fromkeys(ids))), key=f"sim_{title}")
    idx=ids.index(choice)
    same=np.where(labels==labels[idx])[0]
    same=[i for i in same if i!=idx]
    if len(same)==0:
        st.info("No same-cluster neighbours to compare."); return
    # nearest same-cluster rows in scaled feature space
    dists=[(i, float(np.linalg.norm(X[i]-X[idx]))) for i in same]
    dists=sorted(dists, key=lambda x:x[1])[:5]
    rows=[]
    for i,d in dists:
        diffs=[]
        for f in sel_feats:
            a=df_sub.iloc[idx][f]; b=df_sub.iloc[i][f]
            diffs.append((f, abs(a-b), a, b))
        diffs=sorted(diffs, key=lambda t:t[1])[:4]
        rows.append({"Similar country": ids[i], "Distance": round(d,3), "Closest matching indicators": "; ".join([f"{f.replace('_',' ').title()}: {a:.2f} vs {b:.2f}" for f,_,a,b in diffs])})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown('<div class="info-strip">The table explains similarity using the actual preprocessed feature values. Smaller distance means the two countries have more similar profiles across the selected indicators.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
tab_km, tab_db, tab_hc, tab_cmp = st.tabs(["K-Means", "DBSCAN", "Hierarchical", "📊 Comparison"])

# ── K-MEANS ───────────────────────────────────────────────────────────────────
with tab_km:
    st.markdown("""
    <div class="info-strip">
    🔵 <b>K-Means</b> divides your data into exactly K groups by minimising the distance of
    each point to its group centre. Use the <b>Elbow curve</b> to choose K — look for the
    "elbow" where adding more clusters stops giving much improvement.
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c2: km_init = st.selectbox("Initialisation", ["k-means++","random"], key="km_init")

    with st.spinner("Computing elbow curve and suggested K…"):
        inertias, sils, dbs, chs = [], [], [], []
        k_values = list(range(2, min(13, len(X))))
        for ki in k_values:
            km_ = KMeans(n_clusters=ki, init=km_init, random_state=42, n_init=10, max_iter=300)
            lb_ = km_.fit_predict(X)
            inertias.append(km_.inertia_)
            sils.append(silhouette_score(X, lb_))
            dbs.append(davies_bouldin_score(X, lb_))
            chs.append(calinski_harabasz_score(X, lb_))
        suggested_k = estimate_elbow_k(inertias, k_values)
    with c1: k = st.slider("Number of clusters K", 2, max(k_values), suggested_k, key="km_k", help="Default is suggested by the elbow method.")
    with c3: st.markdown(f'<div class="ok-strip">Suggested K by elbow: <b>{suggested_k}</b></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_el = go.Figure()
        fig_el.add_trace(go.Scatter(x=k_values, y=inertias,
            mode="lines+markers", name="WCSS Inertia",
            line=dict(color="#7C3AED", width=2.5), marker=dict(size=8)))
        fig_el.add_vline(x=k, line_dash="dash", line_color="#D97706",
                         annotation_text=f"K={k}", annotation_font_color="#D97706")
        fig_el.update_layout(**fig_layout(300), title="Elbow Curve — pick K at the elbow",
                              xaxis_title="K (number of clusters)", yaxis_title="WCSS Inertia")
        st.plotly_chart(fig_el, use_container_width=True)
        st.markdown("""
        <div class="info-strip" style="margin-top:-.5rem;">
        📌 The <b>elbow point</b> is where the curve bends — adding more clusters
        after that gives diminishing returns.
        </div>""", unsafe_allow_html=True)

    with c2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(x=k_values, y=sils,
            mode="lines+markers", name="Silhouette",
            line=dict(color="#059669", width=2.5), marker=dict(size=8)))
        ch_n = [v/max(chs) for v in chs]
        fig_sil.add_trace(go.Scatter(x=k_values, y=ch_n,
            mode="lines+markers", name="Calinski-Harabasz (norm.)",
            line=dict(color="#2563EB", width=2, dash="dot"), marker=dict(size=5)))
        fig_sil.add_vline(x=k, line_dash="dash", line_color="#D97706")
        fig_sil.update_layout(**fig_layout(300),
                               title="Quality Scores — higher = better",
                               xaxis_title="K", yaxis_title="Score")
        st.plotly_chart(fig_sil, use_container_width=True)

    # Run
    km_final  = KMeans(n_clusters=k, init=km_init, random_state=42, n_init=10)
    labels_km = km_final.fit_predict(X)
    sil_km = silhouette_score(X, labels_km)
    db_km  = davies_bouldin_score(X, labels_km)
    ch_km  = calinski_harabasz_score(X, labels_km)

    stat_row([
        ("Silhouette Score",       f"{sil_km:.4f}", "↑ closer to 1 is better", "#7C3AED"),
        ("Davies-Bouldin",         f"{db_km:.4f}",  "↓ lower is better",       "#2563EB"),
        ("Calinski-Harabasz",      f"{ch_km:.1f}",  "↑ higher is better",      "#059669"),
        ("Clusters",               str(k),           "user-defined",            "#D97706"),
    ])
    explain_metrics(sil_km, db_km, ch_km)

    vt1, vt2, vt3, vt4, vt5 = st.tabs(["PCA Map","World Map","Radar","Heatmap","Sizes"])
    with vt1:
        st.plotly_chart(pca_scatter(labels_km, f"K-Means (K={k}) — PCA 2D Map"),
                        use_container_width=True)
        st.markdown("""<div class="info-strip">
        📌 Each dot is a country. <b>Distance = similarity</b> — nearby dots share similar
        economic and social profiles. Colour = cluster assignment.
        </div>""", unsafe_allow_html=True)
    with vt2:
        fig_ch = world_map(labels_km, f"K-Means (K={k}) — Cluster World Map")
        if fig_ch:
            st.plotly_chart(fig_ch, use_container_width=True)
            st.markdown("""<div class="info-strip">
            📌 Countries with the same colour share similar development profiles —
            notice how clusters often cross traditional geographic regions.
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Add a 'code' column (ISO 3-letter country code) to enable the world map.")
    with vt3:
        st.plotly_chart(radar_chart(labels_km, "Cluster Feature Radar"),
                        use_container_width=True)
        st.markdown("""<div class="info-strip">
        📌 Each shape shows a cluster's average profile. A cluster that bulges outward on
        'GDP per capita' is the wealthy group; one that bulges on 'fertility rate' is
        the high-growth-population group — and so on.
        </div>""", unsafe_allow_html=True)
    with vt4:
        st.plotly_chart(feature_heatmap(labels_km, "Cluster Mean Feature Values"),
                        use_container_width=True)
        st.markdown("""<div class="info-strip">
        📌 Red = high value, blue = low value. Scan each row to see what defines each cluster.
        </div>""", unsafe_allow_html=True)
    with vt5:
        st.plotly_chart(size_bar(labels_km), use_container_width=True)

    st.markdown('<div class="sec-header">👥 AI-Named K-Means Cluster Cards</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-strip">🤖 AI is used only here for K-Means: it names each cluster and explains the shared profile using the selected indicators and country members.</div>', unsafe_allow_html=True)

    ai_sig = f"kmeans-{len(df_sub)}-{k}-{','.join(sel_feats)}-{hash(tuple(map(int, labels_km)))}"
    if "kmeans_ai_sig_v3" not in st.session_state:
        st.session_state.kmeans_ai_sig_v3 = None
    if "kmeans_ai_cards_v3" not in st.session_state:
        st.session_state.kmeans_ai_cards_v3 = None

    run_ai = st.button("🤖 Generate AI cluster names and explanations", key="run_kmeans_ai")
    if run_ai:
        with st.spinner("Generating cluster names and explanations after K-Means is finished..."):
            try:
                ai_cards, ai_err = generate_kmeans_cluster_ai(df_sub, np.array(labels_km), list(sel_feats), id_col)
                st.session_state.kmeans_ai_sig_v3 = ai_sig
                st.session_state.kmeans_ai_cards_v3 = ai_cards
                if ai_err:
                    st.caption("AI note: Groq could not generate a clean response, so the app used smart local fallback names/explanations.")
                else:
                    st.success("AI cluster names and explanations generated.")
            except Exception as e:
                st.session_state.kmeans_ai_sig_v3 = ai_sig
                st.session_state.kmeans_ai_cards_v3 = None
                st.caption(f"AI note: generation failed safely ({type(e).__name__}). Showing normal cluster cards.")

    ai_cards = st.session_state.kmeans_ai_cards_v3 if st.session_state.kmeans_ai_sig_v3 == ai_sig else None
    if ai_cards is None:
        st.info("Click the AI button after the K-Means result appears to name the clusters and add explanations. Until then, normal cluster cards are shown.")

    cluster_cards_fixed(labels_km, ai_cards=ai_cards)
    st.markdown('<div class="sec-header">🔎 Country-to-country similarity explanation</div>', unsafe_allow_html=True)
    similarity_explorer(labels_km, "KMeans")

# ── DBSCAN ────────────────────────────────────────────────────────────────────
with tab_db:
    st.markdown("""
    <div class="info-strip">
    🟠 <b>DBSCAN</b> finds clusters based on <i>density</i> — groups of countries packed
    close together in feature space. It does not need you to specify K.
    Countries in sparse regions are labelled <b>Noise</b> — these are genuinely unusual
    countries that don't fit any cluster. Use the <b>k-NN distance plot</b> to pick ε.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1: eps = st.slider("Epsilon ε (neighbourhood radius)", 0.1, 5.0, 0.8, 0.05, key="db_eps")
    with c2: min_pts = st.slider("Min samples (core point threshold)", 2, 15, 3, key="db_pts")

    # k-NN distance plot
    nbrs = NearestNeighbors(n_neighbors=min_pts).fit(X)
    dists, _ = nbrs.kneighbors(X)
    knn_dist = np.sort(dists[:, min_pts-1])[::-1]
    fig_knn = go.Figure()
    fig_knn.add_trace(go.Scatter(y=knn_dist, mode="lines",
                                  line=dict(color="#7C3AED", width=2),
                                  name=f"{min_pts}-NN distance"))
    fig_knn.add_hline(y=eps, line_dash="dash", line_color="#D97706",
                      annotation_text=f"ε = {eps}", annotation_font_color="#D97706")
    fig_knn.update_layout(**fig_layout(290),
                           title=f"k-NN Distance Plot — choose ε at the elbow",
                           xaxis_title="Points sorted by distance (descending)",
                           yaxis_title=f"{min_pts}-NN Distance")
    st.plotly_chart(fig_knn, use_container_width=True)
    st.markdown("""<div class="info-strip">
    📌 Sort all points by how far they are from their nearest neighbours. The <b>elbow</b>
    (sharp bend) in this curve is the natural separation between dense regions and outliers.
    Set ε to the y-value at that elbow.
    </div>""", unsafe_allow_html=True)

    db = DBSCAN(eps=eps, min_samples=min_pts)
    labels_db = db.fit_predict(X)
    n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    n_noise       = (labels_db == -1).sum()
    n_valid       = len(labels_db) - n_noise

    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters Found", n_clusters_db)
    c2.metric("Noise / Outlier Points", int(n_noise),
              help="Items that don't fit any cluster")
    if n_clusters_db > 1:
        mask_v = labels_db != -1
        sil_db = silhouette_score(X[mask_v], labels_db[mask_v]) if mask_v.sum()>1 else 0
        c3.metric("Silhouette (excl. noise)", f"{sil_db:.4f}")
        explain_metrics(sil_db, 0, 0)
    else:
        c3.metric("Silhouette", "N/A")
        st.markdown('<div class="warn-strip">⚠ Only one cluster found — try lowering ε or min_samples.</div>',
                    unsafe_allow_html=True)

    if n_noise > 0:
        st.markdown(f"""<div class="info-strip">
        🔍 <b>{n_noise} noise points detected</b> — these are countries that don't
        fit any cluster. They may be genuine outliers (unusual economic or social conditions)
        worth investigating individually.
        </div>""", unsafe_allow_html=True)
        if id_col and id_col in df_sub.columns:
            noise_ids = list(dict.fromkeys(
                df_sub.loc[labels_db==-1, id_col].astype(str).tolist()))
            st.markdown("**Noise / Outlier items:** " +
                        ", ".join(f"`{n}`" for n in noise_ids[:20]))

    vt1, vt2, vt3 = st.tabs(["PCA Map","World Map","Profiles"])
    with vt1:
        st.plotly_chart(pca_scatter(labels_db, f"DBSCAN (ε={eps}, minPts={min_pts})"),
                        use_container_width=True)
    with vt2:
        fig_dbm = world_map(labels_db, "DBSCAN — World Map")
        if fig_dbm: st.plotly_chart(fig_dbm, use_container_width=True)
        else: st.info("No 'code' column for world map.")
    with vt3:
        if n_clusters_db > 0:
            cluster_cards_fixed(labels_db)
            st.markdown('<div class="sec-header">🔎 Country-to-country similarity explanation</div>', unsafe_allow_html=True)
            similarity_explorer(labels_db, "DBSCAN")
        else:
            st.warning("No clusters formed. Adjust ε or min_samples.")

# ── HIERARCHICAL ──────────────────────────────────────────────────────────────
with tab_hc:
    st.markdown("""
    <div class="info-strip">
    🟢 <b>Hierarchical clustering</b> builds a <b>family tree</b> (dendrogram) of all items.
    Start with every country in its own cluster, then progressively merge the two most similar
    clusters until everything is one group. You can cut the tree at any level to get K clusters.
    Great for understanding how groups relate — which clusters are most similar to each other?
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        linkage_m = st.selectbox("Linkage method",
                                  ["ward","complete","average","single"], key="hc_link",
                                  help="Ward minimises within-cluster variance (recommended). "
                                       "Complete uses maximum distance. Average uses mean distance.")
    with c2:
        n_hc = st.slider("Number of clusters (where to cut the tree)", 2, 12, 4, key="hc_k")

    st.markdown(f"""<div class="info-strip">
    📌 <b>Linkage = {linkage_m}</b>: {'Ward method — merges clusters that produce the smallest increase in total within-cluster variance. Best general-purpose choice.' if linkage_m=='ward' else 'Complete linkage — merges clusters based on their maximum pairwise distance. Good for compact clusters.' if linkage_m=='complete' else 'Average linkage — uses the average distance between all pairs. A balanced compromise.' if linkage_m=='average' else 'Single linkage — uses the minimum distance. Can create "chaining" where clusters stretch out.'}
    </div>""", unsafe_allow_html=True)

    # Dendrogram
    st.markdown('<div class="sec-header">🌳 Dendrogram — Family Tree of Countries</div>',
                unsafe_allow_html=True)
    st.markdown("""<div class="info-strip">
    📌 Read from <b>bottom to top</b>. Each leaf is one country. When two branches merge,
    those countries/groups joined a cluster. The <b>height of the merge</b> shows how different
    they were — taller = more different. The <b>purple dashed line</b> shows where we cut to
    get your chosen number of clusters.
    </div>""", unsafe_allow_html=True)

    try:
        # Subsample for large datasets to keep dendrogram readable
        MAX_DEND = 80
        if len(X) > MAX_DEND:
            idx_s = np.random.choice(len(X), MAX_DEND, replace=False)
            X_d   = X[idx_s]
            note  = f" (random sample of {MAX_DEND}/{len(X)} items)"
        else:
            X_d   = X
            note  = ""

        Z = sc_linkage(X_d, method=linkage_m)
        cut_h = Z[-n_hc, 2]

        fig_d, ax = plt.subplots(figsize=(16, 4.5), facecolor="#12172A")
        ax.set_facecolor("#0F1117")
        for sp in ax.spines.values(): sp.set_color("#1E2333")
        ax.tick_params(colors="#4B5563", labelsize=7)

        dendrogram(Z, ax=ax, truncate_mode=None if len(X_d) <= 40 else "lastp",
                   p=40, leaf_rotation=75, leaf_font_size=7,
                   color_threshold=cut_h, above_threshold_color="#374151")
        ax.axhline(y=cut_h, color="#7C3AED", linestyle="--",
                   linewidth=2, alpha=.9, label=f"Cut → {n_hc} clusters")
        ax.set_xlabel("Country / Sample" + note, color="#4B5563", fontsize=9)
        ax.set_ylabel("Merge Distance", color="#4B5563", fontsize=9)
        ax.set_title(f"Dendrogram — {linkage_m.title()} linkage{note}",
                     color="#E2E8F0", fontsize=11, fontweight="bold", pad=12)
        ax.legend(fontsize=8, facecolor="#12172A", labelcolor="#94A3B8")
        plt.tight_layout(pad=.5)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#12172A")
        buf.seek(0)
        st.image(buf, use_container_width=True)
        plt.close()
    except Exception as e:
        st.warning(f"Dendrogram error: {e}")

    hc        = AgglomerativeClustering(n_clusters=n_hc, linkage=linkage_m)
    labels_hc = hc.fit_predict(X)
    sil_hc = silhouette_score(X, labels_hc)
    db_hc  = davies_bouldin_score(X, labels_hc)
    ch_hc  = calinski_harabasz_score(X, labels_hc)

    stat_row([
        ("Silhouette",        f"{sil_hc:.4f}", "↑ better",    "#7C3AED"),
        ("Davies-Bouldin",    f"{db_hc:.4f}",  "↓ better",    "#2563EB"),
        ("Calinski-Harabasz", f"{ch_hc:.1f}",  "↑ better",    "#059669"),
        ("Linkage",           linkage_m,        "method used", "#D97706"),
    ])
    explain_metrics(sil_hc, db_hc, ch_hc)

    vt1, vt2, vt3, vt4 = st.tabs(["PCA Map","World Map","Radar","Profiles"])
    with vt1:
        st.plotly_chart(pca_scatter(labels_hc, f"Hierarchical ({linkage_m}, K={n_hc})"),
                        use_container_width=True)
    with vt2:
        fig_hcm = world_map(labels_hc, "Hierarchical — World Map")
        if fig_hcm: st.plotly_chart(fig_hcm, use_container_width=True)
        else: st.info("No 'code' column for world map.")
    with vt3:
        st.plotly_chart(radar_chart(labels_hc, "Cluster Feature Radar"),
                        use_container_width=True)
    with vt4:
        cluster_cards_fixed(labels_hc)
        st.markdown('<div class="sec-header">🔎 Country-to-country similarity explanation</div>', unsafe_allow_html=True)
        similarity_explorer(labels_hc, "Hierarchical")

# ── COMPARISON ────────────────────────────────────────────────────────────────
with tab_cmp:
    st.markdown("""
    <div class="info-strip">
    📊 Compare all three algorithms at the same number of clusters.
    There is no single "best" algorithm — each suits different data shapes.
    Use this table to see which performs best on <i>your</i> data.
    </div>""", unsafe_allow_html=True)

    k_cmp = st.slider("K for comparison", 2, 10, 4, key="cmp_k")
    rows  = []
    for nm, mo in [
        ("K-Means",           KMeans(n_clusters=k_cmp, random_state=42, n_init=10)),
        ("Hierarchical-Ward", AgglomerativeClustering(n_clusters=k_cmp, linkage="ward")),
        ("Hierarchical-Avg",  AgglomerativeClustering(n_clusters=k_cmp, linkage="average")),
        ("Hierarchical-Comp", AgglomerativeClustering(n_clusters=k_cmp, linkage="complete")),
    ]:
        lb = mo.fit_predict(X)
        rows.append({"Algorithm": nm, "Clusters": len(set(lb)),
                     "Silhouette ↑": round(silhouette_score(X, lb), 4),
                     "Davies-Bouldin ↓": round(davies_bouldin_score(X, lb), 4),
                     "Calinski-Harabasz ↑": round(calinski_harabasz_score(X, lb), 1)})

    ld = DBSCAN(eps=0.8, min_samples=3).fit_predict(X)
    nc = len(set(ld)) - (1 if -1 in ld else 0)
    mv = ld != -1
    sv = silhouette_score(X[mv], ld[mv]) if mv.sum()>1 and nc>1 else 0
    rows.append({"Algorithm":"DBSCAN (ε=0.8)", "Clusters": nc,
                 "Silhouette ↑": round(sv,4),
                 "Davies-Bouldin ↓": "—", "Calinski-Harabasz ↑": "—"})

    cdf = pd.DataFrame(rows)
    st.dataframe(cdf, use_container_width=True, hide_index=True)

    # Bar comparison
    num_rows = [r for r in rows if isinstance(r["Silhouette ↑"], float)]
    fig_bar = px.bar(pd.DataFrame(num_rows), x="Algorithm", y="Silhouette ↑",
                     color="Algorithm", color_discrete_sequence=PAL,
                     text="Silhouette ↑")
    fig_bar.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                           textfont=dict(color="#E2E8F0"))
    fig_bar.update_layout(**fig_layout(380), showlegend=False,
                           title="Silhouette Score by Algorithm — higher is better",
                           yaxis_range=[0, max(r["Silhouette ↑"] for r in num_rows)*1.2])
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("""
    <div class="info-strip">
    🧠 <b>Which algorithm should I trust?</b><br>
    • <b>K-Means</b> is fast, reliable, and well-understood. Start here.<br>
    • <b>Hierarchical-Ward</b> often performs similarly to K-Means but gives you the dendrogram.<br>
    • <b>DBSCAN</b> is unique — it finds noise/outliers and doesn't need K. If many countries
      are "Noise", that itself is a meaningful finding (they don't fit any group pattern).<br>
    • The algorithm with the <b>highest Silhouette</b> and <b>lowest Davies-Bouldin</b> fits your
      data best — but always validate with domain knowledge.
    </div>""", unsafe_allow_html=True)
