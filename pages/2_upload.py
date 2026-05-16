"""
pages/2_upload.py — Upload & Inspect
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import CSS, render_sidebar, init_state, page_header, step_bar, stat_row, fig_layout, PAL

st.set_page_config(page_title="Upload · DataMine", page_icon="📂",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)
init_state()
render_sidebar()
step_bar(1)

page_header("📂", "Upload & Inspect",
            "Upload any CSV or Excel file and explore its structure before running the pipeline.",
            tags=[("CSV","blue"),("Excel","green"),("Auto-detect","purple")],
            accent="linear-gradient(90deg,#2563EB,#0891B2)")

# ── Upload section ────────────────────────────────────────────────────────────
col_up, col_info = st.columns([1, 1], gap="large")

with col_up:
    st.markdown('<div class="g-card"><div class="g-card-title">📁 Upload File</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV or Excel (max 200 MB)",
                                type=["csv","xlsx","xls"],
                                help="Drag-and-drop or click to browse")
    st.markdown('<div class="info-strip">💡 No data leaves your browser. The app processes everything locally.</div>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Also allow using sample datasets
    st.markdown('<div class="g-card"><div class="g-card-title">🗂️ Or Use a Sample Dataset</div>', unsafe_allow_html=True)
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "sample_datasets")
    samples = {}
    if os.path.isdir(sample_dir):
        for f in os.listdir(sample_dir):
            if f.endswith(".csv"):
                samples[f] = os.path.join(sample_dir, f)

    if samples:
        chosen = st.selectbox("Choose sample", ["— none —"] + list(samples.keys()))
        if st.button("Load Sample", use_container_width=True) and chosen != "— none —":
            df_s = pd.read_csv(samples[chosen])
            st.session_state.raw_df = df_s
            st.session_state.clean_df = None
            st.session_state.dataset_name = chosen
            st.markdown(f'<div class="ok-strip">✅ Loaded <b>{chosen}</b> — {len(df_s):,} rows × {len(df_s.columns)} cols</div>',
                        unsafe_allow_html=True)
    else:
        st.caption("No sample files found in sample_datasets/")
    st.markdown('</div>', unsafe_allow_html=True)

# Handle upload
if uploaded:
    with st.spinner("Reading file…"):
        try:
            if uploaded.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded)
            else:
                df_up = pd.read_excel(uploaded)
            st.session_state.raw_df = df_up
            st.session_state.clean_df = None
            st.session_state.dataset_name = uploaded.name
        except Exception as e:
            st.error(f"Error reading file: {e}")

with col_info:
    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        miss_pct = df.isnull().sum().sum() / df.size * 100

        stat_row([
            ("Rows",      f"{len(df):,}",    "records",         "#7C3AED"),
            ("Columns",   len(df.columns),   "total",           "#2563EB"),
            ("Numeric",   len(num_cols),      "features",        "#059669"),
            ("Missing",   f"{miss_pct:.1f}%","of all values",   "#D97706"),
        ])
        st.markdown(f'<div class="ok-strip">✅ <b>{st.session_state.dataset_name}</b> is loaded and ready.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="height:100%;display:flex;align-items:center;justify-content:center;
                    padding:3rem;text-align:center;color:#374151;">
            <div>
                <div style="font-size:2.5rem;margin-bottom:.8rem;">👆</div>
                <div style="font-size:.95rem;">Upload a file to see its statistics here.</div>
            </div>
        </div>""", unsafe_allow_html=True)

# ── Inspection tabs ───────────────────────────────────────────────────────────
if st.session_state.raw_df is not None:
    df = st.session_state.raw_df
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    st.markdown('<div class="sec-header">🔍 Dataset Inspection</div>', unsafe_allow_html=True)
    t1, t2, t3, t4, t5 = st.tabs(["📋 Preview", "📊 Column Info", "📈 Statistics", "📉 Distributions", "🔥 Correlation"])

    with t1:
        n = st.slider("Rows to show", 5, min(200, len(df)), 15, key="prev_n")
        st.dataframe(df.head(n), use_container_width=True)

    with t2:
        rows = []
        for c in df.columns:
            n_null = df[c].isnull().sum()
            rows.append({
                "Column": c,
                "Type": str(df[c].dtype),
                "Missing": n_null,
                "Missing %": f"{n_null/len(df)*100:.1f}%",
                "Unique": df[c].nunique(),
                "Sample": str(df[c].dropna().iloc[0])[:45] if df[c].dropna().shape[0] else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with t3:
        if num_cols:
            st.dataframe(df[num_cols].describe().round(4), use_container_width=True)
        else:
            st.info("No numeric columns detected.")

    with t4:
        if num_cols:
            c1, c2 = st.columns(2)
            with c1:
                hcol = st.selectbox("Distribution of", num_cols,
                                    format_func=lambda x: x.replace("_"," ").title(), key="hist_col")
                color_hist = st.selectbox("Color by (optional)", ["None"]+cat_cols, key="hist_color")
                fig_h = px.histogram(df, x=hcol, nbins=35,
                                     color=color_hist if color_hist != "None" else None,
                                     color_discrete_sequence=PAL,
                                     labels={hcol: hcol.replace("_"," ").title()})
                fig_h.update_layout(**fig_layout(310), title=hcol.replace("_"," ").title())
                st.plotly_chart(fig_h, use_container_width=True)

            with c2:
                if len(num_cols) >= 2:
                    xcol = st.selectbox("Scatter X", num_cols, index=0, key="sc_x")
                    ycol = st.selectbox("Scatter Y", num_cols, index=min(1,len(num_cols)-1), key="sc_y")
                    color_sc = st.selectbox("Color by", ["None"]+cat_cols, key="sc_c")
                    fig_s = px.scatter(df, x=xcol, y=ycol,
                                       color=color_sc if color_sc != "None" else None,
                                       color_discrete_sequence=PAL,
                                       hover_name=cat_cols[0] if cat_cols else None)
                    fig_s.update_traces(marker=dict(size=8, line=dict(width=.5,color="#1E2333")))
                    fig_s.update_layout(**fig_layout(310))
                    st.plotly_chart(fig_s, use_container_width=True)

            # Box plots
            st.markdown('<div class="sec-header" style="margin-top:.5rem;">📦 Box Plots</div>',
                        unsafe_allow_html=True)
            box_cols = st.multiselect("Features to show", num_cols, default=num_cols[:6],
                                      format_func=lambda x: x.replace("_"," ").title(), key="box_c")
            if box_cols:
                df_melt = df[box_cols].melt(var_name="Feature", value_name="Value")
                fig_box = px.box(df_melt, x="Feature", y="Value",
                                 color="Feature", color_discrete_sequence=PAL)
                fig_box.update_layout(**fig_layout(380), showlegend=False,
                                       xaxis_tickangle=-30)
                st.plotly_chart(fig_box, use_container_width=True)

    with t5:
        if len(num_cols) >= 3:
            sel_corr = st.multiselect("Features for correlation", num_cols, default=num_cols[:12],
                                       format_func=lambda x: x.replace("_"," ").title(), key="corr_f")
            if len(sel_corr) >= 2:
                corr = df[sel_corr].corr()
                fig_corr = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1,
                                     text_auto=".2f", aspect="auto")
                fig_corr.update_traces(textfont=dict(size=8))
                fig_corr.update_layout(paper_bgcolor="#12172A",
                                        font=dict(family="Manrope", size=9, color="#64748B"),
                                        height=max(350, len(sel_corr)*38+60),
                                        margin=dict(l=0,r=0,t=36,b=0),
                                        title=dict(text="Correlation Matrix",
                                                   font=dict(size=13,color="#E2E8F0")),
                                        coloraxis_colorbar=dict(tickfont=dict(color="#64748B")))
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Need at least 3 numeric columns for a correlation matrix.")
