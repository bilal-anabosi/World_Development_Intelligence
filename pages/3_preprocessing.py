import streamlit as st, pandas as pd, numpy as np, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import CSS, render_sidebar, init_state, page_header, stat_row, fig_layout, label
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
from sklearn.feature_selection import VarianceThreshold

st.set_page_config(page_title="Prepare Data", page_icon="🧹", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True); init_state(); render_sidebar()
page_header("🧹", "Prepare Data", "A full preprocessing pipeline with 8 practical steps before data mining.")
if st.session_state.raw_df is None:
    st.warning("Build or upload a dataset first."); st.stop()

df=st.session_state.raw_df.copy()
all_cols=df.columns.tolist()
num_guess=df.select_dtypes(include=np.number).columns.tolist()
cat_cols=df.select_dtypes(exclude=np.number).columns.tolist()

st.markdown('<div class="sec-header">Step 1 — Choose analysis columns</div>', unsafe_allow_html=True)
c1,c2=st.columns([1,1])
with c1:
    sel_num=st.multiselect("Numeric features for analysis", num_guess, default=[c for c in num_guess if c not in ['year']], format_func=label)
with c2:
    label_col=st.selectbox("Label / group column", ["None"]+all_cols, index=(["None"]+all_cols).index("region") if "region" in all_cols else 0)
    id_col=st.selectbox("Row identifier", ["None"]+all_cols, index=(["None"]+all_cols).index("country") if "country" in all_cols else 0)
if len(sel_num)<2: st.warning("Select at least 2 numeric features."); st.stop()

st.markdown('<div class="sec-header">Step 2 — Data structure checks</div>', unsafe_allow_html=True)
dup_count=int(df.duplicated().sum()); miss_total=int(df[sel_num].isna().sum().sum()); constant=[c for c in sel_num if df[c].nunique(dropna=True)<=1]
stat_row([("Rows",f"{len(df):,}","before preprocessing","#7C3AED"),("Duplicates",dup_count,"full duplicate rows","#D97706"),("Missing",miss_total,"numeric missing cells","#DC2626"),("Constant",len(constant),"near-useless features","#2563EB")])

st.markdown('<div class="sec-header">Step 3 — Duplicates and type conversion</div>', unsafe_allow_html=True)
c1,c2,c3=st.columns(3)
with c1: remove_dups=st.checkbox("Remove duplicate rows", True)
with c2: convert_numeric=st.checkbox("Convert numeric-looking text", True)
with c3: drop_constant=st.checkbox("Remove constant columns", True)

st.markdown('<div class="sec-header">Step 4 — Missing values</div>', unsafe_allow_html=True)
miss_method=st.selectbox("Missing value strategy", ["Fill with mean", "Fill with median", "Fill with mode", "Fill with zero", "Forward/backward fill by ID", "Drop rows with missing values"])
if miss_total:
    miss_df=pd.DataFrame({"Feature":sel_num,"Missing":df[sel_num].isna().sum().values,"Missing %":(df[sel_num].isna().mean().values*100).round(2)})
    st.dataframe(miss_df[miss_df.Missing>0], use_container_width=True, hide_index=True)

st.markdown('<div class="sec-header">Step 5 — Outliers</div>', unsafe_allow_html=True)
out_method=st.selectbox("Outlier strategy", ["Keep all", "Remove rows by IQR", "Cap/Winsorize by IQR", "Remove rows by Z-score", "Cap by percentiles (1%–99%)"])

st.markdown('<div class="sec-header">Step 6 — Skewness transformation</div>', unsafe_allow_html=True)
skew_method=st.selectbox("Transform skewed features", ["None", "Log1p positive skewed features", "Yeo-Johnson power transform"])
skew_threshold=st.slider("Skew threshold", 0.5, 3.0, 1.0, 0.1)

st.markdown('<div class="sec-header">Step 7 — Feature filtering and categorical encoding</div>', unsafe_allow_html=True)
c1,c2=st.columns(2)
with c1:
    variance_filter=st.checkbox("Remove very low variance numeric features", False)
    var_threshold=st.number_input("Variance threshold", value=0.0001, min_value=0.0, format="%.5f")
with c2:
    encode_cats=st.checkbox("One-hot encode selected categorical columns", False)
    enc_cols=st.multiselect("Categorical columns to encode", cat_cols, default=[])

st.markdown('<div class="sec-header">Step 8 — Scaling</div>', unsafe_allow_html=True)
scaling=st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "QuantileTransformer", "None"])

if st.button("🚀 Apply preprocessing", type="primary"):
    log=[]; work=df.copy()
    if convert_numeric:
        for c in work.columns:
            if work[c].dtype=='object':
                converted=pd.to_numeric(work[c].astype(str).str.replace(',','').str.replace('%',''), errors='coerce')
                if converted.notna().sum() > 0.7*len(work) and c not in [id_col,label_col]:
                    work[c]=converted; log.append(f"Converted {c} to numeric")
    if remove_dups:
        before=len(work); work=work.drop_duplicates(); log.append(f"Removed {before-len(work)} duplicate rows")
    sel=[c for c in sel_num if c in work.columns]
    if drop_constant:
        const=[c for c in sel if work[c].nunique(dropna=True)<=1]
        work=work.drop(columns=const); sel=[c for c in sel if c not in const]; log.append(f"Dropped {len(const)} constant columns")
    X=work[sel].copy()
    if miss_method=="Fill with mean": X=X.fillna(X.mean(numeric_only=True)); log.append("Filled missing values with means")
    elif miss_method=="Fill with median": X=X.fillna(X.median(numeric_only=True)); log.append("Filled missing values with medians")
    elif miss_method=="Fill with mode": X=X.fillna(X.mode().iloc[0]); log.append("Filled missing values with modes")
    elif miss_method=="Fill with zero": X=X.fillna(0); log.append("Filled missing values with zero")
    elif miss_method=="Forward/backward fill by ID" and id_col != "None" and id_col in work.columns:
        X=work[[id_col]+sel].sort_index().groupby(id_col)[sel].ffill().bfill(); log.append("Forward/backward filled by ID")
    else:
        before=len(X); X=X.dropna(); work=work.loc[X.index]; log.append(f"Dropped {before-len(X)} rows with missing values")
    if out_method!="Keep all":
        before=len(X)
        if "IQR" in out_method:
            Q1,Q3=X.quantile(.25),X.quantile(.75); IQR=Q3-Q1
            lo,hi=Q1-1.5*IQR,Q3+1.5*IQR
            if out_method.startswith("Remove"):
                mask=~((X<lo)|(X>hi)).any(axis=1); X=X[mask]; work=work.loc[X.index]; log.append(f"Removed {before-len(X)} IQR outlier rows")
            else:
                X=X.clip(lo,hi,axis=1); log.append("Capped outliers using IQR bounds")
        elif "Z-score" in out_method:
            z=np.abs((X-X.mean())/(X.std()+1e-9)); mask=(z<3).all(axis=1); X=X[mask]; work=work.loc[X.index]; log.append(f"Removed {before-len(X)} Z-score outlier rows")
        else:
            X=X.clip(X.quantile(.01), X.quantile(.99), axis=1); log.append("Capped features at 1st–99th percentiles")
    if skew_method!="None":
        skewed=[c for c in X.columns if abs(X[c].skew())>=skew_threshold]
        if skew_method.startswith("Log"):
            for c in skewed:
                if X[c].min()>=0: X[c]=np.log1p(X[c])
            log.append(f"Applied log1p to {len(skewed)} skewed non-negative features")
        else:
            pt=PowerTransformer(method='yeo-johnson'); X[skewed]=pt.fit_transform(X[skewed]); log.append(f"Applied Yeo-Johnson to {len(skewed)} skewed features")
    if variance_filter and X.shape[1]>1:
        vt=VarianceThreshold(threshold=var_threshold); arr=vt.fit_transform(X); kept=X.columns[vt.get_support()].tolist(); log.append(f"Removed {X.shape[1]-len(kept)} low-variance features"); X=pd.DataFrame(arr, columns=kept, index=X.index)
    scaler=None
    if scaling=="StandardScaler": scaler=StandardScaler()
    elif scaling=="MinMaxScaler": scaler=MinMaxScaler()
    elif scaling=="RobustScaler": scaler=RobustScaler()
    elif scaling=="MaxAbsScaler": scaler=MaxAbsScaler()
    elif scaling=="QuantileTransformer": scaler=QuantileTransformer(output_distribution='normal', random_state=42)
    if scaler is not None:
        X=pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index); log.append(f"Applied {scaling}")
    final=work.loc[X.index].drop(columns=[c for c in X.columns if c in work.columns], errors='ignore').reset_index(drop=True)
    final=pd.concat([final, X.reset_index(drop=True)], axis=1)
    if encode_cats and enc_cols:
        final=pd.get_dummies(final, columns=[c for c in enc_cols if c in final.columns], drop_first=False); log.append(f"One-hot encoded {len(enc_cols)} categorical columns")
    st.session_state.clean_df=final
    st.session_state.pp_num_cols=X.columns.tolist()
    st.session_state.pp_label_col=None if label_col=="None" else label_col
    st.session_state.pp_id_col=None if id_col=="None" else id_col
    st.session_state.pp_scaling=scaling
    stat_row([("Output rows",f"{len(final):,}","after preprocessing","#059669"),("Features",len(X.columns),"numeric analysis columns","#7C3AED"),("Scaler",scaling,"selected","#2563EB"),("Steps",len(log),"actions applied","#D97706")])
    for item in log: st.markdown(f'<div class="ok-strip">✅ {item}</div>', unsafe_allow_html=True)
    st.dataframe(final.head(15), use_container_width=True)
