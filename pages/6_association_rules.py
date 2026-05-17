import streamlit as st, pandas as pd, numpy as np, os, sys, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import CSS, render_sidebar, init_state, page_header, stat_row, fig_layout, PAL, label, collapse_country_year_panel
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px

st.set_page_config(page_title="Pattern Rules", page_icon="🔗", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True); init_state(); render_sidebar()
page_header("🔗", "Pattern Rules — Apriori", "Discover readable development patterns using Apriori association rule mining.")
if st.session_state.clean_df is None:
    st.warning("Run preprocessing first."); st.stop()

df=st.session_state.clean_df.copy(); num_cols=[c for c in st.session_state.pp_num_cols if c in df.columns]
label_col = st.session_state.get("pp_label_col")
id_col = st.session_state.get("pp_id_col")
df, num_cols, _collapsed_panel = collapse_country_year_panel(df, num_cols, id_col, label_col)
if _collapsed_panel:
    st.markdown('<div class="info-strip">📌 Pattern Rules are mined from one averaged profile per country, not from repeated country-year rows. Time Series still uses the original yearly data.</div>', unsafe_allow_html=True)
if len(num_cols)<2: st.warning("Need at least 2 numeric features."); st.stop()

st.markdown('<div class="info-strip"><b>Algorithm:</b> Apriori only. The filters are tuned to avoid hundreds of weak rules and keep a readable number of meaningful relationships.</div>', unsafe_allow_html=True)
c1,c2,c3,c4=st.columns(4)
with c1: feats=st.multiselect("Features", num_cols, default=num_cols[:min(8,len(num_cols))], format_func=label)
with c2: bins=st.slider("Bins",2,4,3); min_sup=st.slider("Min support",0.05,0.50,0.18,0.01)
with c3: min_conf=st.slider("Min confidence",0.30,0.95,0.65,0.01); min_lift=st.slider("Min lift",1.0,5.0,1.35,0.05)
with c4: max_rules=st.slider("Max rules to show",10,60,25,5)
if len(feats)<2: st.warning("Choose at least 2 features."); st.stop()

labels_map={2:["Low","High"],3:["Low","Medium","High"],4:["Very Low","Low","High","Very High"]}
def item_name(level, col): return f"{level} {label(col)}"

disc=pd.DataFrame(index=df.index)
for col in feats:
    try:
        disc[col]=pd.qcut(df[col], bins, labels=[item_name(x,col) for x in labels_map[bins]], duplicates='drop')
    except Exception:
        try: disc[col]=pd.cut(df[col], bins, labels=[item_name(x,col) for x in labels_map[bins]], duplicates='drop')
        except Exception: pass

disc=disc.dropna()
transactions=[[str(v) for v in row.values if pd.notna(v)] for _,row in disc.iterrows()]
te=TransactionEncoder(); te_df=pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
with st.spinner("Running Apriori..."):
    freq=apriori(te_df, min_support=min_sup, use_colnames=True)
    if freq.empty:
        st.warning("No itemsets found. Lower support."); st.stop()
    rules=association_rules(freq, metric='lift', min_threshold=min_lift)
    rules=rules[rules.confidence>=min_conf].sort_values(['lift','confidence','support'], ascending=False)
    rules['If']=rules.antecedents.apply(lambda x:' AND '.join(sorted(x)))
    rules['Then']=rules.consequents.apply(lambda x:' AND '.join(sorted(x)))
    rules=rules.head(max_rules)
if rules.empty:
    st.warning("No rules passed the filters. Lower lift or confidence slightly."); st.stop()

stat_row([("Rules",len(rules),"displayed after filters","#7C3AED"),("Max lift",f"{rules.lift.max():.2f}","strongest rule","#059669"),("Avg confidence",f"{rules.confidence.mean():.2f}","reliability","#2563EB"),("Support",f">= {min_sup:.2f}","threshold","#D97706")])

tab1,tab2,tab3=st.tabs(["📋 Rules Table", "💡 Insights", "📉 Graphs"])
with tab1:
    table=rules[['If','Then','support','confidence','lift','leverage','conviction']].round(4)
    st.dataframe(table, use_container_width=True, hide_index=True)
    buf=io.StringIO(); table.to_csv(buf,index=False); st.download_button("Download rules CSV", buf.getvalue(), "apriori_rules.csv", "text/csv")
with tab2:
    for i,(_,r) in enumerate(rules.head(10).iterrows(),1):
        strength='very strong' if r.lift>=3 else 'strong' if r.lift>=2 else 'meaningful'
        st.markdown(f'<div class="g-card"><b>Rule {i}</b><br>If <span style="color:#93C5FD">{r.If}</span>, then <span style="color:#86EFAC">{r.Then}</span>.<br><br>This appears in <b>{r.support*100:.1f}%</b> of rows, is correct <b>{r.confidence*100:.1f}%</b> of the time, and has a <b>{strength}</b> lift of <b>{r.lift:.2f}</b>.</div>', unsafe_allow_html=True)
with tab3:
    fig=px.scatter(rules, x='support', y='confidence', color='lift', size='lift', hover_data=['If','Then'], color_continuous_scale='Purples', labels={'support':'Support','confidence':'Confidence','lift':'Lift'})
    fig.update_layout(**fig_layout(430), title='Support vs Confidence — color/size = Lift')
    st.plotly_chart(fig, use_container_width=True)
    top=rules.head(min(15,len(rules))).copy(); top['Rule']=top['If'].str[:28]+' → '+top['Then'].str[:24]
    fig2=px.bar(top, x='lift', y='Rule', orientation='h', color='confidence', color_continuous_scale='Blues')
    fig2.update_layout(**fig_layout(450, margin=dict(l=10,r=15,t=45,b=10)), title='Top Apriori Rules by Lift')
    st.plotly_chart(fig2, use_container_width=True)
