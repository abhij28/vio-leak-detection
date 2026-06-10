import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import base64
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="VIO — Leak & Integrity Detection",
    page_icon="🛢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Color Palette ─────────────────────────────────────────────────
P_NEON   = "#A832FF"
P_MID    = "#4A157D"
P_DARK   = "#140526"
P_CARD   = "#1E0A3C"
P_GLOW   = "rgba(168,50,255,0.18)"
P_WHITE  = "#F0E6FF"
C_GREEN  = "#00E676"
C_RED    = "#FF1744"
C_YELLOW = "#FFD600"
C_ORANGE = "#FF6B35"

# ── CSS ───────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], .stApp {{
    background: linear-gradient(160deg, {P_DARK} 0%, #1A0535 50%, {P_DARK} 100%) !important;
    color: {P_WHITE};
    font-family: 'Inter', sans-serif;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0D011A 0%, {P_DARK} 60%, #0D011A 100%) !important;
    border-right: 1.5px solid {P_NEON}44;
    box-shadow: 4px 0 32px {P_GLOW};
}}
[data-testid="stSidebar"] * {{ color: {P_WHITE} !important; }}

[data-testid="metric-container"] {{
    background: {P_CARD} !important;
    border-radius: 12px !important;
    padding: 16px !important;
    border: 1px solid {P_NEON}33 !important;
    box-shadow: 0 2px 16px {P_GLOW} !important;
}}
[data-testid="metric-container"] label {{
    color: #C8A8FF !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {P_WHITE} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
}}

.vio-header {{
    background: linear-gradient(120deg, {P_DARK} 0%, {P_MID} 60%, {P_DARK} 100%);
    padding: 22px 30px;
    border-bottom: 2px solid {P_NEON}88;
    margin-bottom: 22px;
    border-radius: 0 0 12px 12px;
    box-shadow: 0 4px 32px {P_GLOW};
}}

.risk-card {{
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin: 10px 0;
}}

.metric-pill {{
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    margin: 2px;
}}

.stButton > button {{
    background: linear-gradient(135deg, {P_MID} 0%, {P_NEON} 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 16px {P_GLOW} !important;
}}
.stButton > button:hover {{
    box-shadow: 0 4px 28px {P_NEON}66 !important;
    transform: translateY(-1px) !important;
}}

.stSelectbox > div > div {{
    background: {P_CARD} !important;
    border: 1px solid {P_NEON}44 !important;
    border-radius: 8px !important;
}}

.stDataFrame {{
    border: 1px solid {P_NEON}33 !important;
    border-radius: 10px !important;
}}

.stTabs [data-baseweb="tab-list"] {{
    background: {P_CARD} !important;
    border-radius: 10px !important;
    padding: 4px !important;
}}
.stTabs [data-baseweb="tab"] {{ color: #C8A8FF !important; }}
.stTabs [aria-selected="true"] {{
    background: {P_NEON}33 !important;
    color: white !important;
}}

.glow-div {{
    height: 1px;
    background: linear-gradient(90deg, transparent, {P_NEON}, transparent);
    margin: 18px 0;
}}

h3 {{ color: {P_WHITE} !important; font-family: 'Space Grotesk', sans-serif !important; }}
h4 {{ color: #C8A8FF !important; }}
#MainMenu {{visibility:hidden;}} footer {{visibility:hidden;}} header {{visibility:hidden;}}
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = "models/leak"
    m = {}
    try:
        m['classifier'] = joblib.load(f"{base}/model_leak_classifier.pkl")
        m['isolation']  = joblib.load(f"{base}/model_leak_isolation.pkl")
        m['scaler']     = joblib.load(f"{base}/scaler_leak.pkl")
        m['features']   = joblib.load(f"{base}/features_leak.pkl")
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

    try:
        from tensorflow.keras.models import load_model
        m['ae_model']    = load_model(f"{base}/ae_leak.keras")
        m['ae_scaler']   = joblib.load(f"{base}/scaler_ae_leak.pkl")
        m['ae_threshold']= joblib.load(f"{base}/threshold_ae_leak.pkl")
        m['ae_features'] = joblib.load(f"{base}/features_ae_leak.pkl")
        m['ae_loaded']   = True
    except:
        m['ae_loaded'] = False

    return m

# ── Load Data ─────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    return pd.read_csv("data/sample_wells.csv", parse_dates=['Log_Date_Time'])

# ── Prediction logic ──────────────────────────────────────────────
def compute_leak_features(well_df):
    """Compute all leak detection features for a well."""
    df = well_df.sort_values('Log_Date_Time').copy()
    df['CHP_FTHP_diff']    = df['CHP']  - df['FTHP']
    df['CHP_ABP_diff']     = df['CHP']  - df['ABP']
    df['FTHP_ABP_diff']    = df['FTHP'] - df['ABP']
    df['FTHP_mean_6']      = df['FTHP'].rolling(6,  min_periods=1).mean()
    df['FTHP_mean_24']     = df['FTHP'].rolling(24, min_periods=1).mean()
    df['FTHP_std_24']      = df['FTHP'].rolling(24, min_periods=1).std().fillna(0)
    df['CHP_mean_6']       = df['CHP'].rolling(6,   min_periods=1).mean()
    df['CHP_mean_24']      = df['CHP'].rolling(24,  min_periods=1).mean()
    df['CHP_std_24']       = df['CHP'].rolling(24,  min_periods=1).std().fillna(0)
    df['ABP_mean_6']       = df['ABP'].rolling(6,   min_periods=1).mean()
    df['ABP_mean_24']      = df['ABP'].rolling(24,  min_periods=1).mean()
    df['ABP_std_24']       = df['ABP'].rolling(24,  min_periods=1).std().fillna(0)
    df['FTHP_diff']        = df['FTHP'].diff().fillna(0)
    df['CHP_diff']         = df['CHP'].diff().fillna(0)
    df['ABP_diff']         = df['ABP'].diff().fillna(0)
    df['FTHP_sudden_drop'] = (df['FTHP_diff'] < -5).astype(int)
    df['CHP_sudden_drop']  = (df['CHP_diff']  < -5).astype(int)
    df['ABP_sudden_drop']  = (df['ABP_diff']  < -5).astype(int)
    df['FTHP_decline_rate']= df['FTHP_mean_6']  - df['FTHP_mean_24']
    df['CHP_decline_rate'] = df['CHP_mean_6']   - df['CHP_mean_24']
    df['scp_indicator']    = (df['CHP'] > df['FTHP'] * 1.5).astype(int)
    df['CHP_trend']        = df['CHP'].diff().rolling(12, min_periods=1).mean().fillna(0)
    df['FTHP_trend']       = df['FTHP'].diff().rolling(12, min_periods=1).mean().fillna(0)
    df['integrity_alert']  = (
        (df['CHP_FTHP_diff'].abs() > df['CHP_FTHP_diff'].abs().rolling(24, min_periods=1).mean() * 2) |
        (df['FTHP_sudden_drop'] == 1) | (df['scp_indicator'] == 1)
    ).astype(int)
    flt_mean               = df['FLT'].mean()
    flt_std                = df['FLT'].std() + 1e-6
    df['FLT_zscore']       = (df['FLT'] - flt_mean) / flt_std
    df['cross_zone_flag']  = (df['FLT_zscore'].abs() > 2.5).astype(int)
    return df

def run_prediction(row_dict, models):
    features  = models['features']
    scaler    = models['scaler']
    X         = np.array([row_dict.get(f, 0) for f in features]).reshape(1, -1)
    X_sc      = scaler.transform(X)
    severity  = models['classifier'].predict(X_sc)[0]
    iso_score = float(models['isolation'].decision_function(X_sc)[0])

    ae_mse, ae_anom = 0.0, False
    if models.get('ae_loaded'):
        try:
            ae_feat = models['ae_features']
            ae_sc   = models['ae_scaler']
            ae_thr  = models['ae_threshold']
            Xa      = np.array([row_dict.get(f, 0) for f in ae_feat]).reshape(1, -1)
            Xa_sc   = ae_sc.transform(Xa)
            Xa_pred = models['ae_model'].predict(Xa_sc, verbose=0)
            ae_mse  = float(np.mean(np.power(Xa_sc - Xa_pred, 2)))
            ae_anom = ae_mse > ae_thr
        except:
            pass

    if ae_anom and severity == "NO LEAK":
        severity = "LOW RISK"

    leak_score = (
        row_dict.get('FTHP_sudden_drop', 0) * 0.35 +
        row_dict.get('scp_indicator',    0) * 0.25 +
        row_dict.get('integrity_alert',  0) * 0.25 +
        row_dict.get('cross_zone_flag',  0) * 0.15
    )

    return {
        "severity"    : severity,
        "iso_score"   : round(iso_score, 6),
        "ae_mse"      : round(ae_mse, 6),
        "ae_anomaly"  : ae_anom,
        "leak_score"  : round(float(leak_score), 3),
        "scp"         : bool(row_dict.get('scp_indicator', 0)),
        "integrity"   : bool(row_dict.get('integrity_alert', 0)),
        "cross_zone"  : bool(row_dict.get('cross_zone_flag', 0)),
        "sudden_drop" : bool(row_dict.get('FTHP_sudden_drop', 0)),
    }

def severity_color(sev):
    return {
        "NO LEAK"      : C_GREEN,
        "LOW RISK"     : "#88FF88",
        "MEDIUM RISK"  : C_YELLOW,
        "HIGH RISK"    : C_ORANGE,
        "CRITICAL LEAK": C_RED,
    }.get(sev, P_NEON)

def plotly_dark(fig, height=350, **kw):
    fig.update_layout(
        paper_bgcolor=P_DARK, plot_bgcolor=P_CARD,
        font_color=P_WHITE, font_family="Inter",
        height=height, margin=dict(l=10,r=10,t=40,b=10),
        legend=dict(bgcolor=P_CARD, bordercolor=f"rgba(168,50,255,0.27)", borderwidth=1),
        xaxis=dict(gridcolor="rgba(168,50,255,0.09)", zerolinecolor="rgba(168,50,255,0.13)"),
        yaxis=dict(gridcolor="rgba(168,50,255,0.09)", zerolinecolor="rgba(168,50,255,0.13)"),
        **kw
    )
    return fig

# ── Load everything ───────────────────────────────────────────────
models = load_models()
df     = load_data()

well_list = sorted(df['well_id'].unique().tolist())

# ── HEADER ────────────────────────────────────────────────────────
st.markdown(f"""
<div class='vio-header'>
    <span style='font-size:36px; font-weight:900; color:{P_NEON}; letter-spacing:4px;
                 text-shadow:0 0 20px {P_NEON}88; vertical-align:middle;'>VIO</span>
    <div style='display:inline-block; vertical-align:middle; margin-left:18px;'>
        <div style='font-size:11px; color:{P_NEON}; letter-spacing:2px; text-transform:uppercase;'>AI CAPABILITY 04</div>
        <div style='font-size:26px; font-weight:800; color:{P_WHITE}; font-family:"Space Grotesk",sans-serif;'>
            Leak & Integrity Detection
        </div>
    </div>
    <span style='float:right; font-size:11px; color:#C8A8FF; margin-top:8px;'>
        {datetime.now().strftime("%Y-%m-%d %H:%M")} &nbsp;|&nbsp; Demo Mode
    </span>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:20px 0 12px;'>
        <div style='font-size:32px; font-weight:900; color:{P_NEON}; letter-spacing:4px;
                    text-shadow:0 0 16px {P_NEON}88;'>VIO</div>
        <div style='font-size:10px; color:#C8A8FF; letter-spacing:3px; text-transform:uppercase; margin-top:4px;'>
            Intelligence Platform
        </div>
        <div style='height:1px; background:linear-gradient(90deg,transparent,{P_NEON},transparent); margin:12px 0;'></div>
    </div>
    """, unsafe_allow_html=True)

    selected_well = st.selectbox("Select Well", well_list)
    well_type_filter = st.selectbox("Filter by Type", ["All", "Self flow", "Gas lift"])

    model_status = "Ensemble (AE + IsolationForest)" if models and models.get('ae_loaded') else "IsolationForest only"
    st.markdown(f"""
    <div style='margin-top:16px; padding:12px; background:{P_DARK}; border-radius:10px;
         border:1px solid {P_NEON}22; font-size:11px; color:#C8A8FF;'>
        <div style='margin-bottom:4px;'>Model: {model_status}</div>
        <div style='margin-bottom:4px;'>Wells loaded: {len(well_list)}</div>
        <div style='color:{P_NEON}; font-weight:600; letter-spacing:1px;'>LEAK DETECTION ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN CONTENT ──────────────────────────────────────────────────
well_data = df[df['well_id'] == selected_well].copy()
if well_type_filter != "All":
    well_data = well_data[well_data['well_type'] == well_type_filter]
    if len(well_data) == 0:
        st.warning(f"No {well_type_filter} wells match. Showing all types.")
        well_data = df[df['well_id'] == selected_well].copy()

well_data  = well_data.sort_values('Log_Date_Time').reset_index(drop=True)
well_feats = compute_leak_features(well_data)
latest     = well_feats.iloc[-1]
well_type  = well_data['well_type'].iloc[0]

# ── TOP METRICS ───────────────────────────────────────────────────
m1,m2,m3,m4,m5,m6 = st.columns(6)
with m1: st.metric("FTHP",       f"{latest['FTHP']:.1f} psi")
with m2: st.metric("CHP",        f"{latest['CHP']:.1f} psi")
with m3: st.metric("ABP",        f"{latest['ABP']:.1f} psi")
with m4: st.metric("CHP-FTHP",   f"{latest['CHP_FTHP_diff']:.1f} psi")
with m5: st.metric("Well Type",  well_type)
with m6:
    scp_now = "YES" if latest['CHP'] > latest['FTHP'] * 1.5 else "NO"
    st.metric("SCP Alert", scp_now)

st.markdown("<div class='glow-div'></div>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "AI Prediction",
    "Pressure Analysis",
    "Integrity Indicators",
    "Fleet Scan"
])

# ════════════ TAB 1: AI PREDICTION ════════════
with tab1:
    col_pred, col_info = st.columns([1, 1])

    with col_pred:
        st.markdown("### Run AI Leak Detection")
        if st.button("Analyze Latest Reading", type="primary", use_container_width=True):
            if models:
                with st.spinner("Running leak detection model..."):
                    result = run_prediction(latest.to_dict(), models)

                sev   = result['severity']
                color = severity_color(sev)

                st.markdown(f"""
                <div class='risk-card' style='background:{P_CARD}; border:2px solid {color};
                     box-shadow:0 0 40px {color}44;'>
                    <div style='font-size:11px; color:#C8A8FF; letter-spacing:2px; text-transform:uppercase; margin-bottom:10px;'>
                        VIO AI — LEAK RISK ASSESSMENT
                    </div>
                    <div style='font-size:56px; font-weight:900; color:{color}; font-family:"Space Grotesk",sans-serif;
                                text-shadow:0 0 24px {color}88; line-height:1;'>{sev}</div>
                    <div style='margin-top:16px; display:flex; justify-content:center; gap:10px; flex-wrap:wrap;'>
                        <span class='metric-pill' style='background:{color}22; color:{color}; border:1px solid {color}44;'>
                            Score: {result["leak_score"]:.3f}
                        </span>
                        <span class='metric-pill' style='background:{P_NEON}22; color:{P_NEON}; border:1px solid {P_NEON}44;'>
                            ISO: {result["iso_score"]:.4f}
                        </span>
                        {"<span class='metric-pill' style='background:" + C_YELLOW + "22; color:" + C_YELLOW + "; border:1px solid " + C_YELLOW + "44;'>AE Drift Detected</span>" if result["ae_anomaly"] else ""}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                action_map = {
                    "CRITICAL LEAK": ("error",   "CRITICAL: Shut in well immediately. Call field engineer now."),
                    "HIGH RISK"    : ("warning",  "HIGH: Notify field engineer within 2 hours."),
                    "MEDIUM RISK"  : ("warning",  "MEDIUM: Inspect within 24 hours."),
                    "LOW RISK"     : ("info",     "LOW: Log and monitor closely."),
                    "NO LEAK"      : ("success",  "No action required. Well integrity confirmed."),
                }
                action_type, action_msg = action_map.get(sev, ("info", "Monitor well"))
                getattr(st, action_type)(action_msg)

                st.markdown("#### Detection Flags")
                f1,f2,f3,f4 = st.columns(4)
                with f1:
                    col = C_RED if result['sudden_drop'] else C_GREEN
                    st.markdown(f"<div style='text-align:center; padding:10px; background:{P_CARD}; border-radius:8px; border:1px solid {col}44;'><div style='color:{col}; font-weight:700; font-size:11px;'>SUDDEN DROP</div><div style='font-size:20px; color:{col};'>{'YES' if result['sudden_drop'] else 'NO'}</div></div>", unsafe_allow_html=True)
                with f2:
                    col = C_RED if result['scp'] else C_GREEN
                    st.markdown(f"<div style='text-align:center; padding:10px; background:{P_CARD}; border-radius:8px; border:1px solid {col}44;'><div style='color:{col}; font-weight:700; font-size:11px;'>SCP ALERT</div><div style='font-size:20px; color:{col};'>{'YES' if result['scp'] else 'NO'}</div></div>", unsafe_allow_html=True)
                with f3:
                    col = C_ORANGE if result['integrity'] else C_GREEN
                    st.markdown(f"<div style='text-align:center; padding:10px; background:{P_CARD}; border-radius:8px; border:1px solid {col}44;'><div style='color:{col}; font-weight:700; font-size:11px;'>INTEGRITY</div><div style='font-size:20px; color:{col};'>{'ALERT' if result['integrity'] else 'OK'}</div></div>", unsafe_allow_html=True)
                with f4:
                    col = C_YELLOW if result['cross_zone'] else C_GREEN
                    st.markdown(f"<div style='text-align:center; padding:10px; background:{P_CARD}; border-radius:8px; border:1px solid {col}44;'><div style='color:{col}; font-weight:700; font-size:11px;'>CROSS ZONE</div><div style='font-size:20px; color:{col};'>{'FLAG' if result['cross_zone'] else 'OK'}</div></div>", unsafe_allow_html=True)
            else:
                st.error("Models not loaded. Check models/leak/ folder.")

    with col_info:
        st.markdown("### Current Sensor Values")
        sensors = pd.DataFrame({
            "Sensor" : ["FTHP", "CHP", "ABP", "FLT", "CHP-FTHP Diff", "FTHP-ABP Diff", "Battery V"],
            "Value"  : [
                f"{latest['FTHP']:.2f} psi",
                f"{latest['CHP']:.2f} psi",
                f"{latest['ABP']:.2f} psi",
                f"{latest['FLT']:.2f} C",
                f"{latest['CHP_FTHP_diff']:.2f} psi",
                f"{latest['FTHP_ABP_diff']:.2f} psi",
                f"{latest['Battery_Voltage']:.2f} V"
            ]
        })
        st.dataframe(sensors, use_container_width=True, hide_index=True)

        st.markdown("### Risk Guide")
        risk_rows = [
            ("NO LEAK",       C_GREEN,   "Well integrity confirmed"),
            ("LOW RISK",      "#88FF88", "Minor anomaly — log & monitor"),
            ("MEDIUM RISK",   C_YELLOW,  "Inspect within 24 hours"),
            ("HIGH RISK",     C_ORANGE,  "Field engineer within 2 hours"),
            ("CRITICAL LEAK", C_RED,     "Shut-in well immediately"),
        ]
        for label, color, desc in risk_rows:
            st.markdown(f"""
            <div style='display:flex; align-items:center; padding:8px 12px; margin:4px 0;
                 background:{P_CARD}; border-radius:8px; border-left:3px solid {color};'>
                <span style='color:{color}; font-weight:700; font-size:12px; min-width:130px;'>{label}</span>
                <span style='color:#C8A8FF; font-size:11px;'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)

# ════════════ TAB 2: PRESSURE ANALYSIS ════════════
with tab2:
    recent = well_feats.tail(300).copy()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Casing-Tubing Pressure Differential")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent['Log_Date_Time'], y=recent['CHP_FTHP_diff'],
            mode='lines', fill='tozeroy',
            fillcolor='rgba(255,23,68,0.08)',
            line=dict(color=C_RED, width=1.5), name='CHP-FTHP Diff'
        ))
        fig.add_hline(y=0, line_color=P_WHITE, line_dash='dash', line_width=1)
        plotly_dark(fig, height=320, yaxis_title="Differential (psi)", title="Casing vs Tubing Pressure")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### FTHP Sudden Drop Events")
        drop_events = recent[recent['FTHP_sudden_drop'] == 1]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=recent['Log_Date_Time'], y=recent['FTHP'],
            mode='lines', line=dict(color=P_NEON, width=1.5), name='FTHP'
        ))
        fig2.add_trace(go.Scatter(
            x=recent['Log_Date_Time'], y=recent['CHP'],
            mode='lines', line=dict(color=C_ORANGE, width=1.5), name='CHP'
        ))
        if len(drop_events) > 0:
            fig2.add_trace(go.Scatter(
                x=drop_events['Log_Date_Time'], y=drop_events['FTHP'],
                mode='markers',
                marker=dict(color=C_RED, size=10, symbol='x', line=dict(width=2, color=C_RED)),
                name='Sudden Drop'
            ))
        plotly_dark(fig2, height=320, yaxis_title="Pressure (psi)", title="FTHP & CHP with Drop Events")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Multi-Pressure Timeline")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=recent['Log_Date_Time'], y=recent['FTHP'], mode='lines', line=dict(color=P_NEON,   width=1.5), name='FTHP'))
    fig3.add_trace(go.Scatter(x=recent['Log_Date_Time'], y=recent['CHP'],  mode='lines', line=dict(color=C_ORANGE, width=1.5), name='CHP'))
    fig3.add_trace(go.Scatter(x=recent['Log_Date_Time'], y=recent['ABP'],  mode='lines', line=dict(color=C_GREEN,  width=1.5), name='ABP'))
    plotly_dark(fig3, height=280, yaxis_title="Pressure (psi)", title="All Pressures — Last 300 Readings")
    st.plotly_chart(fig3, use_container_width=True)

    d1,d2,d3,d4 = st.columns(4)
    with d1: st.metric("Avg FTHP",    f"{recent['FTHP'].mean():.1f} psi")
    with d2: st.metric("Avg CHP",     f"{recent['CHP'].mean():.1f} psi")
    with d3: st.metric("Avg ABP",     f"{recent['ABP'].mean():.1f} psi")
    with d4: st.metric("Drop Events", int(recent['FTHP_sudden_drop'].sum()))

# ════════════ TAB 3: INTEGRITY INDICATORS ════════════
with tab3:
    recent = well_feats.tail(300).copy()

    scp_count  = int(recent['scp_indicator'].sum())
    drop_count = int(recent['FTHP_sudden_drop'].sum())
    cross_zone = int(recent['cross_zone_flag'].sum())
    integrity  = int(recent['integrity_alert'].sum())
    total_risk = scp_count + drop_count + cross_zone
    overall    = "CRITICAL" if total_risk>50 else "HIGH" if total_risk>20 else "MEDIUM" if total_risk>5 else "LOW"
    ov_color   = severity_color(
        "CRITICAL LEAK" if overall=="CRITICAL" else
        "HIGH RISK"     if overall=="HIGH"     else
        "MEDIUM RISK"   if overall=="MEDIUM"   else "NO LEAK"
    )

    st.markdown(f"""
    <div style='background:{P_CARD}; border:2px solid {ov_color}; border-radius:14px;
         padding:20px; text-align:center; margin-bottom:20px;
         box-shadow: 0 0 32px {ov_color}33;'>
        <div style='font-size:11px; color:#C8A8FF; letter-spacing:2px; text-transform:uppercase;'>OVERALL INTEGRITY RISK</div>
        <div style='font-size:44px; font-weight:900; color:{ov_color}; font-family:"Space Grotesk",sans-serif;
                    text-shadow:0 0 20px {ov_color}66;'>{overall}</div>
        <div style='font-size:12px; color:#8877AA; margin-top:6px;'>Based on last 300 readings</div>
    </div>
    """, unsafe_allow_html=True)

    i1,i2,i3,i4 = st.columns(4)
    with i1: st.metric("SCP Events",        scp_count,  help="Sustained Casing Pressure events")
    with i2: st.metric("Sudden Drops",      drop_count, help="FTHP drops > 5 psi in one step")
    with i3: st.metric("Cross-Zone Flags",  cross_zone, help="FLT z-score > 2.5")
    with i4: st.metric("Integrity Alerts",  integrity,  help="Combined integrity indicator")

    st.markdown("<div class='glow-div'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### SCP Indicator Timeline")
        scp_events = recent[recent['scp_indicator'] == 1]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=recent['Log_Date_Time'], y=recent['CHP_FTHP_diff'], mode='lines', line=dict(color=P_NEON, width=1.5), name='CHP-FTHP'))
        if len(scp_events) > 0:
            fig4.add_trace(go.Scatter(
                x=scp_events['Log_Date_Time'], y=scp_events['CHP_FTHP_diff'],
                mode='markers',
                marker=dict(color=C_RED, size=8, symbol='circle'),
                name='SCP Event'
            ))
        plotly_dark(fig4, height=280, yaxis_title="Differential (psi)")
        st.plotly_chart(fig4, use_container_width=True)

    with c2:
        st.markdown("#### FLT Z-Score (Cross-Zone Indicator)")
        cross_events = recent[recent['cross_zone_flag'] == 1]
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=recent['Log_Date_Time'], y=recent['FLT_zscore'], mode='lines', fill='tozeroy', fillcolor='rgba(168,50,255,0.07)', line=dict(color=P_NEON, width=1.5), name='FLT Z-Score'))
        fig5.add_hline(y= 2.5, line_color=C_RED, line_dash='dash', annotation_text="+2.5 threshold")
        fig5.add_hline(y=-2.5, line_color=C_RED, line_dash='dash', annotation_text="-2.5 threshold")
        if len(cross_events) > 0:
            fig5.add_trace(go.Scatter(x=cross_events['Log_Date_Time'], y=cross_events['FLT_zscore'], mode='markers', marker=dict(color=C_YELLOW, size=8), name='Cross-Zone Flag'))
        plotly_dark(fig5, height=280, yaxis_title="Z-Score")
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("#### Integrity Events Log")
    alerts_df = recent[
        (recent['scp_indicator']==1) |
        (recent['FTHP_sudden_drop']==1) |
        (recent['cross_zone_flag']==1) |
        (recent['integrity_alert']==1)
    ][['Log_Date_Time','FTHP','CHP','ABP','FLT','scp_indicator','FTHP_sudden_drop','cross_zone_flag','integrity_alert']].tail(20)
    if len(alerts_df) > 0:
        st.dataframe(alerts_df, use_container_width=True, hide_index=True)
    else:
        st.success("No integrity events detected in last 300 readings.")

# ════════════ TAB 4: FLEET SCAN ════════════
with tab4:
    st.markdown("### Fleet-Wide Leak Risk Scan")
    st.info("Scans the latest reading for every well in the demo dataset.")

    if st.button("Run Fleet Scan", type="primary", use_container_width=True):
        if not models:
            st.error("Models not loaded.")
        else:
            fleet_results = []
            progress = st.progress(0)
            all_wells = df['well_id'].unique().tolist()

            for i, wid in enumerate(all_wells):
                wd   = df[df['well_id'] == wid].copy()
                wf   = compute_leak_features(wd)
                last = wf.iloc[-1]
                res  = run_prediction(last.to_dict(), models)
                fleet_results.append({
                    "well_id"  : wid,
                    "well_type": wd['well_type'].iloc[0],
                    "FTHP"     : round(float(last['FTHP']), 1),
                    "CHP"      : round(float(last['CHP']),  1),
                    "severity" : res['severity'],
                    "score"    : res['leak_score'],
                    "scp"      : int(res['scp']),
                    "ae_drift" : int(res['ae_anomaly'])
                })
                progress.progress((i+1)/len(all_wells))

            results_df = pd.DataFrame(fleet_results)

            fc1,fc2,fc3,fc4,fc5 = st.columns(5)
            with fc1: st.metric("Wells Scanned",    len(results_df))
            with fc2: st.metric("No Leak",          (results_df['severity']=="NO LEAK").sum())
            with fc3: st.metric("Low / Medium",     ((results_df['severity'].isin(["LOW RISK","MEDIUM RISK"]))).sum())
            with fc4: st.metric("High Risk",        (results_df['severity']=="HIGH RISK").sum())
            with fc5: st.metric("Critical",         (results_df['severity']=="CRITICAL LEAK").sum())

            sev_order = ["NO LEAK","LOW RISK","MEDIUM RISK","HIGH RISK","CRITICAL LEAK"]
            sev_cols  = [C_GREEN, "#88FF88", C_YELLOW, C_ORANGE, C_RED]
            sev_counts= results_df['severity'].value_counts().reindex(sev_order, fill_value=0)

            fig6 = go.Figure(go.Bar(
                x=sev_order, y=sev_counts.values,
                marker_color=sev_cols,
                marker_line_color=P_DARK, marker_line_width=1.5
            ))
            plotly_dark(fig6, height=280, title="Fleet Leak Risk Distribution", yaxis_title="Number of Wells")
            st.plotly_chart(fig6, use_container_width=True)

            def color_severity_cell(val):
                colors = {
                    'CRITICAL LEAK': f'background-color:#2A0015; color:{C_RED}',
                    'HIGH RISK'    : f'background-color:#2A0F00; color:{C_ORANGE}',
                    'MEDIUM RISK'  : f'background-color:#2A2200; color:{C_YELLOW}',
                    'LOW RISK'     : f'background-color:#0A0020; color:{C_GREEN}',
                    'NO LEAK'      : f'background-color:#0D0020; color:{P_NEON}',
                }
                return colors.get(val, '')

            styled = results_df.sort_values('score', ascending=False).style.applymap(
                color_severity_cell, subset=['severity']
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

            critical_wells = results_df[results_df['severity'].isin(['CRITICAL LEAK','HIGH RISK'])]
            if len(critical_wells) > 0:
                st.markdown("#### Wells Requiring Immediate Attention")
                for _, row in critical_wells.iterrows():
                    color = C_RED if row['severity']=='CRITICAL LEAK' else C_ORANGE
                    st.markdown(f"""
                    <div style='background:{P_CARD}; border-left:4px solid {color}; border-radius:10px;
                         padding:14px; margin:6px 0; box-shadow:0 2px 16px {color}22;'>
                        <b style='color:{color};'>{row['severity']}</b> —
                        <span style='color:{P_WHITE};'>{row['well_id'][:30]}...</span>
                        <span style='color:#C8A8FF; float:right; font-size:12px;'>
                            FTHP={row['FTHP']} | CHP={row['CHP']} | Score={row['score']:.3f}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
