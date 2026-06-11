import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import logging
import warnings
import os
import requests
import io
from datetime import datetime
warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)

st.set_page_config(
    page_title="VIO — Leak & Integrity Detection",
    page_icon="🛢",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=Space+Grotesk:wght@400;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"], .stApp {{
    background: linear-gradient(160deg, {P_DARK} 0%, #1A0535 50%, {P_DARK} 100%) !important;
    color: {P_WHITE}; font-family: 'Inter', sans-serif;
}}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0D011A 0%, {P_DARK} 60%, #0D011A 100%) !important;
    border-right: 1.5px solid {P_NEON}44;
    box-shadow: 4px 0 32px {P_GLOW};
}}
[data-testid="stSidebar"] * {{ color: {P_WHITE} !important; }}
[data-testid="metric-container"] {{
    background: {P_CARD} !important; border-radius: 12px !important;
    padding: 16px !important; border: 1px solid {P_NEON}33 !important;
    box-shadow: 0 2px 16px {P_GLOW} !important;
}}
[data-testid="metric-container"] label {{
    color: #C8A8FF !important; font-size: 11px !important;
    letter-spacing: 1px !important; text-transform: uppercase !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {P_WHITE} !important; font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
}}
.vio-header {{
    background: linear-gradient(120deg, {P_DARK} 0%, {P_MID} 60%, {P_DARK} 100%);
    padding: 22px 30px; border-bottom: 2px solid {P_NEON}88;
    margin-bottom: 22px; border-radius: 0 0 12px 12px;
    box-shadow: 0 4px 32px {P_GLOW};
}}
.risk-card {{ border-radius: 16px; padding: 28px; text-align: center; margin: 10px 0; }}
.metric-pill {{
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 12px; font-weight: 600; letter-spacing: 1px; margin: 2px;
}}
.stButton > button {{
    background: linear-gradient(135deg, {P_MID} 0%, {P_NEON} 100%) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; box-shadow: 0 2px 16px {P_GLOW} !important;
}}
.stButton > button:hover {{
    box-shadow: 0 4px 28px {P_NEON}66 !important; transform: translateY(-1px) !important;
}}
.stSelectbox > div > div {{
    background: {P_CARD} !important; border: 1px solid {P_NEON}44 !important;
    border-radius: 8px !important;
}}
.stTabs [data-baseweb="tab-list"] {{
    background: {P_CARD} !important; border-radius: 10px !important; padding: 4px !important;
}}
.stTabs [data-baseweb="tab"] {{ color: #C8A8FF !important; }}
.stTabs [aria-selected="true"] {{ background: {P_NEON}33 !important; color: white !important; }}
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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(filename):
    return os.path.join(BASE_DIR, filename)

# ── Google Drive file ID ──────────────────────────────────────────────────────
GDRIVE_FILE_ID = "13iYJfC6kEFutN5nGGr0MlrQ_V2LQOLF2"

@st.cache_data(ttl=3600, show_spinner="Loading fleet telemetry from Google Drive...")
def load_data():
    """Download parquet from Google Drive and load into DataFrame."""
    # Direct download URL for Google Drive
    url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}&confirm=t"

    try:
        session = requests.Session()
        response = session.get(url, stream=True, timeout=120)

        # Handle Google's virus-scan warning page for large files
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}&confirm={value}"
                response = session.get(url, stream=True, timeout=120)
                break

        response.raise_for_status()
        content = response.content
        df = pd.read_parquet(io.BytesIO(content))

        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df['Log_Date_Time']):
            df['Log_Date_Time'] = pd.to_datetime(df['Log_Date_Time'])

        # Fill missing sensor columns with 0
        for col in ['FTHP','CHP','ABP','FLT','GIP','GIR','Battery_Voltage',
                    'FTHP_battery_volt','CHP_battery_volt','ABP_battery_volt']:
            if col not in df.columns:
                df[col] = 0.0

        return df

    except Exception as e:
        st.error(f"Failed to load data from Google Drive: {e}")
        st.info("Make sure the Google Drive file is shared as 'Anyone with the link can view'.")
        st.stop()


@st.cache_resource
def load_models():
    m = {}
    try:
        m['classifier'] = joblib.load(get_path("model_leak_classifier.pkl"))
        m['isolation']  = joblib.load(get_path("model_leak_isolation.pkl"))
        m['scaler']     = joblib.load(get_path("scaler_leak.pkl"))
        m['features']   = joblib.load(get_path("features_leak.pkl"))
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

    try:
        from tensorflow.keras.models import load_model
        keras_path = get_path("ae_leak.keras")
        if not os.path.exists(keras_path):
            zip_path = get_path("ae_leak.keras.zip")
            if os.path.exists(zip_path):
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(BASE_DIR)
        m['ae_model']     = load_model(keras_path)
        m['ae_scaler']    = joblib.load(get_path("scaler_ae_leak.pkl"))
        m['ae_threshold'] = joblib.load(get_path("threshold_ae_leak.pkl"))
        m['ae_features']  = joblib.load(get_path("features_ae_leak.pkl"))
        m['ae_loaded']    = True
    except Exception:
        m['ae_loaded'] = False

    return m


def compute_leak_features(well_df):
    df = well_df.sort_values('Log_Date_Time').copy()
    df['CHP_FTHP_diff']     = df['CHP']  - df['FTHP']
    df['CHP_ABP_diff']      = df['CHP']  - df['ABP']
    df['FTHP_ABP_diff']     = df['FTHP'] - df['ABP']
    df['FTHP_mean_6']       = df['FTHP'].rolling(6,  min_periods=1).mean()
    df['FTHP_mean_24']      = df['FTHP'].rolling(24, min_periods=1).mean()
    df['FTHP_std_24']       = df['FTHP'].rolling(24, min_periods=1).std().fillna(0)
    df['CHP_mean_6']        = df['CHP'].rolling(6,   min_periods=1).mean()
    df['CHP_mean_24']       = df['CHP'].rolling(24,  min_periods=1).mean()
    df['CHP_std_24']        = df['CHP'].rolling(24,  min_periods=1).std().fillna(0)
    df['ABP_mean_6']        = df['ABP'].rolling(6,   min_periods=1).mean()
    df['ABP_mean_24']       = df['ABP'].rolling(24,  min_periods=1).mean()
    df['ABP_std_24']        = df['ABP'].rolling(24,  min_periods=1).std().fillna(0)
    df['FTHP_diff']         = df['FTHP_delta'] if 'FTHP_delta' in df.columns else df['FTHP'].diff().fillna(0)
    df['CHP_diff']          = df['CHP_delta']  if 'CHP_delta'  in df.columns else df['CHP'].diff().fillna(0)
    df['ABP_diff']          = df['ABP'].diff().fillna(0)
    df['FTHP_sudden_drop']  = (df['FTHP_diff'] < -5).astype(int)
    df['CHP_sudden_drop']   = (df['CHP_diff']  < -5).astype(int)
    df['ABP_sudden_drop']   = (df['ABP_diff']  < -5).astype(int)
    df['FTHP_decline_rate'] = df['FTHP_mean_6'] - df['FTHP_mean_24']
    df['CHP_decline_rate']  = df['CHP_mean_6']  - df['CHP_mean_24']
    df['scp_indicator']     = (df['CHP'] > df['FTHP'] * 1.5).astype(int)
    df['CHP_trend']         = df['CHP'].diff().rolling(12, min_periods=1).mean().fillna(0)
    df['FTHP_trend']        = df['FTHP'].diff().rolling(12, min_periods=1).mean().fillna(0)
    df['integrity_alert']   = (
        (df['CHP_FTHP_diff'].abs() > df['CHP_FTHP_diff'].abs().rolling(24, min_periods=1).mean() * 2) |
        (df['FTHP_sudden_drop'] == 1) | (df['scp_indicator'] == 1)
    ).astype(int)
    flt_mean              = df['FLT'].mean()
    flt_std               = df['FLT'].std() + 1e-6
    df['FLT_zscore']      = (df['FLT'] - flt_mean) / flt_std
    df['cross_zone_flag'] = (df['FLT_zscore'].abs() > 2.5).astype(int)
    return df


def run_prediction(row_dict, mdls):
    features  = mdls['features']
    scaler    = mdls['scaler']
    X         = np.array([row_dict.get(f, 0) for f in features]).reshape(1, -1)
    X_sc      = scaler.transform(X)
    severity  = mdls['classifier'].predict(X_sc)[0]
    iso_score = float(mdls['isolation'].decision_function(X_sc)[0])
    ae_mse, ae_anom = 0.0, False
    if mdls.get('ae_loaded'):
        try:
            Xa      = np.array([row_dict.get(f, 0) for f in mdls['ae_features']]).reshape(1, -1)
            Xa_sc   = mdls['ae_scaler'].transform(Xa)
            Xa_pred = mdls['ae_model'].predict(Xa_sc, verbose=0)
            ae_mse  = float(np.mean(np.power(Xa_sc - Xa_pred, 2)))
            ae_anom = ae_mse > mdls['ae_threshold']
        except Exception:
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
        "severity"   : severity,
        "iso_score"  : round(iso_score, 6),
        "ae_mse"     : round(ae_mse, 6),
        "ae_anomaly" : ae_anom,
        "leak_score" : round(float(leak_score), 3),
        "scp"        : bool(row_dict.get('scp_indicator', 0)),
        "integrity"  : bool(row_dict.get('integrity_alert', 0)),
        "cross_zone" : bool(row_dict.get('cross_zone_flag', 0)),
        "sudden_drop": bool(row_dict.get('FTHP_sudden_drop', 0)),
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
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor=P_CARD, bordercolor="rgba(168,50,255,0.27)", borderwidth=1),
        xaxis=dict(gridcolor="rgba(168,50,255,0.09)", zerolinecolor="rgba(168,50,255,0.13)"),
        yaxis=dict(gridcolor="rgba(168,50,255,0.09)", zerolinecolor="rgba(168,50,255,0.13)"),
        **kw
    )
    return fig


# ── Load ──────────────────────────────────────────────────────────────────────
mdls = load_models()
df   = load_data()
well_list = sorted(df['well_id'].unique().tolist())

# ── Header ────────────────────────────────────────────────────────────────────
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
        {datetime.now().strftime("%Y-%m-%d %H:%M")} &nbsp;|&nbsp; {len(well_list)} Wells Loaded
    </span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
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

    well_type_filter = st.selectbox("Filter by Well Type", ["All", "Self flow", "Gas lift"])

    if well_type_filter != "All":
        filtered_wells = sorted(df[df['well_type'] == well_type_filter]['well_id'].unique().tolist())
        if not filtered_wells:
            st.warning("No wells for this type.")
            filtered_wells = well_list
    else:
        filtered_wells = well_list

    selected_well = st.selectbox("Select Well", filtered_wells)

    well_dates = df[df['well_id'] == selected_well]['Log_Date_Time']
    min_date   = well_dates.min().date()
    max_date   = well_dates.max().date()
    date_range = st.date_input("Date Range", value=(min_date, max_date),
                                min_value=min_date, max_value=max_date)

    model_status = "Ensemble (AE + IsolationForest)" if mdls and mdls.get('ae_loaded') else "IsolationForest only"
    st.markdown(f"""
    <div style='margin-top:16px; padding:12px; background:{P_DARK}; border-radius:10px;
         border:1px solid {P_NEON}22; font-size:11px; color:#C8A8FF;'>
        <div style='margin-bottom:4px;'>Model: {model_status}</div>
        <div style='margin-bottom:4px;'>Total wells: {len(well_list)}</div>
        <div style='margin-bottom:4px;'>Data rows: {len(df):,}</div>
        <div style='color:{P_NEON}; font-weight:600; letter-spacing:1px;'>LEAK DETECTION ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)

# ── Well data ─────────────────────────────────────────────────────────────────
well_data = df[df['well_id'] == selected_well].copy()

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_dt = pd.Timestamp(date_range[0])
    end_dt   = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    well_data = well_data[(well_data['Log_Date_Time'] >= start_dt) &
                          (well_data['Log_Date_Time'] <  end_dt)]
    if len(well_data) == 0:
        st.warning("No data in selected date range. Showing all dates.")
        well_data = df[df['well_id'] == selected_well].copy()

well_data  = well_data.sort_values('Log_Date_Time').reset_index(drop=True)
well_feats = compute_leak_features(well_data)
latest     = well_feats.iloc[-1]
well_type  = well_data['well_type'].iloc[0]

# ── Top metrics ───────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
with m1: st.metric("FTHP",      f"{latest['FTHP']:.1f} psi")
with m2: st.metric("CHP",       f"{latest['CHP']:.1f} psi")
with m3: st.metric("ABP",       f"{latest['ABP']:.1f} psi")
with m4: st.metric("GIP",       f"{latest.get('GIP', 0):.1f}")
with m5: st.metric("CHP-FTHP",  f"{latest['CHP_FTHP_diff']:.1f} psi")
with m6: st.metric("Well Type", well_type)
with m7: st.metric("SCP Alert", "YES" if latest['CHP'] > latest['FTHP'] * 1.5 else "NO")

st.markdown("<div class='glow-div'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "AI Prediction", "Pressure Analysis", "Integrity Indicators", "Power & GIR", "Fleet Scan"
])

# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_pred, col_info = st.columns(2)
    with col_pred:
        st.markdown("### Run AI Leak Detection")
        if st.button("Analyze Latest Reading", type="primary", use_container_width=True):
            if mdls:
                with st.spinner("Running leak detection model..."):
                    result = run_prediction(latest.to_dict(), mdls)
                sev   = result['severity']
                color = severity_color(sev)
                st.markdown(f"""
                <div class='risk-card' style='background:{P_CARD}; border:2px solid {color};
                     box-shadow:0 0 40px {color}44;'>
                    <div style='font-size:11px; color:#C8A8FF; letter-spacing:2px;
                         text-transform:uppercase; margin-bottom:10px;'>VIO AI — LEAK RISK ASSESSMENT</div>
                    <div style='font-size:56px; font-weight:900; color:{color};
                                font-family:"Space Grotesk",sans-serif;
                                text-shadow:0 0 24px {color}88; line-height:1;'>{sev}</div>
                    <div style='margin-top:16px; display:flex; justify-content:center; gap:10px; flex-wrap:wrap;'>
                        <span class='metric-pill' style='background:{color}22; color:{color}; border:1px solid {color}44;'>
                            Score: {result["leak_score"]:.3f}</span>
                        <span class='metric-pill' style='background:{P_NEON}22; color:{P_NEON}; border:1px solid {P_NEON}44;'>
                            ISO: {result["iso_score"]:.4f}</span>
                        {"<span class='metric-pill' style='background:" + C_YELLOW + "22; color:" + C_YELLOW + "; border:1px solid " + C_YELLOW + "44;'>AE Drift</span>" if result["ae_anomaly"] else ""}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                action_map = {
                    "CRITICAL LEAK": ("error",   "CRITICAL: Shut in well immediately."),
                    "HIGH RISK"    : ("warning",  "HIGH: Notify field engineer within 2 hours."),
                    "MEDIUM RISK"  : ("warning",  "MEDIUM: Inspect within 24 hours."),
                    "LOW RISK"     : ("info",     "LOW: Log and monitor closely."),
                    "NO LEAK"      : ("success",  "No action required. Well integrity confirmed."),
                }
                action_type, action_msg = action_map.get(sev, ("info", "Monitor well"))
                getattr(st, action_type)(action_msg)
                st.markdown("#### Detection Flags")
                f1, f2, f3, f4 = st.columns(4)
                flags = [
                    (f1, result['sudden_drop'], C_RED,    "SUDDEN DROP", "YES",   "NO"),
                    (f2, result['scp'],         C_RED,    "SCP ALERT",   "YES",   "NO"),
                    (f3, result['integrity'],   C_ORANGE, "INTEGRITY",   "ALERT", "OK"),
                    (f4, result['cross_zone'],  C_YELLOW, "CROSS ZONE",  "FLAG",  "OK"),
                ]
                for col, val, ac, label, y, n in flags:
                    c = ac if val else C_GREEN
                    with col:
                        st.markdown(f"""
                        <div style='text-align:center; padding:10px; background:{P_CARD};
                             border-radius:8px; border:1px solid {c}44;'>
                            <div style='color:{c}; font-weight:700; font-size:11px;'>{label}</div>
                            <div style='font-size:20px; color:{c};'>{y if val else n}</div>
                        </div>""", unsafe_allow_html=True)
            else:
                st.error("Models not loaded. Check that all .pkl files are in the repo root.")

    with col_info:
        st.markdown("### Latest Sensor Snapshot")
        sensor_rows = [
            ("FTHP",          f"{latest['FTHP']:.2f} psi"),
            ("CHP",           f"{latest['CHP']:.2f} psi"),
            ("ABP",           f"{latest['ABP']:.2f} psi"),
            ("GIP",           f"{latest.get('GIP', 0):.2f}"),
            ("FLT",           f"{latest.get('FLT', 0):.2f} °C"),
            ("GIR",           f"{latest.get('GIR', 0):.2f}"),
            ("Battery V",     f"{latest.get('Battery_Voltage', 0):.2f} V"),
            ("FTHP Batt V",   f"{latest.get('FTHP_battery_volt', 0):.2f} V"),
            ("CHP Batt V",    f"{latest.get('CHP_battery_volt', 0):.2f} V"),
            ("ABP Batt V",    f"{latest.get('ABP_battery_volt', 0):.2f} V"),
            ("CHP-FTHP Diff", f"{latest['CHP_FTHP_diff']:.2f} psi"),
            ("FTHP-ABP Diff", f"{latest['FTHP_ABP_diff']:.2f} psi"),
            ("GIR scmd",      f"{latest.get('gir_scmd', 0):.2f}" if pd.notna(latest.get('gir_scmd')) else "N/A"),
            ("GIR mmscfd",    f"{latest.get('gir_mmscfd', 0):.4f}" if pd.notna(latest.get('gir_mmscfd')) else "N/A"),
            ("Log Time",      str(latest['Log_Date_Time'])[:19]),
        ]
        sensors = pd.DataFrame(sensor_rows, columns=["Sensor", "Value"])
        st.dataframe(sensors, use_container_width=True, hide_index=True)
        st.markdown("### Risk Guide")
        for label, color, desc in [
            ("NO LEAK",       C_GREEN,   "Well integrity confirmed"),
            ("LOW RISK",      "#88FF88", "Minor anomaly — log & monitor"),
            ("MEDIUM RISK",   C_YELLOW,  "Inspect within 24 hours"),
            ("HIGH RISK",     C_ORANGE,  "Field engineer within 2 hours"),
            ("CRITICAL LEAK", C_RED,     "Shut-in well immediately"),
        ]:
            st.markdown(f"""
            <div style='display:flex; align-items:center; padding:8px 12px; margin:4px 0;
                 background:{P_CARD}; border-radius:8px; border-left:3px solid {color};'>
                <span style='color:{color}; font-weight:700; font-size:12px; min-width:130px;'>{label}</span>
                <span style='color:#C8A8FF; font-size:11px;'>{desc}</span>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    recent = well_feats.tail(300).copy()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Casing-Tubing Pressure Differential")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent['Log_Date_Time'], y=recent['CHP_FTHP_diff'],
            mode='lines', fill='tozeroy', fillcolor='rgba(255,23,68,0.08)',
            line=dict(color=C_RED, width=1.5), name='CHP-FTHP Diff'
        ))
        fig.add_hline(y=0, line_color=P_WHITE, line_dash='dash', line_width=1)
        plotly_dark(fig, height=320, yaxis_title="Differential (psi)", title="Casing vs Tubing Pressure")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### FTHP Sudden Drop Events")
        drop_events = recent[recent['FTHP_sudden_drop'] == 1]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=recent['Log_Date_Time'], y=recent['FTHP'],
            mode='lines', line=dict(color=P_NEON, width=1.5), name='FTHP'))
        fig2.add_trace(go.Scatter(x=recent['Log_Date_Time'], y=recent['CHP'],
            mode='lines', line=dict(color=C_ORANGE, width=1.5), name='CHP'))
        if len(drop_events) > 0:
            fig2.add_trace(go.Scatter(
                x=drop_events['Log_Date_Time'], y=drop_events['FTHP'], mode='markers',
                marker=dict(color=C_RED, size=10, symbol='x', line=dict(width=2, color=C_RED)),
                name='Sudden Drop'
            ))
        plotly_dark(fig2, height=320, yaxis_title="Pressure (psi)", title="FTHP & CHP with Drop Events")
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("#### Multi-Pressure Timeline (FTHP / CHP / ABP / GIP)")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=recent['Log_Date_Time'], y=recent['FTHP'], mode='lines', line=dict(color=P_NEON,   width=1.5), name='FTHP'))
    fig3.add_trace(go.Scatter(x=recent['Log_Date_Time'], y=recent['CHP'],  mode='lines', line=dict(color=C_ORANGE, width=1.5), name='CHP'))
    fig3.add_trace(go.Scatter(x=recent['Log_Date_Time'], y=recent['ABP'],  mode='lines', line=dict(color=C_GREEN,  width=1.5), name='ABP'))
    if 'GIP' in recent.columns:
        fig3.add_trace(go.Scatter(x=recent['Log_Date_Time'], y=recent['GIP'], mode='lines', line=dict(color=C_YELLOW, width=1.5), name='GIP'))
    plotly_dark(fig3, height=280, yaxis_title="Pressure (psi)", title="All Pressures — Last 300 Readings")
    st.plotly_chart(fig3, use_container_width=True)
    d1, d2, d3, d4, d5 = st.columns(5)
    with d1: st.metric("Avg FTHP",    f"{recent['FTHP'].mean():.1f} psi")
    with d2: st.metric("Avg CHP",     f"{recent['CHP'].mean():.1f} psi")
    with d3: st.metric("Avg ABP",     f"{recent['ABP'].mean():.1f} psi")
    with d4: st.metric("Avg GIP",     f"{recent['GIP'].mean():.1f}" if 'GIP' in recent.columns else "N/A")
    with d5: st.metric("Drop Events", int(recent['FTHP_sudden_drop'].sum()))

# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    recent     = well_feats.tail(300).copy()
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
         padding:20px; text-align:center; margin-bottom:20px; box-shadow:0 0 32px {ov_color}33;'>
        <div style='font-size:11px; color:#C8A8FF; letter-spacing:2px; text-transform:uppercase;'>OVERALL INTEGRITY RISK</div>
        <div style='font-size:44px; font-weight:900; color:{ov_color}; font-family:"Space Grotesk",sans-serif;
                    text-shadow:0 0 20px {ov_color}66;'>{overall}</div>
        <div style='font-size:12px; color:#8877AA; margin-top:6px;'>Based on last 300 readings</div>
    </div>
    """, unsafe_allow_html=True)
    i1, i2, i3, i4 = st.columns(4)
    with i1: st.metric("SCP Events",       scp_count)
    with i2: st.metric("Sudden Drops",     drop_count)
    with i3: st.metric("Cross-Zone Flags", cross_zone)
    with i4: st.metric("Integrity Alerts", integrity)
    st.markdown("<div class='glow-div'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### SCP Indicator Timeline")
        scp_events = recent[recent['scp_indicator'] == 1]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=recent['Log_Date_Time'], y=recent['CHP_FTHP_diff'],
            mode='lines', line=dict(color=P_NEON, width=1.5), name='CHP-FTHP'))
        if len(scp_events) > 0:
            fig4.add_trace(go.Scatter(x=scp_events['Log_Date_Time'], y=scp_events['CHP_FTHP_diff'],
                mode='markers', marker=dict(color=C_RED, size=8, symbol='circle'), name='SCP Event'))
        plotly_dark(fig4, height=280, yaxis_title="Differential (psi)")
        st.plotly_chart(fig4, use_container_width=True)
    with c2:
        st.markdown("#### FLT Z-Score (Cross-Zone Indicator)")
        cross_events = recent[recent['cross_zone_flag'] == 1]
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=recent['Log_Date_Time'], y=recent['FLT_zscore'],
            mode='lines', fill='tozeroy', fillcolor='rgba(168,50,255,0.07)',
            line=dict(color=P_NEON, width=1.5), name='FLT Z-Score'))
        fig5.add_hline(y= 2.5, line_color=C_RED, line_dash='dash', annotation_text="+2.5 threshold")
        fig5.add_hline(y=-2.5, line_color=C_RED, line_dash='dash', annotation_text="-2.5 threshold")
        if len(cross_events) > 0:
            fig5.add_trace(go.Scatter(x=cross_events['Log_Date_Time'], y=cross_events['FLT_zscore'],
                mode='markers', marker=dict(color=C_YELLOW, size=8), name='Cross-Zone Flag'))
        plotly_dark(fig5, height=280, yaxis_title="Z-Score")
        st.plotly_chart(fig5, use_container_width=True)
    st.markdown("#### Integrity Events Log (last 20)")
    alerts_df = recent[
        (recent['scp_indicator']==1) | (recent['FTHP_sudden_drop']==1) |
        (recent['cross_zone_flag']==1) | (recent['integrity_alert']==1)
    ][['Log_Date_Time','FTHP','CHP','ABP','FLT',
       'scp_indicator','FTHP_sudden_drop','cross_zone_flag','integrity_alert']].tail(20)
    if len(alerts_df) > 0:
        st.dataframe(alerts_df, use_container_width=True, hide_index=True)
    else:
        st.success("No integrity events detected in last 300 readings.")

# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    recent = well_feats.tail(300).copy()
    st.markdown("### Power & Gas Injection Rate")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("#### Active Power & Frequency")
        fig_p = go.Figure()
        if 'output_active_Power' in recent.columns:
            fig_p.add_trace(go.Scatter(x=recent['Log_Date_Time'], y=recent['output_active_Power'],
                mode='lines', line=dict(color=P_NEON, width=1.5), name='Active Power'))
        if 'output_System_Frequency' in recent.columns:
            fig_p.add_trace(go.Scatter(x=recent['Log_Date_Time'], y=recent['output_System_Frequency'],
                mode='lines', line=dict(color=C_YELLOW, width=1.5), name='Frequency', yaxis='y2'))
        fig_p.update_layout(yaxis2=dict(overlaying='y', side='right', gridcolor='rgba(0,0,0,0)'))
        plotly_dark(fig_p, height=280, title="Power & Frequency")
        st.plotly_chart(fig_p, use_container_width=True)
    with p2:
        st.markdown("#### Voltage & Current")
        fig_v = go.Figure()
        if 'output_Average_Voltage_L2N' in recent.columns:
            fig_v.add_trace(go.Scatter(x=recent['Log_Date_Time'], y=recent['output_Average_Voltage_L2N'],
                mode='lines', line=dict(color=C_GREEN, width=1.5), name='Avg Voltage L2N'))
        if 'output_Average_Current' in recent.columns:
            fig_v.add_trace(go.Scatter(x=recent['Log_Date_Time'], y=recent['output_Average_Current'],
                mode='lines', line=dict(color=C_ORANGE, width=1.5), name='Avg Current', yaxis='y2'))
        fig_v.update_layout(yaxis2=dict(overlaying='y', side='right', gridcolor='rgba(0,0,0,0)'))
        plotly_dark(fig_v, height=280, title="Voltage & Current")
        st.plotly_chart(fig_v, use_container_width=True)
    st.markdown("<div class='glow-div'></div>", unsafe_allow_html=True)
    st.markdown("### Gas Injection Rate (GIR)")
    fig_g = go.Figure()
    gir_series = {'GIR': P_NEON, 'gir_scmd': C_GREEN, 'gir_sm3_hr': C_YELLOW, 'gir_mmscfd': C_ORANGE}
    for col, color in gir_series.items():
        if col in recent.columns:
            valid = recent[recent[col].notna()]
            if len(valid) > 0:
                fig_g.add_trace(go.Scatter(x=valid['Log_Date_Time'], y=valid[col],
                    mode='lines', line=dict(color=color, width=1.5), name=col))
    plotly_dark(fig_g, height=260, title="GIR Metrics Over Time")
    st.plotly_chart(fig_g, use_container_width=True)
    g1, g2, g3, g4 = st.columns(4)
    with g1: st.metric("Avg GIR",       f"{recent['GIR'].mean():.2f}"        if 'GIR'        in recent.columns else "N/A")
    with g2: st.metric("Avg GIR scmd",  f"{recent['gir_scmd'].mean():.2f}"   if 'gir_scmd'   in recent.columns else "N/A")
    with g3: st.metric("Avg sm3/hr",    f"{recent['gir_sm3_hr'].mean():.2f}" if 'gir_sm3_hr' in recent.columns else "N/A")
    with g4: st.metric("Avg mmscfd",    f"{recent['gir_mmscfd'].mean():.4f}" if 'gir_mmscfd' in recent.columns else "N/A")

# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Fleet-Wide Leak Risk Scan")
    st.info(f"Scans the latest reading for every well — {len(well_list)} wells total.")
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        scan_type = st.selectbox("Well Type to Scan", ["All", "Self flow", "Gas lift"])
    with col_opt2:
        max_wells = st.slider("Max Wells to Scan", 10, len(well_list), min(100, len(well_list)), 10)
    if st.button("Run Fleet Scan", type="primary", use_container_width=True):
        if not mdls:
            st.error("Models not loaded.")
        else:
            scan_wells = well_list
            if scan_type != "All":
                scan_wells = [w for w in well_list if
                              df[df['well_id']==w]['well_type'].iloc[0] == scan_type]
            scan_wells = scan_wells[:max_wells]
            fleet_results = []
            progress    = st.progress(0)
            status_text = st.empty()
            for i, wid in enumerate(scan_wells):
                status_text.text(f"Scanning {wid} ({i+1}/{len(scan_wells)})...")
                wd   = df[df['well_id'] == wid].copy()
                wf   = compute_leak_features(wd)
                last = wf.iloc[-1]
                res  = run_prediction(last.to_dict(), mdls)
                fleet_results.append({
                    "well_id"  : wid,
                    "well_type": wd['well_type'].iloc[0],
                    "FTHP"     : round(float(last['FTHP']), 1),
                    "CHP"      : round(float(last['CHP']),  1),
                    "ABP"      : round(float(last['ABP']),  1),
                    "GIP"      : round(float(last.get('GIP', 0)), 1),
                    "severity" : res['severity'],
                    "score"    : res['leak_score'],
                    "scp"      : int(res['scp']),
                    "ae_drift" : int(res['ae_anomaly']),
                    "last_seen": str(last['Log_Date_Time'])[:19],
                })
                progress.progress((i+1)/len(scan_wells))
            status_text.empty()
            results_df = pd.DataFrame(fleet_results)
            fc1, fc2, fc3, fc4, fc5 = st.columns(5)
            with fc1: st.metric("Wells Scanned", len(results_df))
            with fc2: st.metric("No Leak",       (results_df['severity']=="NO LEAK").sum())
            with fc3: st.metric("Low / Medium",  results_df['severity'].isin(["LOW RISK","MEDIUM RISK"]).sum())
            with fc4: st.metric("High Risk",     (results_df['severity']=="HIGH RISK").sum())
            with fc5: st.metric("Critical",      (results_df['severity']=="CRITICAL LEAK").sum())
            sev_order  = ["NO LEAK","LOW RISK","MEDIUM RISK","HIGH RISK","CRITICAL LEAK"]
            sev_cols   = [C_GREEN, "#88FF88", C_YELLOW, C_ORANGE, C_RED]
            sev_counts = results_df['severity'].value_counts().reindex(sev_order, fill_value=0)
            fig6 = go.Figure(go.Bar(
                x=sev_order, y=sev_counts.values,
                marker_color=sev_cols, marker_line_color=P_DARK, marker_line_width=1.5
            ))
            plotly_dark(fig6, height=260, title="Fleet Leak Risk Distribution", yaxis_title="Wells")
            st.plotly_chart(fig6, use_container_width=True)
            def color_severity_cell(val):
                return {
                    'CRITICAL LEAK': f'background-color:#2A0015; color:{C_RED}',
                    'HIGH RISK'    : f'background-color:#2A0F00; color:{C_ORANGE}',
                    'MEDIUM RISK'  : f'background-color:#2A2200; color:{C_YELLOW}',
                    'LOW RISK'     : f'background-color:#0A1A0A; color:{C_GREEN}',
                    'NO LEAK'      : f'background-color:#0D0020; color:{P_NEON}',
                }.get(val, '')
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
                        <span style='color:{P_WHITE};'>{row['well_id']}</span>
                        <span style='color:#8877AA; font-size:11px;'> [{row['well_type']}]</span>
                        <span style='color:#C8A8FF; float:right; font-size:12px;'>
                            FTHP={row['FTHP']} | CHP={row['CHP']} | Score={row['score']:.3f} | {row['last_seen']}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
