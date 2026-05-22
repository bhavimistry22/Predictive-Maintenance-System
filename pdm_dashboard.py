"""
╔══════════════════════════════════════════════════════════════╗
║   FOUNDRY MACHINE — PREDICTIVE MAINTENANCE DASHBOARD         ║
║   Streamlit App  |  Uses saved models from pm_models/        ║
╚══════════════════════════════════════════════════════════════╝

Run:
    streamlit run pdm_dashboard.py

Requirements:
    pip install streamlit pandas numpy scikit-learn xgboost plotly shap pickle5
"""

import warnings
warnings.filterwarnings("ignore")

import os, json, pickle, datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Page Config (MUST be first Streamlit call) ───────────────
st.set_page_config(
    page_title="FM Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Theme / CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Rajdhani:wght@400;600;700&display=swap');

/* Root */
:root {
    --bg:       #0a0d14;
    --card:     #111520;
    --border:   #1e2535;
    --normal:   #00e5a0;
    --warning:  #ffb74d;
    --critical: #ff4757;
    --accent:   #4fc3f7;
    --purple:   #ce93d8;
    --text:     #e0e4f0;
    --muted:    #6b7494;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display:none;}
.block-container {padding: 1rem 2rem 2rem 2rem !important;}

/* Background */
.stApp { background: var(--bg); }

/* All text */
body, .stMarkdown, .stText, label, .stSelectbox, .stNumberInput {
    font-family: 'Rajdhani', sans-serif !important;
    color: var(--text) !important;
}

/* Title bar */
.dash-title {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 2rem;
    letter-spacing: 0.12em;
    color: var(--accent);
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}

/* KPI cards */
.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height:2px;
}
.kpi-card.normal::before  { background: var(--normal); }
.kpi-card.warning::before { background: var(--warning); }
.kpi-card.critical::before{ background: var(--critical); }
.kpi-card.accent::before  { background: var(--accent); }
.kpi-card.purple::before  { background: var(--purple); }

.kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.kpi-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.kpi-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    margin-top: 0.2rem;
}

/* Alert badge */
.alert-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-critical { background: rgba(255,71,87,0.15); border:1px solid var(--critical); color:var(--critical);}
.badge-warning  { background: rgba(255,183,77,0.15); border:1px solid var(--warning); color:var(--warning);}
.badge-normal   { background: rgba(0,229,160,0.15);  border:1px solid var(--normal);  color:var(--normal);}

/* Action list */
.action-item {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.95rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid var(--border);
    color: var(--text);
}
.action-item:last-child { border-bottom: none; }

/* Section headers */
.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.8rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: var(--card); border-radius:6px; }
.stTabs [data-baseweb="tab"]      { font-family:'Rajdhani',sans-serif; font-weight:600; color:var(--muted); }
.stTabs [aria-selected="true"]    { color:var(--accent) !important; }

/* Slider, number input */
.stSlider, .stNumberInput { color: var(--accent) !important; }

/* DataFrame */
.stDataFrame { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
MODEL_DIR = "pm_models"

ANOM_FEATURES = [
    'FM_Total_Cycle_TIme', 'Eco_MP_Pressure', 'Eco_FM_Actual_Oil_Temp',
    'Eco_V9_Actual_Pressure', 'Drag_Sand_Weight', 'Cope_Sand_Weight',
    'fault_rate_roll10', 'FM_Total_Cycle_TIme_roll_std_15',
    'Eco_MP_Pressure_roc_1', 'cope_drag_time_diff'
]

SENSOR_COLS = [
    'FM_Total_Cycle_TIme', 'Life_Cycle_Counter', 'Shift_Cycle_Counter',
    'Cope_Cycle_Time', 'Drag_Cycle_Time', 'Eco_MP_Pressure',
    'Eco_FM_Actual_Oil_Temp', 'V9_Cylinder_Up', 'V9_Cylinder_Down',
    'V10_Cylinder_in', 'V10_Cylinder_out', 'Eco_V9_Actual_Pressure',
    'Cycle_No', 'Drag_Sand_Weight', 'Drag_Seq_Pressure', 'Cope_Sand_Weight',
]

FAULT_COLS_CRITICAL = [
    'FM Emergency Stop', 'FM Remote Emergency Stop',
    'FM HPU Oil Level Low', 'FM Air Pressure Low Fault',
    'FM Hyd Powerpack Trip', 'FM Pressure Filter Chokup',
    'FM Offline Filter Trip', 'FM HPU Oil Temperature High',
    'Right Arm Not Lock Position', 'Left Arm Not Lock Position',
]

MAINTENANCE_RULES = {
    'Hydraulic': {
        'priority': 'HIGH',
        'actions': [
            'Check hydraulic oil level immediately',
            'Inspect HPU for leaks or pressure drop',
            'Verify hydraulic filter condition — replace if clogged',
            'Monitor oil temperature — check cooler if above 65°C',
            'Test HPU relief valve setting',
        ],
        'downtime_est': '2–4 hours', 'cost_est': '₹15,000–₹40,000', 'urgency_hours': 4,
    },
    'Pneumatic': {
        'priority': 'HIGH',
        'actions': [
            'Check air supply pressure (target: 6–7 bar)',
            'Inspect pneumatic fittings for leaks',
            'Clean/replace air filters and lubricators',
            'Verify solenoid valve operation (PS1/PS1-C)',
        ],
        'downtime_est': '1–2 hours', 'cost_est': '₹5,000–₹15,000', 'urgency_hours': 6,
    },
    'Mechanical': {
        'priority': 'MEDIUM',
        'actions': [
            'Inspect V9/V10 cylinder seals and guides',
            'Check squeeze pressure settings',
            'Verify stripper plate alignment',
            'Lubricate moving parts per schedule',
            'Check sand distribution uniformity',
        ],
        'downtime_est': '3–6 hours', 'cost_est': '₹20,000–₹60,000', 'urgency_hours': 12,
    },
    'Electrical': {
        'priority': 'MEDIUM',
        'actions': [
            'Check sensor wiring for shorts / loose connections',
            'Test proximity switches (V9, V10, V11)',
            'Inspect PLC input cards',
            'Verify control panel power supply',
        ],
        'downtime_est': '1–3 hours', 'cost_est': '₹8,000–₹25,000', 'urgency_hours': 8,
    },
    'Safety_Stop': {
        'priority': 'CRITICAL',
        'actions': [
            '⚠️ MACHINE STOPPED — identify cause before restart',
            'Check E-stop circuits and reset conditions',
            'Verify area is clear of personnel',
            'Inspect for mechanical obstruction',
            'Review PLC fault log',
        ],
        'downtime_est': '30 min – 2 hours', 'cost_est': '₹0 (safety hold)', 'urgency_hours': 0,
    },
    'Lubrication': {
        'priority': 'LOW',
        'actions': [
            'Refill lubrication oil tank',
            'Check lubrication motor and pump',
            'Inspect lubrication lines for blockage',
            'Verify lubrication cycle frequency settings',
        ],
        'downtime_est': '30 min – 1 hour', 'cost_est': '₹2,000–₹8,000', 'urgency_hours': 24,
    },
    'Normal': {
        'priority': 'NONE',
        'actions': ['Machine operating normally — continue monitoring'],
        'downtime_est': '0', 'cost_est': '₹0', 'urgency_hours': 999,
    },
}

COLORS = {
    'normal':   '#00e5a0',
    'warning':  '#ffb74d',
    'critical': '#ff4757',
    'accent':   '#4fc3f7',
    'purple':   '#ce93d8',
    'bg':       '#0a0d14',
    'card':     '#111520',
    'border':   '#1e2535',
    'text':     '#e0e4f0',
    'muted':    '#6b7494',
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,21,32,0.6)',
    font=dict(family='Rajdhani, sans-serif', color=COLORS['text'], size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor=COLORS['border'], showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=COLORS['border'], showgrid=True, zeroline=False),
)


# ══════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Load all saved models. Returns None tuple if not found."""
    try:
        def _load(path):
            with open(path, 'rb') as f:
                return pickle.load(f)

        fail_m  = _load(f"{MODEL_DIR}/failure_model.pkl")
        fault_m = _load(f"{MODEL_DIR}/fault_clf.pkl")
        rul_m   = _load(f"{MODEL_DIR}/rul_model.pkl")
        anom_m  = _load(f"{MODEL_DIR}/anomaly_model.pkl")
        sc      = _load(f"{MODEL_DIR}/scaler.pkl")
        le_f    = _load(f"{MODEL_DIR}/label_enc_fault.pkl")
        le_c    = _load(f"{MODEL_DIR}/label_enc_cond.pkl")
        sc_anom = _load(f"{MODEL_DIR}/scaler_anom.pkl") if os.path.exists(f"{MODEL_DIR}/scaler_anom.pkl") else None

        with open(f"{MODEL_DIR}/selected_features.json") as jf:
            feats = json.load(jf)

        return fail_m, fault_m, rul_m, anom_m, sc, le_f, le_c, feats, sc_anom
    except FileNotFoundError as e:
        return None, None, None, None, None, None, None, None, None


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (mirrors notebook Section 4)
# ══════════════════════════════════════════════════════════════
def engineer_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same feature engineering as notebook on a DataFrame."""
    df = df.copy().sort_values('Cycle_No').reset_index(drop=True)

    ROLLING_SENSORS = [c for c in [
        'FM_Total_Cycle_TIme', 'Eco_MP_Pressure', 'Eco_FM_Actual_Oil_Temp',
        'Eco_V9_Actual_Pressure', 'Drag_Sand_Weight', 'Cope_Sand_Weight', 'Drag_Seq_Pressure'
    ] if c in df.columns]

    WINDOWS = [5, 15, 30]

    for sensor in ROLLING_SENSORS:
        for w in WINDOWS:
            df[f'{sensor}_roll_mean_{w}'] = df[sensor].rolling(w, min_periods=1).mean()
            df[f'{sensor}_roll_std_{w}']  = df[sensor].rolling(w, min_periods=1).std().fillna(0)
            df[f'{sensor}_roll_max_{w}']  = df[sensor].rolling(w, min_periods=1).max()
            df[f'{sensor}_roll_min_{w}']  = df[sensor].rolling(w, min_periods=1).min()

    LAG_SENSORS = [c for c in ['FM_Total_Cycle_TIme', 'Eco_MP_Pressure', 'Eco_FM_Actual_Oil_Temp'] if c in df.columns]
    for sensor in LAG_SENSORS:
        for lag in [1, 3, 5, 10]:
            df[f'{sensor}_lag_{lag}'] = df[sensor].shift(lag).bfill()

    for sensor in ROLLING_SENSORS:
        df[f'{sensor}_roc_1']     = df[sensor].diff(1).fillna(0)
        df[f'{sensor}_roc_5']     = df[sensor].diff(5).fillna(0)
        df[f'{sensor}_roc_accel'] = df[f'{sensor}_roc_1'].diff(1).fillna(0)

    if 'Eco_MP_Pressure' in df.columns and 'Eco_FM_Actual_Oil_Temp' in df.columns:
        df['pressure_temp_ratio'] = df['Eco_MP_Pressure'] / (df['Eco_FM_Actual_Oil_Temp'] + 1)
    if 'Cope_Cycle_Time' in df.columns and 'Drag_Cycle_Time' in df.columns:
        df['cope_drag_time_diff'] = df['Cope_Cycle_Time'] - df['Drag_Cycle_Time']
    if 'Drag_Sand_Weight' in df.columns and 'Cope_Sand_Weight' in df.columns:
        df['sand_weight_balance'] = df['Drag_Sand_Weight'] - df['Cope_Sand_Weight']
    if 'V9_Cylinder_Up' in df.columns:
        df['v9_position_sum']  = df['V9_Cylinder_Up'] + df['V9_Cylinder_Down']
    if 'V10_Cylinder_in' in df.columns:
        df['v10_position_sum'] = df['V10_Cylinder_in'] + df['V10_Cylinder_out']
    if 'FM_Total_Cycle_TIme' in df.columns and 'Life_Cycle_Counter' in df.columns:
        df['cycle_time_normalized'] = df['FM_Total_Cycle_TIme'] / (df['Life_Cycle_Counter'] + 1)

    # Fault history features (zero if no alarm cols present)
    df['fault_rate_roll10']  = 0.0
    df['total_fault_count']  = 0.0

    return df


def health_index_single(row: dict) -> float:
    """Compute health index proxy for a single new reading."""
    HEALTH_PARAMS = {
        'FM_Total_Cycle_TIme':    {'ideal': 710,  'warn': 1000, 'crit': 2000},
        'Eco_MP_Pressure':        {'ideal': 60,   'warn': 45,   'crit': 35},
        'Eco_FM_Actual_Oil_Temp': {'ideal': 45,   'warn': 60,   'crit': 70},
        'Eco_V9_Actual_Pressure': {'ideal': 60,   'warn': 50,   'crit': 40},
    }
    scores, weights = [], []
    for param, p in HEALTH_PARAMS.items():
        if param not in row:
            continue
        val = row[param]
        if p['ideal'] >= p['warn']:
            if val >= p['ideal']:     s = 100.0
            elif val >= p['warn']:    s = 60 + 40*(val-p['warn'])/(p['ideal']-p['warn'])
            elif val >= p['crit']:    s = 20 + 40*(val-p['crit'])/(p['warn']-p['crit'])
            else:                     s = max(0, 20*val/(p['crit']+1e-9))
        else:
            if val <= p['ideal']:     s = 100.0
            elif val <= p['warn']:    s = 60 + 40*(p['warn']-val)/(p['warn']-p['ideal']+1e-9)
            elif val <= p['crit']:    s = 20 + 40*(p['crit']-val)/(p['crit']-p['warn']+1e-9)
            else:                     s = max(0.0, 20-20*(val-p['crit'])/(p['crit']+1e-9))
        scores.append(s)
        weights.append(1.0)
    if not scores:
        return 75.0
    return float(np.average(scores, weights=weights))


# ══════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════
def generate_alert(cycle_no, health_idx, fault_type, fail_prob, rul_val, anomaly):
    if health_idx >= 70:   condition = 'NORMAL'
    elif health_idx >= 40: condition = 'WARNING'
    else:                  condition = 'CRITICAL'

    if fail_prob > 0.8 or condition == 'CRITICAL' or fault_type == 'Safety_Stop':
        alert_level = 'CRITICAL'
    elif fail_prob > 0.5 or condition == 'WARNING' or anomaly:
        alert_level = 'WARNING'
    else:
        alert_level = 'NORMAL'

    rules = MAINTENANCE_RULES.get(fault_type, MAINTENANCE_RULES['Normal'])
    return {
        'cycle_no':      cycle_no,
        'timestamp':     datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'alert_level':   alert_level,
        'machine_cond':  condition,
        'health_index':  round(health_idx, 1),
        'fail_prob':     round(fail_prob * 100, 1),
        'rul_cycles':    max(0, int(rul_val)),
        'fault_type':    fault_type,
        'priority':      rules['priority'],
        'actions':       rules['actions'],
        'downtime_est':  rules['downtime_est'],
        'cost_est':      rules['cost_est'],
        'urgency_hours': rules['urgency_hours'],
        'anomaly':       anomaly,
    }


def predict_single(sensor_dict, models):
    """Predict on one sensor reading dict."""
    fail_m, fault_m, rul_m, anom_m, sc, le_f, le_c, feats, sc_anom = models

    new_df = pd.DataFrame([sensor_dict])
    cope  = sensor_dict.get('Cope_Cycle_Time', 700)
    drag  = sensor_dict.get('Drag_Cycle_Time', 700)
    press = sensor_dict.get('Eco_MP_Pressure', 60)
    temp  = sensor_dict.get('Eco_FM_Actual_Oil_Temp', 45)
    dsand = sensor_dict.get('Drag_Sand_Weight', 75)
    csand = sensor_dict.get('Cope_Sand_Weight', 75)

    new_df['fault_rate_roll10']               = 0.0
    new_df['total_fault_count']               = 0.0
    new_df['cope_drag_time_diff']             = cope - drag
    new_df['sand_weight_balance']             = dsand - csand
    new_df['pressure_temp_ratio']             = press / (temp + 1)
    new_df['FM_Total_Cycle_TIme_roll_std_15'] = 0.0
    new_df['Eco_MP_Pressure_roc_1']           = 0.0
    for f in feats:
        if f not in new_df.columns:
            new_df[f] = 0.0

    X_new     = sc.transform(new_df[feats].fillna(0).values)
    fail_prob = float(fail_m.predict_proba(X_new)[0, 1])
    fault_enc = int(fault_m.predict(X_new)[0])
    fault_lbl = le_f.inverse_transform([fault_enc])[0]
    rul_val   = max(0.0, float(rul_m.predict(X_new)[0]))

    # Anomaly — use scaler_anom if saved, otherwise use sc on anom subset
    anom_row = {f: new_df[f].values[0] if f in new_df.columns else sensor_dict.get(f, 0.0)
                for f in ANOM_FEATURES}
    anom_df_row = pd.DataFrame([anom_row]).fillna(0)
    if sc_anom is not None:
        X_anom = sc_anom.transform(anom_df_row.values)
    else:
        anom_cols_present = [c for c in ANOM_FEATURES if c in new_df.columns]
        X_anom_vals = anom_df_row[[c for c in ANOM_FEATURES]].values
        # fallback: crude normalise
        X_anom = (X_anom_vals - X_anom_vals.mean()) / (X_anom_vals.std() + 1e-9)

    try:
        anom_flag = bool(anom_m.predict(X_anom)[0] == -1)
        anom_score = float(-anom_m.score_samples(X_anom)[0])
    except Exception:
        anom_flag  = False
        anom_score = 0.0

    hi = health_index_single(sensor_dict)
    return generate_alert(
        cycle_no   = int(sensor_dict.get('Cycle_No', 0)),
        health_idx = hi,
        fault_type = fault_lbl,
        fail_prob  = fail_prob,
        rul_val    = rul_val,
        anomaly    = anom_flag,
    ), anom_score, X_new


def predict_dataset(df_raw: pd.DataFrame, models):
    """Run predictions on an entire uploaded dataset."""
    fail_m, fault_m, rul_m, anom_m, sc, le_f, le_c, feats, sc_anom = models

    df = engineer_features_df(df_raw)

    for f in feats:
        if f not in df.columns:
            df[f] = 0.0

    X_all = sc.transform(df[feats].fillna(0).values)

    fail_probs  = fail_m.predict_proba(X_all)[:, 1]
    fault_preds = le_f.inverse_transform(fault_m.predict(X_all))
    rul_preds   = rul_m.predict(X_all).clip(0)

    # Anomaly
    anom_feats_present = [f for f in ANOM_FEATURES if f in df.columns]
    if anom_feats_present and len(anom_feats_present) == len(ANOM_FEATURES):
        X_anom_all = df[ANOM_FEATURES].fillna(0).values
        if sc_anom is not None:
            X_anom_sc = sc_anom.transform(X_anom_all)
        else:
            X_anom_sc = (X_anom_all - X_anom_all.mean(axis=0)) / (X_anom_all.std(axis=0) + 1e-9)
        try:
            anom_flags  = (anom_m.predict(X_anom_sc) == -1).astype(int)
            anom_scores = -anom_m.score_samples(X_anom_sc)
        except Exception:
            anom_flags  = np.zeros(len(df), dtype=int)
            anom_scores = np.zeros(len(df))
    else:
        anom_flags  = np.zeros(len(df), dtype=int)
        anom_scores = np.zeros(len(df))

    # Health index
    health_scores = []
    for _, row in df.iterrows():
        health_scores.append(health_index_single(row.to_dict()))
    health_arr = np.array(health_scores)

    def _alert(fp, hi, ft, anom):
        if fp > 0.8 or hi < 40 or ft == 'Safety_Stop': return 'CRITICAL'
        elif fp > 0.5 or hi < 70 or anom == 1:         return 'WARNING'
        else:                                            return 'NORMAL'

    alert_levels = [_alert(fp, hi, ft, a)
                    for fp, hi, ft, a in zip(fail_probs, health_arr, fault_preds, anom_flags)]

    result = df[['Cycle_No']].copy() if 'Cycle_No' in df.columns else pd.DataFrame({'Cycle_No': range(len(df))})
    result['fail_prob']   = fail_probs
    result['fault_pred']  = fault_preds
    result['rul_pred']    = rul_preds
    result['health_idx']  = health_arr
    result['anom_flag']   = anom_flags
    result['anom_score']  = anom_scores
    result['alert_level'] = alert_levels

    return result, df


# ══════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════
def kpi_card(label, value, sub, color_class="accent"):
    st.markdown(f"""
    <div class="kpi-card {color_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:var(--{color_class})">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def alert_badge(level):
    cls = {'CRITICAL': 'badge-critical', 'WARNING': 'badge-warning', 'NORMAL': 'badge-normal'}.get(level, 'badge-normal')
    icon = {'CRITICAL': '🔴', 'WARNING': '🟡', 'NORMAL': '🟢'}.get(level, '⚪')
    st.markdown(f'<span class="alert-badge {cls}">{icon} {level}</span>', unsafe_allow_html=True)


def gauge_chart(value, title, min_val=0, max_val=100, thresholds=(40, 70)):
    low, high = thresholds
    if value < low:      color = COLORS['critical']
    elif value < high:   color = COLORS['warning']
    else:                color = COLORS['normal']

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14, 'family': 'Rajdhani', 'color': COLORS['muted']}},
        number={'font': {'size': 28, 'family': 'Rajdhani', 'color': color}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickfont': {'size': 9, 'color': COLORS['muted']}},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': COLORS['card'],
            'bordercolor': COLORS['border'],
            'steps': [
                {'range': [0, low],      'color': 'rgba(255,71,87,0.12)'},
                {'range': [low, high],   'color': 'rgba(255,183,77,0.12)'},
                {'range': [high, max_val],'color': 'rgba(0,229,160,0.12)'},
            ],
            'threshold': {'line': {'color': color, 'width': 3}, 'thickness': 0.75, 'value': value},
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        height=200,
        margin=dict(l=20, r=20, t=30, b=10),
        font=dict(family='Rajdhani'),
    )
    return fig


def prob_gauge(value, title):
    return gauge_chart(value * 100, title, 0, 100, (30, 50))


# ══════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════
def chart_health_timeline(result_df):
    fig = go.Figure()
    color_map = {'NORMAL': COLORS['normal'], 'WARNING': COLORS['warning'], 'CRITICAL': COLORS['critical']}
    for level, grp in result_df.groupby('alert_level'):
        fig.add_trace(go.Scatter(
            x=grp['Cycle_No'], y=grp['health_idx'],
            mode='markers', marker=dict(color=color_map.get(level, COLORS['accent']), size=3, opacity=0.7),
            name=level,
        ))
    fig.add_hline(y=70, line=dict(color=COLORS['warning'], dash='dash', width=1), annotation_text='Warning')
    fig.add_hline(y=40, line=dict(color=COLORS['critical'], dash='dash', width=1), annotation_text='Critical')
    fig.update_layout(**PLOTLY_LAYOUT, title='🏥 Machine Health Index', height=320,
                      yaxis_range=[0, 110], showlegend=True,
                      legend=dict(orientation='h', y=-0.15))
    return fig


def chart_fail_prob(result_df):
    colors = [COLORS['critical'] if p > 0.8 else
              COLORS['warning']  if p > 0.5 else COLORS['accent']
              for p in result_df['fail_prob']]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=result_df['Cycle_No'], y=result_df['fail_prob'],
        marker_color=colors, name='Failure Prob',
    ))
    fig.add_hline(y=0.5, line=dict(color=COLORS['warning'], dash='dash', width=1))
    fig.add_hline(y=0.8, line=dict(color=COLORS['critical'], dash='dash', width=1))
    fig.update_layout(**PLOTLY_LAYOUT, title=' Failure Probability per Cycle',
                      height=280, yaxis_range=[0, 1.05])
    return fig


def chart_rul(result_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result_df['Cycle_No'], y=result_df['rul_pred'],
        fill='tozeroy', mode='lines',
        line=dict(color=COLORS['purple'], width=2),
        fillcolor='rgba(206,147,216,0.12)',
        name='Predicted RUL',
    ))
    fig.update_layout(
    **PLOTLY_LAYOUT,
    title=dict(
        text='Remaining Useful Life (Cycles)',
        font=dict(color='#ffcc00', size=16)  # 👈 change color here
    ),
    height=280
    )
    return fig


def chart_anomaly_score(result_df):
    threshold = float(np.percentile(result_df['anom_score'], 95)) if result_df['anom_score'].std() > 0 else 0.5
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result_df['Cycle_No'], y=result_df['anom_score'],
        mode='lines', line=dict(color=COLORS['accent'], width=1, ), name='Anomaly Score',
    ))
    fig.add_hline(y=threshold, line=dict(color=COLORS['critical'], dash='dot', width=1.5),
                  annotation_text='95th pct')
    fig.add_trace(go.Scatter(
        x=result_df[result_df['anom_score'] > threshold]['Cycle_No'],
        y=result_df[result_df['anom_score'] > threshold]['anom_score'],
        mode='markers', marker=dict(color=COLORS['critical'], size=5), name='Anomaly',
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title='Anomaly Score Timeline', height=280)
    return fig


def chart_fault_donut(result_df):
    ft_counts = result_df['fault_pred'].value_counts()
    colors_ft = [COLORS['critical'], COLORS['warning'], COLORS['accent'],
                 COLORS['normal'], COLORS['purple'], '#ff8f00', '#80cbc4']
    fig = go.Figure(go.Pie(
        labels=ft_counts.index, values=ft_counts.values,
        hole=0.55, marker=dict(colors=colors_ft[:len(ft_counts)],
                               line=dict(color=COLORS['bg'], width=2)),
        textfont=dict(family='Rajdhani', size=11),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', height=280, showlegend=True,
        legend=dict(font=dict(family='Rajdhani', size=10, color=COLORS['text'])),
        title=dict(text='Fault Distribution', font=dict(family='Rajdhani', color=COLORS['muted'])),
        margin=dict(l=10, r=10, t=40, b=10),
        annotations=[dict(text='FAULTS', x=0.5, y=0.5, showarrow=False,
                          font=dict(size=10, family='Rajdhani', color=COLORS['muted']))],
    )
    return fig


def chart_alert_breakdown(result_df):
    al_counts = result_df['alert_level'].value_counts()
    colors_al = {'NORMAL': COLORS['normal'], 'WARNING': COLORS['warning'], 'CRITICAL': COLORS['critical']}

    fig = go.Figure(go.Bar(
        x=al_counts.index,
        y=al_counts.values,
        marker_color=[colors_al.get(a, COLORS['accent']) for a in al_counts.index],
        text=al_counts.values,
        textposition='outside',
        textfont=dict(family='JetBrains Mono', size=11, color=COLORS['text']),
    ))

    # FIXED PART
    fig.update_layout(**PLOTLY_LAYOUT, title='Alert Breakdown', height=280)
    fig.update_xaxes(gridcolor='rgba(0,0,0,0)')

    return fig


def chart_sensor_overview(df_raw, cols=4):
    sensors_to_plot = [c for c in [
        'FM_Total_Cycle_TIme', 'Eco_MP_Pressure', 'Eco_FM_Actual_Oil_Temp',
        'Eco_V9_Actual_Pressure', 'Drag_Sand_Weight', 'Cope_Sand_Weight',
    ] if c in df_raw.columns]

    rows_needed = (len(sensors_to_plot) + cols - 1) // cols
    fig = make_subplots(rows=rows_needed, cols=cols,
                        subplot_titles=[s.replace('_', ' ') for s in sensors_to_plot])
    for i, sensor in enumerate(sensors_to_plot):
        r, c = divmod(i, cols)
        x = df_raw.get('Cycle_No', pd.Series(range(len(df_raw))))
        fig.add_trace(go.Scatter(
            x=x, y=df_raw[sensor],
            mode='lines', line=dict(color=COLORS['accent'], width=1),
            name=sensor, showlegend=False,
        ), row=r+1, col=c+1)
        rolling = df_raw[sensor].rolling(50, center=True).mean()
        fig.add_trace(go.Scatter(
            x=x, y=rolling,
            mode='lines', line=dict(color=COLORS['normal'], width=2),
            name='Trend', showlegend=False,
        ), row=r+1, col=c+1)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17,21,32,0.6)',
        font=dict(family='Rajdhani', color=COLORS['text']),
        height=280 * rows_needed,
        margin=dict(l=30, r=20, t=60, b=30),
        title=dict(text='Sensor Trends', font=dict(size=15)),
    )
    for ax in fig.layout:
        if ax.startswith('xaxis') or ax.startswith('yaxis'):
            fig.layout[ax].update(gridcolor=COLORS['border'], showgrid=True)
    return fig


def chart_shap_bar(shap_values, feature_names, top_n=15):
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)[-top_n:]
    vals = mean_abs[idx]
    names = [feature_names[i] for i in idx]
    colors = [COLORS['critical'] if v > np.percentile(vals, 80) else
              COLORS['warning']  if v > np.percentile(vals, 50) else COLORS['accent']
              for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation='h',
        marker=dict(color=colors),
        text=[f'{v:.4f}' for v in vals], textposition='outside',
        textfont=dict(family='JetBrains Mono', size=9, color=COLORS['text']),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title='SHAP — Mean |Feature Importance|',
                      height=420, xaxis_title='Mean |SHAP|', yaxis_title='',
                      yaxis=dict(tickfont=dict(size=9)))
    return fig


def chart_business_impact(result_df):
    total = len(result_df)
    failures = int((result_df['fail_prob'] > 0.5).sum())
    true_det = int(failures * 0.95)
    false_al = failures - true_det

    CYCLES_PER_HOUR, REV_PER_CYCLE = 5, 500
    cost_reactive = failures * (80000 + 8 * CYCLES_PER_HOUR * REV_PER_CYCLE)
    cost_planned  = true_det * (25000 + 2 * CYCLES_PER_HOUR * REV_PER_CYCLE) + false_al * 5000
    savings       = max(0, cost_reactive - cost_planned)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Cost Comparison (₹)', 'Downtime (Hours)'])
    cats = ['Reactive', 'Proactive']
    costs = [cost_reactive / 1e5, cost_planned / 1e5]
    dtimes = [failures * 8, true_det * 2]
    bar_colors = [COLORS['critical'], COLORS['normal']]

    fig.add_trace(go.Bar(x=cats, y=costs, marker_color=bar_colors,
                         text=[f'₹{v:.1f}L' for v in costs], textposition='outside',
                         textfont=dict(family='Rajdhani', size=12)), row=1, col=1)
    fig.add_trace(go.Bar(x=cats, y=dtimes, marker_color=bar_colors,
                         text=[f'{v}h' for v in dtimes], textposition='outside',
                         textfont=dict(family='Rajdhani', size=12)), row=1, col=2)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,21,32,0.6)',
        font=dict(family='Rajdhani', color=COLORS['text']),
        height=320, showlegend=False,
        margin=dict(l=30, r=20, t=60, b=30),
        annotations=[
            dict(text=f'Savings: ₹{savings/1e5:.1f}L', x=0.5, y=-0.22,
                 xref='paper', yref='paper', showarrow=False,
                 font=dict(size=14, color=COLORS['normal'], family='Rajdhani')),
        ]
    )
    for ax in ['xaxis', 'xaxis2', 'yaxis', 'yaxis2']:
        if hasattr(fig.layout, ax):
            getattr(fig.layout, ax).update(gridcolor=COLORS['border'])
    return fig, savings, failures


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
def build_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:1rem 0 0.5rem">
            <div style="font-size:2.5rem">🏭</div>
            <div style="font-family:Rajdhani;font-weight:700;font-size:1.1rem;
                        color:#4fc3f7;letter-spacing:0.1em">FM PREDICTIVE</div>
            <div style="font-family:JetBrains Mono;font-size:0.65rem;
                        color:#6b7494;letter-spacing:0.15em">MAINTENANCE SYSTEM</div>
        </div>
        <hr style="border-color:#1e2535;margin:0.8rem 0">
        """, unsafe_allow_html=True)

        mode = st.radio("Input Mode", [" Upload Dataset", " Manual Sensor Input"],
                        label_visibility="collapsed")

        st.markdown('<div class="section-header" style="margin-top:1rem">Model Status</div>',
                    unsafe_allow_html=True)
        models = load_models()
        if models[0] is None:
            st.error("Models not found in `pm_models/`")
        else:
            st.success(" All models loaded")
            model_names = ['Failure Model', 'Fault Classifier', 'RUL Model',
                           'Anomaly Detector', 'Scaler', 'Fault LabelEnc',
                           'Cond LabelEnc', 'Features JSON']
            for name in model_names:
                st.markdown(f"<div style='font-family:JetBrains Mono;font-size:0.68rem;"
                            f"color:#00e5a0;padding:0.1rem 0'>✓ {name}</div>",
                            unsafe_allow_html=True)

        st.markdown('<hr style="border-color:#1e2535;margin:0.8rem 0">', unsafe_allow_html=True)
        st.markdown("<div style='font-family:JetBrains Mono;font-size:0.6rem;"
                    "color:#6b7494;text-align:center'>v1.0 | FM Machine PDM</div>",
                    unsafe_allow_html=True)

    return mode, models


# ══════════════════════════════════════════════════════════════
# ALERT PANEL
# ══════════════════════════════════════════════════════════════
def render_alert_panel(alert):
    level = alert['alert_level']
    border = {'CRITICAL': COLORS['critical'], 'WARNING': COLORS['warning'],
               'NORMAL': COLORS['normal']}.get(level, COLORS['accent'])
    icon = {'CRITICAL': '🔴', 'WARNING': '🟡', 'NORMAL': '🟢'}.get(level, '⚪')

    st.markdown(f"""
    <div style="border:1px solid {border};border-radius:8px;padding:1.2rem;
                background:rgba(17,21,32,0.8);margin-bottom:1rem;">
        <div style="font-family:JetBrains Mono;font-size:0.75rem;color:{border};
                    letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem">
            {icon} {level} ALERT  ·  Cycle #{alert['cycle_no']}  ·  {alert['timestamp']}
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.8rem;margin-bottom:1rem">
            <div style="text-align:center">
                <div style="font-family:JetBrains Mono;font-size:0.6rem;color:#6b7494">HEALTH INDEX</div>
                <div style="font-family:Rajdhani;font-size:1.6rem;font-weight:700;color:{border}">{alert['health_index']}</div>
            </div>
            <div style="text-align:center">
                <div style="font-family:JetBrains Mono;font-size:0.6rem;color:#6b7494">FAIL PROB</div>
                <div style="font-family:Rajdhani;font-size:1.6rem;font-weight:700;color:{border}">{alert['fail_prob']}%</div>
            </div>
            <div style="text-align:center">
                <div style="font-family:JetBrains Mono;font-size:0.6rem;color:#6b7494">RUL</div>
                <div style="font-family:Rajdhani;font-size:1.6rem;font-weight:700;color:#ce93d8">{alert['rul_cycles']}</div>
            </div>
            <div style="text-align:center">
                <div style="font-family:JetBrains Mono;font-size:0.6rem;color:#6b7494">FAULT TYPE</div>
                <div style="font-family:Rajdhani;font-size:1.2rem;font-weight:700;color:#4fc3f7">{alert['fault_type']}</div>
            </div>
        </div>
        <div style="border-top:1px solid #1e2535;padding-top:0.8rem">
            <div style="font-family:JetBrains Mono;font-size:0.6rem;color:#6b7494;
                        margin-bottom:0.5rem"> RECOMMENDED ACTIONS · Priority: {alert['priority']} · 
                Act within {alert['urgency_hours']}h · Est: {alert['downtime_est']} · {alert['cost_est']}</div>
            {''.join(f"<div class='action-item'>{'✓' if i==0 else str(i+1)+'.'} {a}</div>" for i,a in enumerate(alert['actions']))}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TABS — DATASET MODE
# ══════════════════════════════════════════════════════════════
def render_dataset_mode(models):
    st.markdown('<div class="section-header">Upload FM Sensor Data</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload FM_data.csv", type=['csv'],
                                help="Upload your FM_data.csv sensor file")

    if uploaded is None:
        st.markdown("""
        <div style="border:1px dashed #1e2535;border-radius:8px;padding:3rem;
                    text-align:center;color:#6b7494;font-family:Rajdhani">
            <div style="font-size:3rem;margin-bottom:1rem">📂</div>
            <div style="font-size:1.1rem">Drop your <strong>FM_data.csv</strong> here</div>
            <div style="font-size:0.85rem;margin-top:0.5rem">
                Columns needed: FM_Total_Cycle_TIme, Eco_MP_Pressure, Eco_FM_Actual_Oil_Temp,
                Eco_V9_Actual_Pressure, Drag_Sand_Weight, Cope_Sand_Weight, Cycle_No …
            </div>
        </div>""", unsafe_allow_html=True)
        return

    df_raw = pd.read_csv(uploaded)

    # Parse timestamps if present
    if 'timestamp' in df_raw.columns:
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], errors='coerce')

    # Convert numerics
    for col in df_raw.select_dtypes(include='object').columns:
        if col != 'timestamp':
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    # Fill missing sensor cols with median
    for col in [c for c in SENSOR_COLS if c in df_raw.columns]:
        df_raw[col] = df_raw[col].fillna(df_raw[col].median())

    if 'Cycle_No' not in df_raw.columns:
        df_raw['Cycle_No'] = range(len(df_raw))

    df_raw = df_raw.sort_values('Cycle_No').reset_index(drop=True)

    st.success(f"Loaded {len(df_raw):,} rows × {df_raw.shape[1]} columns")

    with st.spinner("Running predictions on dataset..."):
        result_df, df_feat = predict_dataset(df_raw, models)

    # ── KPI Row ──
    total    = len(result_df)
    n_crit   = (result_df['alert_level'] == 'CRITICAL').sum()
    n_warn   = (result_df['alert_level'] == 'WARNING').sum()
    n_anom   = result_df['anom_flag'].sum()
    avg_hi   = result_df['health_idx'].mean()
    avg_fp   = result_df['fail_prob'].mean()
    avg_rul  = result_df['rul_pred'].mean()
    top_fault= result_df['fault_pred'].mode()[0]

    cols = st.columns(8)
    with cols[0]: kpi_card("Total Cycles", f"{total:,}", "analysed", "accent")
    with cols[1]: kpi_card("Critical", f"{n_crit:,}", f"{n_crit/total*100:.1f}%", "critical")
    with cols[2]: kpi_card("Warning", f"{n_warn:,}", f"{n_warn/total*100:.1f}%", "warning")
    with cols[3]: kpi_card("Anomalies", f"{n_anom:,}", f"{n_anom/total*100:.1f}%", "warning")
    with cols[4]: kpi_card("Avg Health", f"{avg_hi:.1f}", "/ 100", "normal")
    with cols[5]: kpi_card("Avg Fail%", f"{avg_fp*100:.1f}%", "probability", "critical")
    with cols[6]: kpi_card("Avg RUL", f"{avg_rul:.0f}", "cycles left", "purple")
    with cols[7]: kpi_card("Top Fault", top_fault, MAINTENANCE_RULES.get(top_fault,{}).get('priority','—'), "accent")

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

    # ── TABS ──
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Alerts", "Anomaly",
        "RUL", "Sensors", "Business Impact"
    ])

    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(chart_health_timeline(result_df), use_container_width=True)
            st.plotly_chart(chart_fail_prob(result_df), use_container_width=True)
        with c2:
            st.plotly_chart(chart_fault_donut(result_df), use_container_width=True)
            st.plotly_chart(chart_alert_breakdown(result_df), use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">Top Critical Cycles</div>', unsafe_allow_html=True)
        n_show = st.slider("Show top N critical cycles", 1, 20, 5)
        critical_rows = result_df[result_df['alert_level'] == 'CRITICAL'].nlargest(n_show, 'fail_prob')
        if len(critical_rows) == 0:
            critical_rows = result_df.nlargest(n_show, 'fail_prob')
            st.info("No CRITICAL cycles found — showing top by failure probability.")

        for _, row in critical_rows.iterrows():
            alert = generate_alert(
                cycle_no   = int(row['Cycle_No']),
                health_idx = row['health_idx'],
                fault_type = row['fault_pred'],
                fail_prob  = row['fail_prob'],
                rul_val    = row['rul_pred'],
                anomaly    = bool(row['anom_flag']),
            )
            render_alert_panel(alert)

        st.markdown('<div class="section-header">Full Prediction Table</div>', unsafe_allow_html=True)
        display_df = result_df[['Cycle_No', 'fail_prob', 'fault_pred', 'rul_pred',
                                 'health_idx', 'anom_flag', 'alert_level']].copy()
        display_df.columns = ['Cycle No', 'Fail Prob', 'Fault Type', 'RUL', 'Health', 'Anomaly', 'Alert']
        display_df['Fail Prob'] = display_df['Fail Prob'].map('{:.3f}'.format)
        display_df['RUL']       = display_df['RUL'].map('{:.0f}'.format)
        display_df['Health']    = display_df['Health'].map('{:.1f}'.format)
        st.dataframe(display_df, height=350, use_container_width=True)

        csv = result_df.to_csv(index=False)
        st.download_button("Download Predictions CSV", csv,
                           file_name="pdm_predictions.csv", mime="text/csv")

    with tab3:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.plotly_chart(chart_anomaly_score(result_df), use_container_width=True)
        with c2:
            pct_anom = n_anom / total * 100
            st.metric("Anomaly Rate", f"{pct_anom:.1f}%")
            st.metric("Anomalies", f"{n_anom:,}")
            st.metric("Normal", f"{total - n_anom:,}")

        # Anomaly scatter
        fig_sc = go.Figure()
        normal_mask = result_df['anom_flag'] == 0
        if 'Eco_MP_Pressure' in df_raw.columns and 'FM_Total_Cycle_TIme' in df_raw.columns:
            # merge result back
            plot_df = df_raw[['Cycle_No', 'Eco_MP_Pressure', 'FM_Total_Cycle_TIme']].merge(
                result_df[['Cycle_No', 'anom_flag']], on='Cycle_No', how='left')
            nm = plot_df['anom_flag'] == 0
            fig_sc.add_trace(go.Scatter(
                x=plot_df[nm]['Eco_MP_Pressure'], y=plot_df[nm]['FM_Total_Cycle_TIme'],
                mode='markers', marker=dict(color=COLORS['accent'], size=3, opacity=0.4),
                name='Normal'))
            fig_sc.add_trace(go.Scatter(
                x=plot_df[~nm]['Eco_MP_Pressure'], y=plot_df[~nm]['FM_Total_Cycle_TIme'],
                mode='markers', marker=dict(color=COLORS['critical'], size=6, opacity=0.9),
                name='Anomaly'))
            fig_sc.update_layout(**PLOTLY_LAYOUT, title='Anomaly Scatter: Pressure vs Cycle Time',
                                 height=320, xaxis_title='MP Pressure', yaxis_title='Cycle Time')
            st.plotly_chart(fig_sc, use_container_width=True)

    with tab4:
        st.plotly_chart(chart_rul(result_df), use_container_width=True)

        # RUL distribution
        fig_rul_hist = px.histogram(result_df, x='rul_pred', nbins=50,
                                    color_discrete_sequence=[COLORS['purple']],
                                    title='RUL Distribution Across All Cycles')
        fig_rul_hist.update_layout(**PLOTLY_LAYOUT, height=260)
        st.plotly_chart(fig_rul_hist, use_container_width=True)

        # Summary stats
        rul_stats = result_df['rul_pred'].describe()
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1: st.metric("Min RUL", f"{rul_stats['min']:.0f}")
        with sc2: st.metric("25th pct", f"{rul_stats['25%']:.0f}")
        with sc3: st.metric("Median", f"{rul_stats['50%']:.0f}")
        with sc4: st.metric("Max RUL", f"{rul_stats['max']:.0f}")

    with tab5:
        st.plotly_chart(chart_sensor_overview(df_raw), use_container_width=True)

        # Correlation heatmap
        sens_cols = [c for c in SENSOR_COLS if c in df_raw.columns and c != 'Cycle_No']
        if len(sens_cols) > 2:
            corr = df_raw[sens_cols].corr().round(2)
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale='RdYlGn', zmid=0, zmin=-1, zmax=1,
                text=corr.values, texttemplate='%{text:.2f}',
                textfont=dict(size=8),
                colorbar=dict(tickfont=dict(color=COLORS['text'])),
            ))
            fig_corr.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(17,21,32,0.8)',
                font=dict(family='Rajdhani', color=COLORS['text'], size=9),
                title='Sensor Correlation Matrix',
                height=420, margin=dict(l=100, r=20, t=50, b=100),
                xaxis=dict(tickangle=-35),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    with tab6:
        fig_biz, savings, failures = chart_business_impact(result_df)
        st.plotly_chart(fig_biz, use_container_width=True)
        b1, b2, b3, b4 = st.columns(4)
        with b1: st.metric("Cost Savings", f"₹{savings/1e5:.1f}L")
        with b2: st.metric("Failures Detected", f"{failures:,}")
        with b3: st.metric("Downtime Saved", f"{failures*6:,}h")
        with b4: st.metric("Revenue Protected", f"₹{failures*6*5*500/1e5:.1f}L")

        # Maintenance schedule
        st.markdown('<div class="section-header" style="margin-top:1rem">Recommended Maintenance Schedule</div>',
                    unsafe_allow_html=True)
        urgent = result_df[result_df['alert_level'].isin(['CRITICAL', 'WARNING'])]
        if len(urgent) > 0:
            top_f  = urgent['fault_pred'].mode()[0]
            rules  = MAINTENANCE_RULES.get(top_f, MAINTENANCE_RULES['Normal'])
            st.markdown(f"""
            <div style="border:1px solid {COLORS['warning']};border-radius:8px;padding:1rem;
                        background:rgba(255,183,77,0.05)">
                <div style="font-family:Rajdhani;font-weight:700;font-size:1rem;color:{COLORS['warning']}">
                    {top_f} — {rules['priority']} Priority
                </div>
                <div style="font-family:JetBrains Mono;font-size:0.7rem;color:{COLORS['muted']};margin:0.4rem 0">
                    Act within {rules['urgency_hours']}h  ·  Est. downtime: {rules['downtime_est']}  ·  Cost: {rules['cost_est']}
                </div>
                {''.join(f"<div class='action-item'>{i+1}. {a}</div>" for i,a in enumerate(rules['actions']))}
            </div>""", unsafe_allow_html=True)
        else:
            st.success("No urgent maintenance required. Continue routine monitoring.")


# ══════════════════════════════════════════════════════════════
# TABS — MANUAL SENSOR INPUT MODE
# ══════════════════════════════════════════════════════════════
def render_manual_mode(models):
    st.markdown('<div class="section-header">Manual Sensor Reading</div>', unsafe_allow_html=True)

    with st.form("sensor_form"):
        st.markdown("#### Cycle Identity")
        ci1, ci2 = st.columns(2)
        with ci1: cycle_no        = st.number_input("Cycle No",            value=46123, step=1)
        with ci2: life_cycle      = st.number_input("Life Cycle Counter",   value=45000, step=100)

        st.markdown("#### Timing Parameters")
        t1, t2, t3 = st.columns(3)
        with t1: total_cycle_time = st.number_input("FM Total Cycle Time (ms)", value=1850, step=10)
        with t2: cope_time        = st.number_input("Cope Cycle Time (ms)",     value=780,  step=10)
        with t3: drag_time        = st.number_input("Drag Cycle Time (ms)",     value=820,  step=10)

        st.markdown("#### Pressure & Temperature")
        p1, p2, p3 = st.columns(3)
        with p1: mp_pressure      = st.number_input("MP Pressure (bar)",         value=42.0, step=0.5)
        with p2: v9_pressure      = st.number_input("V9 Actual Pressure (bar)",  value=48.0, step=0.5)
        with p3: oil_temp         = st.number_input("Oil Temperature (°C)",       value=65.0, step=0.5)

        st.markdown("#### Sand & Squeeze")
        s1, s2, s3 = st.columns(3)
        with s1: drag_sand        = st.number_input("Drag Sand Weight (kg)",  value=74.0, step=0.5)
        with s2: cope_sand        = st.number_input("Cope Sand Weight (kg)",  value=73.0, step=0.5)
        with s3: drag_seq         = st.number_input("Drag Seq Pressure (bar)",value=55.0, step=0.5)

        st.markdown("#### Cylinder States")
        cy1, cy2, cy3, cy4 = st.columns(4)
        with cy1: v9_up    = st.selectbox("V9 Cylinder Up",   [1, 0], index=0)
        with cy2: v9_down  = st.selectbox("V9 Cylinder Down", [0, 1], index=0)
        with cy3: v10_in   = st.selectbox("V10 Cylinder In",  [1, 0], index=0)
        with cy4: v10_out  = st.selectbox("V10 Cylinder Out", [0, 1], index=0)

        shift_cycle = st.number_input("Shift Cycle Counter", value=120, step=1)

        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    if not submitted:
        st.markdown("""
        <div style="text-align:center;color:#6b7494;font-family:Rajdhani;
                    padding:2rem;font-size:0.95rem">
            Fill in sensor values above and click <strong>Run Prediction</strong>
        </div>""", unsafe_allow_html=True)
        return

    sensor_dict = {
        'FM_Total_Cycle_TIme':    total_cycle_time,
        'Life_Cycle_Counter':     life_cycle,
        'Shift_Cycle_Counter':    shift_cycle,
        'Cope_Cycle_Time':        cope_time,
        'Drag_Cycle_Time':        drag_time,
        'Eco_MP_Pressure':        mp_pressure,
        'Eco_FM_Actual_Oil_Temp': oil_temp,
        'V9_Cylinder_Up':         v9_up,
        'V9_Cylinder_Down':       v9_down,
        'V10_Cylinder_in':        v10_in,
        'V10_Cylinder_out':       v10_out,
        'Eco_V9_Actual_Pressure': v9_pressure,
        'Cycle_No':               cycle_no,
        'Drag_Sand_Weight':       drag_sand,
        'Drag_Seq_Pressure':      drag_seq,
        'Cope_Sand_Weight':       cope_sand,
    }

    with st.spinner("Computing prediction..."):
        alert, anom_score, X_new = predict_single(sensor_dict, models)

    # ── Alert panel ──
    render_alert_panel(alert)

    # ── Gauges ──
    g1, g2, g3 = st.columns(3)
    with g1: st.plotly_chart(gauge_chart(alert['health_index'], 'Health Index'), use_container_width=True)
    with g2: st.plotly_chart(gauge_chart(alert['fail_prob'], 'Failure Probability (%)', thresholds=(30, 50)), use_container_width=True)
    with g3: st.plotly_chart(gauge_chart(min(100, alert['rul_cycles'] / 10), 'RUL (scaled)', thresholds=(20, 50)), use_container_width=True)

    # ── Sensor Radar ──
    st.markdown('<div class="section-header" style="margin-top:1rem">Sensor Normalised View</div>',
                unsafe_allow_html=True)
    sensor_norms = {
        'Cycle Time':    min(1.0, total_cycle_time / 2000),
        'MP Pressure':   1 - min(1.0, max(0, (mp_pressure - 35) / 25)),
        'Oil Temp':      min(1.0, max(0, (oil_temp - 35) / 35)),
        'V9 Pressure':   1 - min(1.0, max(0, (v9_pressure - 40) / 20)),
        'Drag Sand':     min(1.0, drag_sand / 100),
        'Cope Sand':     min(1.0, cope_sand / 100),
        'Anomaly Score': min(1.0, anom_score),
    }
    labels = list(sensor_norms.keys())
    values = list(sensor_norms.values())
    values_plot = values + [values[0]]
    labels_plot  = labels + [labels[0]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=values_plot, theta=labels_plot,
        fill='toself',
        line=dict(color=COLORS['accent'], width=2),
        fillcolor='rgba(79,195,247,0.15)',
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(17,21,32,0.6)',
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=COLORS['border'],
                            tickfont=dict(size=8, color=COLORS['muted'])),
            angularaxis=dict(gridcolor=COLORS['border'],
                             tickfont=dict(size=10, family='Rajdhani', color=COLORS['text'])),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Rajdhani', color=COLORS['text']),
        title=dict(text='Sensor Risk Profile (1 = worst)', font=dict(size=13)),
        height=380, margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── SHAP (if available) ──
    try:
        import shap
        fail_m = models[0]
        feats  = models[7]
        explainer = shap.TreeExplainer(fail_m)
        sv = explainer.shap_values(X_new)
        if isinstance(sv, list): sv = sv[1]
        elif sv.ndim == 3:       sv = sv[:, :, 1]
        st.plotly_chart(chart_shap_bar(sv, feats), use_container_width=True)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    mode, models = build_sidebar()

    # Title
    st.markdown("""
    <div class="dash-title">
        Foundry Machine — Predictive Maintenance Dashboard
    </div>
    """, unsafe_allow_html=True)

    if models[0] is None:
        st.error("""
        **Models not found.** Place your `pm_models/` folder (generated by the notebook) 
        in the same directory as this script.
        
        Expected files:
        - `pm_models/failure_model.pkl`
        - `pm_models/fault_clf.pkl`
        - `pm_models/rul_model.pkl`
        - `pm_models/anomaly_model.pkl`
        - `pm_models/scaler.pkl`
        - `pm_models/label_enc_fault.pkl`
        - `pm_models/label_enc_cond.pkl`
        - `pm_models/selected_features.json`
        """)
        return

    if "Upload" in mode:
        render_dataset_mode(models)
    else:
        render_manual_mode(models)

if __name__ == "__main__":
    main()