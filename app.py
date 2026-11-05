import os
# Suppress noisy TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# Page & Layout Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Reizman Suzuki Yield Predictor | Industrial AI Analytics",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling: Premium Industrial Dark Theme (Blue & Cyan)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Background & Font styling */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0f172a 0%, #080d19 100%);
        color: #e2e8f0;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Headers & Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Card Container */
    .industrial-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5), 0 0 15px rgba(6, 182, 212, 0.08);
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
        transition: border 0.3s ease, box-shadow 0.3s ease;
    }
    .industrial-card:hover {
        border-color: rgba(56, 189, 248, 0.38);
        box-shadow: 0 4px 25px -2px rgba(0, 0, 0, 0.6), 0 0 20px rgba(6, 182, 212, 0.16);
    }
    
    /* Industrial Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .badge-cyan {
        background: rgba(6, 182, 212, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
    }
    .badge-blue {
        background: rgba(37, 99, 235, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.35);
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.35);
    }
    .badge-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.35);
    }
    
    /* Prominent Button Glow */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%);
        color: #ffffff;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 12px 28px;
        border: none;
        border-radius: 8px;
        box-shadow: 0 0 15px rgba(6, 182, 212, 0.45);
        transition: all 0.25s ease-in-out;
        letter-spacing: 0.03em;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0369a1 0%, #0891b2 100%);
        box-shadow: 0 0 25px rgba(6, 182, 212, 0.75);
        transform: translateY(-1px);
        color: #f0fdfa;
    }
    
    /* Metric Card Styling */
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #a5f3fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0c1222;
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }
    
    /* Custom Warning / Alert Banner */
    .warning-box {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 8px;
        padding: 12px 16px;
        color: #fde68a;
        font-size: 0.88rem;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    
    /* Tabs custom styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(56, 189, 248, 0.2);
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 600;
        border-radius: 6px 6px 0 0;
        padding: 10px 18px;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
        background: rgba(56, 189, 248, 0.06);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Domain Constants & Specifications
# ---------------------------------------------------------
CATALYST_OPTIONS = [
    "P1-L1", "P1-L2", "P1-L3", "P1-L4", 
    "P1-L5", "P1-L6", "P1-L7", "P2-L1"
]

FEATURE_COLUMNS = ["catalyst", "t_res", "temperature", "catalyst_loading"]

# Experimentally validated domain bounds from Reizman et al. Case 1
RANGES = {
    "t_res": {"min": 60.0, "max": 600.0, "default": 189.7, "unit": "seconds"},
    "temperature": {"min": 30.0, "max": 110.0, "default": 65.0, "unit": "°C"},
    "catalyst_loading": {"min": 0.496, "max": 2.515, "default": 1.250, "unit": "mol%"}
}

# ---------------------------------------------------------
# Path Configuration (Relative to Project Root)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "yield_preprocessor.joblib")
MODEL_PATH = os.path.join(BASE_DIR, "yield_nn_model.keras")
DATASET_PATH = os.path.join(BASE_DIR, "reizman_suzuki_case_1_raw.csv")

# ---------------------------------------------------------
# Model & Preprocessor Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Neural Network Inference Engine...")
def load_ml_pipeline():
    """
    Loads the trained scikit-learn preprocessor and Keras 3 neural network model.
    Cached across user sessions to avoid reloading.
    """
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor file not found at: {PREPROCESSOR_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Keras model file not found at: {MODEL_PATH}")

    # Set Keras backend to tensorflow if required
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    import keras

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = keras.models.load_model(MODEL_PATH)
    return preprocessor, model

@st.cache_data
def load_dataset():
    """Loads historical experimental data for statistics & visualizations."""
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    return None

# Load assets
try:
    preprocessor, model = load_ml_pipeline()
    pipeline_loaded = True
except Exception as e:
    pipeline_loaded = False
    pipeline_error = str(e)

raw_df = load_dataset()

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = None

# ---------------------------------------------------------
# Sidebar: System Monitor & Presets
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ SYSTEM TELEMETRY")
    
    if pipeline_loaded:
        st.markdown(
            """
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(52, 211, 153, 0.3); border-radius: 8px; padding: 10px 14px; margin-bottom: 16px;">
                <span style="color: #34d399; font-weight: 700;">● SYSTEM OPERATIONAL</span><br>
                <small style="color: #94a3b8;">Inference Engine: Keras 3 (TensorFlow)</small><br>
                <small style="color: #94a3b8;">Topology: 11 → 32 → 16 → 8 → 1</small>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.error(f"Engine Offline: {pipeline_error}")

    st.markdown("#### 🔬 Process Boundaries")
    st.markdown(
        """
        - **Residence Time:** `60.0 - 600.0 s`
        - **Temperature:** `30.0 - 110.0 °C`
        - **Catalyst Loading:** `0.496 - 2.515 mol%`
        - **Validated Ligands:** 8 Suzuki pre-catalysts
        """
    )
    st.markdown("---")
    
    st.markdown("#### ⚡ Reaction Condition Presets")
    st.caption("Load verified reference conditions:")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("🔥 High Yield (P1-L4)", use_container_width=True):
            st.session_state["sel_catalyst"] = "P1-L4"
            st.session_state["sel_t_res"] = 600.0
            st.session_state["sel_temp"] = 110.0
            st.session_state["sel_loading"] = 1.106
            st.rerun()
    with col_p2:
        if st.button("⏱️ Rapid Flow (P1-L7)", use_container_width=True):
            st.session_state["sel_catalyst"] = "P1-L7"
            st.session_state["sel_t_res"] = 155.6
            st.session_state["sel_temp"] = 110.0
            st.session_state["sel_loading"] = 2.499
            st.rerun()

    if st.button("❄️ Ambient Baseline (P1-L1)", use_container_width=True):
        st.session_state["sel_catalyst"] = "P1-L1"
        st.session_state["sel_t_res"] = 60.0
        st.session_state["sel_temp"] = 30.0
        st.session_state["sel_loading"] = 0.513
        st.rerun()

    st.markdown("---")
    st.caption("Reizman Suzuki Continuous-Flow Cross-Coupling Case 1 • Neural Network Surrogate Model")

# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.markdown("""
<div style="margin-bottom: 24px;">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
        <div>
            <h1 style="margin: 0; font-size: 2.1rem; color: #f8fafc;">
                <span style="color: #38bdf8;">REIZMAN SUZUKI</span> YIELD PREDICTOR
            </h1>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                Continuous-Flow Synthesis Analytics & Neural Network Process Optimization
            </p>
        </div>
        <div>
            <span class="badge badge-cyan">Keras 3.x Engine</span>
            <span class="badge badge-blue">Scikit-Learn Transformer</span>
            <span class="badge badge-green">Validated On 96 Runs</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tabs Navigation
# ---------------------------------------------------------
tab_prediction, tab_history, tab_dataset = st.tabs([
    "🎯 Reaction Simulator & Yield Inference",
    "📜 Prediction History & Audit Log",
    "📊 Dataset Analytics & Process Space"
])

# =========================================================
# TAB 1: Prediction & Simulator
# =========================================================
with tab_prediction:
    col_input, col_result = st.columns([1.1, 1.2], gap="large")

    with col_input:
        st.markdown("""
        <div class="industrial-card">
            <h3 style="margin-top: 0; font-size: 1.25rem; display: flex; align-items: center; gap: 8px;">
                <span>🧪</span> Reaction Parameters Configuration
            </h3>
            <p style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 18px;">
                Adjust continuous-flow parameters below to evaluate predicted Suzuki-Miyaura reaction yield.
            </p>
        """, unsafe_allow_html=True)

        # 1. Catalyst Dropdown
        default_cat = st.session_state.get("sel_catalyst", "P1-L4")
        cat_index = CATALYST_OPTIONS.index(default_cat) if default_cat in CATALYST_OPTIONS else 3
        catalyst = st.selectbox(
            "1. Catalyst / Pre-catalyst Complex",
            options=CATALYST_OPTIONS,
            index=cat_index,
            help="Select the palladium catalyst / ligand combination tested in the continuous-flow platform."
        )

        # 2. Residence Time (t_res)
        default_tres = float(st.session_state.get("sel_t_res", RANGES["t_res"]["default"]))
        t_res = st.slider(
            "2. Residence Time (t_res) [seconds]",
            min_value=30.0,
            max_value=700.0,
            value=default_tres,
            step=1.0,
            help="Total time the reaction mixture spends inside the flow microreactor. Valid domain: 60 - 600 s."
        )

        # 3. Temperature
        default_temp = float(st.session_state.get("sel_temp", RANGES["temperature"]["default"]))
        temperature = st.slider(
            "3. Reactor Temperature [°C]",
            min_value=20.0,
            max_value=130.0,
            value=default_temp,
            step=0.5,
            help="Continuous microreactor heating block temperature. Valid domain: 30 - 110 °C."
        )

        # 4. Catalyst Loading
        default_loading = float(st.session_state.get("sel_loading", RANGES["catalyst_loading"]["default"]))
        catalyst_loading = st.slider(
            "4. Catalyst Loading [mol%]",
            min_value=0.200,
            max_value=3.000,
            value=default_loading,
            step=0.005,
            format="%.3f",
            help="Molar concentration ratio of catalyst relative to limiting substrate. Valid domain: 0.496 - 2.515 mol%."
        )

        # Out-of-bounds warning checks
        warnings = []
        if t_res < RANGES["t_res"]["min"] or t_res > RANGES["t_res"]["max"]:
            warnings.append(f"Residence Time ({t_res:.1f} s) is outside the experimentally validated dataset domain [{RANGES['t_res']['min']} – {RANGES['t_res']['max']} s].")
        if temperature < RANGES["temperature"]["min"] or temperature > RANGES["temperature"]["max"]:
            warnings.append(f"Temperature ({temperature:.1f} °C) is outside the experimental range [{RANGES['temperature']['min']} – {RANGES['temperature']['max']} °C].")
        if catalyst_loading < RANGES["catalyst_loading"]["min"] or catalyst_loading > RANGES["catalyst_loading"]["max"]:
            warnings.append(f"Catalyst loading ({catalyst_loading:.3f} mol%) is outside experimental boundaries [{RANGES['catalyst_loading']['min']} – {RANGES['catalyst_loading']['max']} mol%].")

        if warnings:
            warning_html = "<div class='warning-box'><strong>⚠️ Extrapolation Boundary Warning:</strong><br>" + "<br>".join([f"• {w}" for w in warnings]) + "<br><small style='color: #fcd34d;'>Model predictions in extrapolation regimes carry higher epistemic uncertainty.</small></div>"
            st.markdown(warning_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Prominent Predict Button
        predict_clicked = st.button("⚡ PREDICT REACTION YIELD", use_container_width=True)

    with col_result:
        # Perform prediction when button is clicked or if current_prediction exists
        if predict_clicked:
            if not pipeline_loaded:
                st.error("Cannot predict: Inference engine failed to load.")
            else:
                # 1. Construct input DataFrame with exact feature order
                input_df = pd.DataFrame([{
                    "catalyst": str(catalyst),
                    "t_res": float(t_res),
                    "temperature": float(temperature),
                    "catalyst_loading": float(catalyst_loading)
                }])
                input_df = input_df[FEATURE_COLUMNS]

                # 2. Transform using preprocessor
                try:
                    transformed_features = preprocessor.transform(input_df)
                    raw_prediction = float(model.predict(transformed_features, verbose=0)[0][0])
                    
                    # Bounded yield for physical visualization (0 - 100%)
                    bounded_yield = max(0.0, min(100.0, raw_prediction))

                    # Timestamp & record
                    record = {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "catalyst": catalyst,
                        "t_res": round(t_res, 1),
                        "temperature": round(temperature, 1),
                        "catalyst_loading": round(catalyst_loading, 3),
                        "predicted_yield": round(raw_prediction, 2),
                        "bounded_yield": round(bounded_yield, 2),
                        "extrapolation": "Yes" if len(warnings) > 0 else "No"
                    }
                    
                    st.session_state.current_prediction = record
                    # Append to history
                    st.session_state.prediction_history.insert(0, record)

                except Exception as pred_err:
                    st.error(f"Inference Pipeline Error: {pred_err}")

        # Display Result Card
        pred_data = st.session_state.current_prediction

        if pred_data is not None:
            raw_val = pred_data["predicted_yield"]
            display_val = pred_data["bounded_yield"]
            
            # Categorize regime
            if display_val >= 75.0:
                qual_status = "Optimal High Yield Regime"
                status_color = "#34d399"
                badge_class = "badge-green"
            elif display_val >= 40.0:
                qual_status = "Moderate Yield Regime"
                status_color = "#38bdf8"
                badge_class = "badge-cyan"
            else:
                qual_status = "Sub-optimal / Low Yield"
                status_color = "#fbbf24"
                badge_class = "badge-amber"

            st.markdown(f"""
            <div class="industrial-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="badge {badge_class}">{qual_status}</span>
                    <small style="color: #64748b;">Evaluated at: {pred_data['timestamp'].split(' ')[1]}</small>
                </div>
                <div style="display: flex; align-items: baseline; gap: 12px; margin-top: 6px;">
                    <span class="metric-value">{display_val:.1f}%</span>
                    <span style="color: #94a3b8; font-size: 1.1rem; font-weight: 500;">Predicted Suzuki Yield</span>
                </div>
                <div class="metric-sub">
                    Raw Neural Network Output: <strong style="color: #e2e8f0;">{raw_val:.2f}%</strong>
                </div>
            """, unsafe_allow_html=True)

            # Circular Gauge with Plotly
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=display_val,
                number={"suffix": "%", "font": {"size": 36, "color": "#f8fafc"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                    "bar": {"color": "#38bdf8", "thickness": 0.28},
                    "bgcolor": "rgba(15, 23, 42, 0.8)",
                    "borderwidth": 2,
                    "bordercolor": "rgba(56, 189, 248, 0.2)",
                    "steps": [
                        {"range": [0, 30], "color": "rgba(239, 68, 68, 0.15)"},
                        {"range": [30, 70], "color": "rgba(245, 158, 11, 0.15)"},
                        {"range": [70, 100], "color": "rgba(16, 185, 129, 0.15)"}
                    ],
                    "threshold": {
                        "line": {"color": "#00f2fe", "width": 4},
                        "thickness": 0.8,
                        "value": display_val
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=25, b=20),
                height=240,
                font={"color": "#e2e8f0"}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Input Summary Breakdown Cards
            st.markdown("#### 📋 Operational Input Summary")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); padding: 10px; border-radius: 8px; border: 1px solid rgba(56, 189, 248, 0.15); text-align: center;">
                    <small style="color: #94a3b8;">Catalyst</small><br>
                    <strong style="color: #38bdf8; font-size: 1.1rem;">{pred_data['catalyst']}</strong>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); padding: 10px; border-radius: 8px; border: 1px solid rgba(56, 189, 248, 0.15); text-align: center;">
                    <small style="color: #94a3b8;">Residence Time</small><br>
                    <strong style="color: #e2e8f0; font-size: 1.1rem;">{pred_data['t_res']} s</strong>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); padding: 10px; border-radius: 8px; border: 1px solid rgba(56, 189, 248, 0.15); text-align: center;">
                    <small style="color: #94a3b8;">Temperature</small><br>
                    <strong style="color: #e2e8f0; font-size: 1.1rem;">{pred_data['temperature']} °C</strong>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); padding: 10px; border-radius: 8px; border: 1px solid rgba(56, 189, 248, 0.15); text-align: center;">
                    <small style="color: #94a3b8;">Loading</small><br>
                    <strong style="color: #e2e8f0; font-size: 1.1rem;">{pred_data['catalyst_loading']} mol%</strong>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        else:
            # Standby placeholder card
            st.markdown("""
            <div class="industrial-card" style="text-align: center; padding: 40px 20px;">
                <div style="font-size: 3rem; margin-bottom: 12px; color: #38bdf8;">⚡</div>
                <h3 style="margin-bottom: 8px;">Inference Standby</h3>
                <p style="color: #94a3b8; font-size: 0.95rem; max-width: 440px; margin: 0 auto 20px auto;">
                    Adjust chemical reaction parameters on the left or choose a verified preset, then click 
                    <strong style="color: #38bdf8;">Predict Reaction Yield</strong> to evaluate model output.
                </p>
                <div style="display: flex; justify-content: center; gap: 8px;">
                    <span class="badge badge-cyan">Feature Dimension: 11</span>
                    <span class="badge badge-blue">Keras 3 Sequential</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# TAB 2: Prediction History & Audit Log
# =========================================================
with tab_history:
    st.markdown("### 📜 Session Prediction Audit Log")
    st.caption("All evaluations executed during this session are recorded in memory.")

    if len(st.session_state.prediction_history) == 0:
        st.info("No predictions recorded yet. Run a prediction in the simulator tab to populate the audit log.")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Summary metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Predictions Run", len(history_df))
        with col_m2:
            st.metric("Max Yield Achieved", f"{history_df['predicted_yield'].max():.2f}%")
        with col_m3:
            st.metric("Average Yield", f"{history_df['predicted_yield'].mean():.2f}%")
        with col_m4:
            st.metric("Top Performing Catalyst", history_df.loc[history_df['predicted_yield'].idxmax()]['catalyst'])

        # Display Table
        st.dataframe(
            history_df[[
                "timestamp", "catalyst", "t_res", "temperature", 
                "catalyst_loading", "predicted_yield", "extrapolation"
            ]].rename(columns={
                "timestamp": "Timestamp",
                "catalyst": "Catalyst",
                "t_res": "t_res (s)",
                "temperature": "Temp (°C)",
                "catalyst_loading": "Loading (mol%)",
                "predicted_yield": "Predicted Yield (%)",
                "extrapolation": "Extrapolated?"
            }),
            use_container_width=True,
            hide_index=True
        )

        col_dl, col_clr = st.columns([1, 1])
        with col_dl:
            csv_export = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Prediction History (CSV)",
                data=csv_export,
                file_name=f"suzuki_yield_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_clr:
            if st.button("🗑️ Clear Session History", use_container_width=True):
                st.session_state.prediction_history = []
                st.session_state.current_prediction = None
                st.rerun()

# =========================================================
# TAB 3: Dataset Analytics & Space
# =========================================================
with tab_dataset:
    st.markdown("### 📊 Reizman Suzuki-Miyaura Experimental Space")
    st.caption("Exploration of the continuous-flow training dataset (Case 1: 96 experimental runs).")

    if raw_df is not None:
        # Top KPI stats
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Experimental Runs", len(raw_df))
        with kpi2:
            st.metric("Highest Experimental Yield", f"{raw_df['yld'].max():.1f}%")
        with kpi3:
            st.metric("Average Experimental Yield", f"{raw_df['yld'].mean():.1f}%")
        with kpi4:
            st.metric("Lowest Experimental Yield", f"{raw_df['yld'].min():.1f}%")

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            # Yield by Catalyst Boxplot
            fig_box = px.box(
                raw_df, 
                x="catalyst", 
                y="yld", 
                color="catalyst",
                title="Yield Distribution by Catalyst Complex",
                labels={"catalyst": "Catalyst Complex", "yld": "Yield (%)"},
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_box.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                font={"color": "#94a3b8"},
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with col_chart2:
            # Temperature vs t_res scatter
            fig_scatter = px.scatter(
                raw_df,
                x="temperature",
                y="t_res",
                color="yld",
                size="catalyst_loading",
                hover_data=["catalyst", "ton"],
                title="Process Space: Temperature vs Residence Time",
                labels={"temperature": "Temperature (°C)", "t_res": "Residence Time (s)", "yld": "Yield (%)"},
                color_continuous_scale="Blues"
            )
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                font={"color": "#94a3b8"},
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Dataset Preview
        with st.expander("🔍 View Raw Training Dataset (reizman_suzuki_case_1_raw.csv)"):
            st.dataframe(raw_df, use_container_width=True, hide_index=True)
    else:
        st.warning("`reizman_suzuki_case_1_raw.csv` not found in workspace.")

