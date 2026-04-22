import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="Heat Exchanger Noise Analysis", layout="wide")

# App Header
st.title("📊 Model Comparison: Clean vs. Noisy Data")
st.markdown(f"""
**Student:** Lovelesh Borile | **Roll No:** 254107204  
**Course:** CL653 - Applications of AI and ML for Chemical Engineering  
This tool compares predictions from a model trained on ideal data (A) against one trained on noisy data (B).
""")

# 1. Load Assets
@st.cache_resource
def load_assets():
    model_a = joblib.load('heat_exchanger_model_A.pkl')
    model_b = joblib.load('heat_exchanger_model_B.pkl')
    scaler = joblib.load('scaler.pkl')
    return model_a, model_b, scaler

model_a, model_b, scaler = load_assets()

# 2. Sidebar Inputs (9 features as expected by your model)
st.sidebar.header("📥 Input Operating Parameters")

def get_inputs():
    h_in_t = st.sidebar.number_input("Hot Inlet Temp (K)", value=373.15)
    c_in_f = st.sidebar.number_input("Cold Inlet Mass Flow (kg/s)", value=0.55)
    h_out_t = st.sidebar.number_input("Hot Outlet Temp (K)", value=378.0)
    c_out_t = st.sidebar.number_input("Cold Outlet Temp (K)", value=200.0)
    h_out_p = st.sidebar.number_input("Hot Outlet Pressure (Pa)", value=2011.0)
    c_out_p = st.sidebar.number_input("Cold Outlet Pressure (Pa)", value=500000.0)
    h_out_f = st.sidebar.number_input("Hot Outlet Mass Flow (kg/s)", value=100.0)
    c_out_f = st.sidebar.number_input("Cold Outlet Mass Flow (kg/s)", value=1.0)
    lmtd = st.sidebar.number_input("LMTD (K)", value=0.55)

    data = {
        'hot_inlet_temperature_k': h_in_t,
        'cold_inlet_mass_flow_kg_s': c_in_f,
        'hot_outlet_temperature_k': h_out_t,
        'cold_outlet_temperature_k': c_out_t,
        'hot_outlet_pressure_pa': h_out_p,
        'cold_outlet_pressure_pa': c_out_p,
        'hot_outlet_mass_flow_kg_s': h_out_f,
        'cold_outlet_mass_flow_kg_s': c_out_f,
        'hx_1_logarithmic_mean_temperature_difference_lmtd_k': lmtd
    }
    return pd.DataFrame([data])

input_df = get_inputs()

# 3. Main Dashboard Prediction Logic
if st.button("Generate Comparative Prediction"):
    # Pre-processing
    scaled_input = scaler.transform(input_df)
    
    # Run both models
    pred_A = model_a.predict(scaled_input)[0]
    pred_B = model_b.predict(scaled_input)[0]
    
    # Calculate Error/Difference
    abs_diff = abs(pred_A - pred_B)
    pct_diff = (abs_diff / pred_A) * 100

    # Display Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model A (Clean)", f"{pred_A:.2f} kW")
        st.caption("Theoretical/Ideal Prediction")
        
    with col2:
        st.metric("Model B (Noisy)", f"{pred_B:.2f} kW")
        st.caption("Realistic Industrial Prediction")
        
    with col3:
        st.metric("Prediction Discrepancy", f"{abs_diff:.2f} kW", delta=f"{pct_diff:.2f}%", delta_color="inverse")
        st.caption("Impact of Sensor Noise")

    # 4. Insight and Maintenance Logic
    st.markdown("---")
    st.subheader("Analysis & Maintenance Insight")
    
    if pct_diff > 10:
        st.warning(f"⚠️ High Discrepancy detected ({pct_diff:.1f}%). The noise in the current operating data is significantly affecting the prediction accuracy. Consider instrument calibration or denoising.")
    else:
        st.success(f"✅ Low Discrepancy ({pct_diff:.1f}%). The model remains stable under these specific operating conditions despite the noise.")

    # Show raw data for transparency
    with st.expander("View Input Data Table"):
        st.table(input_df)

else:
    st.info("Adjust parameters in the sidebar and click the button to see the comparison.")