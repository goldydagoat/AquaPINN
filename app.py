"""
AquaPINN: Physics-Informed Digital Twin for Wastewater Treatment Plants.
Streamlit Web Application combining ASM1 First-Principles ODE Simulation with PyTorch PINN Surrogate Models.
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from models.asm1_ode import (
    STATE_NAMES,
    STATE_UNITS,
    NUM_STATES,
    ASM1Parameters,
    ASM1Simulator,
    simulate_asm1,
    get_default_influent,
)
from models.surrogate_pinn import ASM1PINN, PINNConfig, PINNTrainer, train_pinn
from data_handler import process_scada_file, generate_sample_scada_csv

# Page Configuration
st.set_page_config(
    page_title="AquaPINN Digital Twin",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Controls
st.sidebar.markdown("## ⚙️ Operational Controls")
input_mode = st.sidebar.radio(
    "Operating Mode",
    options=["🎛️ Manual Slider Controls", "📊 SCADA Dataset Mode"],
    index=0,
    help="Switch seamlessly between manual slider parameters and uploaded dynamic SCADA time-series CSV datasets.",
)

# Session state initialization for SCADA data
if "scada_df" not in st.session_state:
    st.session_state.scada_df = None

if input_mode == "📊 SCADA Dataset Mode":
    st.sidebar.markdown("### SCADA Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload SCADA Time-Series CSV",
        type=["csv"],
        help="Upload CSV containing timestamp, flow Q, influent COD, NH4-N, and temperature.",
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Sample Dataset Template**")
    sample_csv_bytes = generate_sample_scada_csv()
    st.sidebar.download_button(
        "📥 Download Sample SCADA CSV",
        data=sample_csv_bytes,
        file_name="sample_scada_influent.csv",
        mime="text/csv",
        help="Download a realistic synthetic 7-day SCADA dataset with diurnal loading patterns and missing values.",
    )

    with st.sidebar.expander("Reactor Configuration", expanded=True):
        volume = st.slider("Reactor Volume ($V$, m³)", 200.0, 5000.0, 1000.0, 50.0)
        kla = st.slider("Oxygen Transfer Coefficient ($K_L a$, 1/d)", 20.0, 300.0, 120.0, 10.0)

    if uploaded_file is not None:
        scada_df, errors, warnings = process_scada_file(uploaded_file)
        if errors:
            for err in errors:
                st.sidebar.error(err)
            st.session_state.scada_df = None
            flow_rate, influent_cod, influent_nh4, temp, srt = 1000.0, 150.0, 35.0, 20.0, 5.0
        else:
            for warn in warnings:
                st.sidebar.warning(warn)
            st.session_state.scada_df = scada_df
            flow_rate = float(scada_df["flow_rate"].mean())
            influent_cod = float(scada_df["influent_cod"].mean())
            influent_nh4 = float(scada_df["influent_nh4"].mean())
            temp = float(scada_df["temperature"].mean())
            srt = float(max(scada_df["time_days"].max(), 1.0))
            st.sidebar.success(f"✅ SCADA Loaded Successfully!\n• Data Points: {len(scada_df)}\n• Duration: {srt:.1f} days\n• Mean Q: {flow_rate:.0f} m³/d")
    else:
        st.session_state.scada_df = None
        flow_rate, influent_cod, influent_nh4, temp, srt = 1000.0, 150.0, 35.0, 20.0, 5.0
        st.sidebar.info("ℹ️ Please upload a SCADA CSV file or download the sample dataset above to start.")

else:
    st.session_state.scada_df = None
    with st.sidebar.expander("Reactor & Hydraulics", expanded=True):
        flow_rate = st.slider("Influent Flow Rate ($Q$, m³/d)", 200.0, 5000.0, 1000.0, 50.0)
        volume = st.slider("Reactor Volume ($V$, m³)", 200.0, 5000.0, 1000.0, 50.0)
        srt = st.slider("Simulation Duration (Days)", 1.0, 30.0, 5.0, 0.5)
        temp = st.slider("Temperature (°C)", 10.0, 35.0, 20.0, 0.5)

    with st.sidebar.expander("Influent Characteristics", expanded=True):
        influent_cod = st.slider("Influent Readily Biodegradable COD ($S_S$, g/m³)", 50.0, 500.0, 150.0, 10.0)
        influent_nh4 = st.slider("Influent Ammonium ($S_{NH}$, g/m³)", 10.0, 100.0, 35.0, 2.0)
        kla = st.slider("Oxygen Transfer Coefficient ($K_L a$, 1/d)", 20.0, 300.0, 120.0, 10.0)

with st.sidebar.expander("PINN Surrogate Training", expanded=False):
    pinn_epochs = st.slider("Training Epochs", 50, 500, 150, 25)
    hidden_units = st.selectbox("Hidden Layer Neurons", [32, 64, 128, 256], index=1)
    physics_w = st.slider("Physics Loss Weight ($\lambda$)", 0.0, 0.1, 0.01, 0.005)
    train_button = st.button("🚀 Train PINN Surrogate", type="primary")

# Compute Derived Hydraulics
hrt_val = volume / max(flow_rate, 1.0)

# Main Title Area
st.markdown('<div class="main-header">💧 AquaPINN Digital Twin</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Physics-Informed Neural Network (PINN) & ASM1 Bioreactor Simulation Platform for Wastewater Treatment</div>',
    unsafe_allow_html=True,
)

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Hydraulic Retention Time", f"{hrt_val:.2f} days", f"Q = {flow_rate:.0f} m³/d")
with col2:
    st.metric("Aeration KLa", f"{kla:.1f} d⁻¹", f"DO Sat = 8.5 g/m³")
with col3:
    st.metric("Operating Temp", f"{temp:.1f} °C", f"Arrhenius scaled")
with col4:
    st.metric("Bioreactor Volume", f"{volume:.0f} m³", f"CSTR Configuration")

st.divider()

# Run ASM1 ODE Simulation
@st.cache_data
def run_simulation(q, v, k, ss_in, snh_in, days, t_c):
    influent_override = {"S_S": ss_in, "S_NH": snh_in}
    return simulate_asm1(
        days=days,
        num_points=300,
        q_in=q,
        volume=v,
        kla=k,
        influent_override=influent_override,
        temp_celsius=t_c,
    )

sim_df = run_simulation(flow_rate, volume, kla, influent_cod, influent_nh4, srt, temp)

# Session state management for PINN training
if "pinn_model" not in st.session_state:
    st.session_state.pinn_model = None
    st.session_state.train_history = None

if train_button:
    with st.spinner("Training Physics-Informed Neural Network (PINN) surrogate on ASM1 dynamics..."):
        model, history = train_pinn(
            days=srt,
            num_samples=300,
            epochs=pinn_epochs,
            hidden_dim=hidden_units,
            physics_weight=physics_w,
        )
        st.session_state.pinn_model = model
        st.session_state.train_history = history
    st.success("✅ PINN Surrogate model successfully trained with mass conservation constraints!")

# Tabs Layout Setup
has_scada = (input_mode == "📊 SCADA Dataset Mode" and st.session_state.scada_df is not None)
if has_scada:
    scada_tab, tab1, tab2, tab3, tab4 = st.tabs([
        "📁 SCADA Time-Series Data",
        "📈 ASM1 ODE Simulation",
        "🤖 PINN vs ODE Surrogate",
        "⚖️ Mass Balance & Physics Loss",
        "💡 Process Control & Advisor",
    ])
    
    with scada_tab:
        st.subheader("Uploaded SCADA Time-Series Dataset")
        st.markdown("Interpolated and validated operational time-series data from plant SCADA systems.")
        
        scada_df = st.session_state.scada_df
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            st.metric("Total Records", f"{len(scada_df)}")
        with s_col2:
            st.metric("Mean Influent Flow ($Q$)", f"{scada_df['flow_rate'].mean():.1f} m³/d")
        with s_col3:
            st.metric("Mean Influent COD ($S_S$)", f"{scada_df['influent_cod'].mean():.1f} g/m³")
        with s_col4:
            st.metric("Mean Temperature", f"{scada_df['temperature'].mean():.1f} °C")

        # Plot SCADA time series
        fig_scada = go.Figure()
        fig_scada.add_trace(go.Scatter(x=scada_df["timestamp"], y=scada_df["flow_rate"], name="Flow Q (m³/d)", line=dict(color="#2563EB")))
        fig_scada.add_trace(go.Scatter(x=scada_df["timestamp"], y=scada_df["influent_cod"], name="Influent COD (g/m³)", yaxis="y2", line=dict(color="#DC2626")))
        fig_scada.update_layout(
            title="SCADA Influent Flow and COD Time Series",
            xaxis_title="Timestamp",
            yaxis=dict(title="Flow Rate (m³/d)", titlefont=dict(color="#2563EB"), tickfont=dict(color="#2563EB")),
            yaxis2=dict(title="COD (g/m³)", titlefont=dict(color="#DC2626"), tickfont=dict(color="#DC2626"), overlaying="y", side="right"),
            template="plotly_white",
            height=400,
            hovermode="x unified",
        )
        st.plotly_chart(fig_scada, use_container_width=True)

        with st.expander("🔍 View Processed SCADA Data Table"):
            st.dataframe(scada_df, use_container_width=True)
else:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 ASM1 ODE Simulation",
        "🤖 PINN vs ODE Surrogate",
        "⚖️ Mass Balance & Physics Loss",
        "💡 Process Control & Advisor",
    ])

with tab1:
    st.subheader("Benchmark ASM1 ODE Dynamic Simulation")
    st.markdown("Dynamic evolution of activated sludge state variables in the continuous stirred-tank reactor (CSTR).")

    selected_states = st.multiselect(
        "Select State Variables to Plot",
        options=STATE_NAMES,
        default=["S_S", "S_O", "S_NO", "S_NH", "X_BH"],
    )

    if selected_states:
        fig_ode = go.Figure()
        for state in selected_states:
            unit = STATE_UNITS.get(state, "")
            fig_ode.add_trace(
                go.Scatter(
                    x=sim_df["time"],
                    y=sim_df[state],
                    mode="lines",
                    name=f"{state} ({unit})",
                    line=dict(width=2.5),
                )
            )
        fig_ode.update_layout(
            title="Biochemical State Concentrations over Time",
            xaxis_title="Time (days)",
            yaxis_title="Concentration (g/m³ or mol/m³)",
            hovermode="x unified",
            template="plotly_white",
            height=450,
        )
        st.plotly_chart(fig_ode, use_container_width=True)

    # Effluent Performance Summary Metrics
    st.markdown("### Effluent Quality Indicators")
    e_col1, e_col2, e_col3, e_col4 = st.columns(4)
    final_row = sim_df.iloc[-1]
    with e_col1:
        st.metric("Final Effluent COD", f"{final_row['COD_total']:.1f} g/m³", f"{final_row['COD_removal_pct']:.1f}% removal")
    with e_col2:
        st.metric("Final Effluent NH₄-N", f"{final_row['S_NH']:.2f} g/m³", f"{final_row['NH_removal_pct']:.1f}% nitrification")
    with e_col3:
        st.metric("Dissolved Oxygen (DO)", f"{final_row['S_O']:.2f} g O₂/m³", "Aerobic status")
    with e_col4:
        st.metric("Total Nitrogen (TN)", f"{final_row['TN']:.2f} g N/m³", f"NO₃: {final_row['S_NO']:.1f}")

with tab2:
    st.subheader("PINN Surrogate vs. First-Principles ODE Solution")
    st.markdown("Compare the real-time inference predictions of the PyTorch Physics-Informed Neural Network against numerical ODE integration.")

    if st.session_state.pinn_model is None:
        st.info("ℹ️ Train the PINN surrogate model using the sidebar control panel (**Train PINN Surrogate**) to view surrogate predictions.")
        # Show mock or un-trained comparison notice
        if st.button("Train Quick PINN Now"):
            with st.spinner("Quick training in progress..."):
                model, history = train_pinn(days=srt, num_samples=200, epochs=80, hidden_dim=64, physics_weight=0.01)
                st.session_state.pinn_model = model
                st.session_state.train_history = history
            st.rerun()
    else:
        model = st.session_state.pinn_model
        model.eval()
        
        # Prepare evaluation inputs
        t_vals = sim_df["time"].values
        d_vals = np.full_like(t_vals, flow_rate / volume)
        kla_vals = np.full_like(t_vals, kla)
        ss_vals = np.full_like(t_vals, influent_cod)
        snh_vals = np.full_like(t_vals, influent_nh4)
        
        X_eval = np.stack([t_vals, d_vals, kla_vals, ss_vals, snh_vals], axis=-1)
        with torch.no_grad():
            preds_tensor = model(torch.tensor(X_eval, dtype=torch.float32))
            preds_np = preds_tensor.numpy()

        compare_state = st.selectbox("Select State for PINN vs ODE Comparison", options=STATE_NAMES, index=1)
        state_idx = STATE_NAMES.index(compare_state)

        fig_comp = go.Figure()
        fig_comp.add_trace(
            go.Scatter(
                x=sim_df["time"],
                y=sim_df[compare_state],
                mode="lines",
                name=f"ASM1 ODE (Ground Truth)",
                line=dict(color="#2563EB", width=3),
            )
        )
        fig_comp.add_trace(
            go.Scatter(
                x=sim_df["time"],
                y=preds_np[:, state_idx],
                mode="lines",
                name=f"PINN Surrogate Prediction",
                line=dict(color="#DC2626", width=2, dash="dash"),
            )
        )
        fig_comp.update_layout(
            title=f"Comparison: {compare_state} ({STATE_UNITS.get(compare_state, '')})",
            xaxis_title="Time (days)",
            yaxis_title="Concentration",
            hovermode="x unified",
            template="plotly_white",
            height=430,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Accuracy Metrics
        ode_vals = sim_df[compare_state].values
        pinn_vals = preds_np[:, state_idx]
        rmse = np.sqrt(np.mean((ode_vals - pinn_vals) ** 2))
        mae = np.mean(np.abs(ode_vals - pinn_vals))
        
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f} {STATE_UNITS.get(compare_state, '')}")
        with m2:
            st.metric("Mean Absolute Error (MAE)", f"{mae:.4f} {STATE_UNITS.get(compare_state, '')}")

with tab3:
    st.subheader("Physics Loss & Mass Conservation Tracking")
    st.markdown("Monitoring mass-balance conservation residuals and PINN training convergence history.")

    if st.session_state.train_history is not None:
        history_df = pd.DataFrame({
            "Epoch": range(1, len(st.session_state.train_history) + 1),
            "Total Loss": st.session_state.train_history,
        })
        fig_loss = px.line(
            history_df,
            x="Epoch",
            y="Total Loss",
            title="PINN Training Convergence (Data Loss + Physics Conservation Loss)",
            template="plotly_white",
        )
        fig_loss.update_layout(height=400)
        st.plotly_chart(fig_loss, use_container_width=True)
    else:
        st.info("ℹ️ Train the PINN model to visualize training convergence and physics loss curves.")

    st.markdown("### Mass Conservation Analysis")
    st.markdown(
        """
        The ASM1 model strictly obeys mass conservation laws for Chemical Oxygen Demand (COD), Total Nitrogen (TN),
        and electrical charge (Alkalinity). The PINN surrogate embeds these governing differential equations
        directly into the neural network loss function via automatic differentiation ($\mathcal{L}_{physics} = ||\frac{dy}{dt} - f(y,u)||^2$).
        """
    )
    
    # COD balance check
    cod_in = influent_cod + 30.0 + 50.0 + 200.0  # approximate total influent COD
    cod_out = final_row["COD_total"]
    st.progress(min(max(cod_out / max(cod_in, 1.0), 0.0), 1.0), text=f"Reactor COD Mass Balance Ratio: Effluent / Influent = {cod_out/cod_in:.2f}")

with tab4:
    st.subheader("Digital Twin Process Control & Advisor")
    st.markdown("Intelligent recommendations for wastewater treatment plant operations based on simulated effluent quality.")

    # Rule-based expert advisor
    do_val = final_row["S_O"]
    nh4_val = final_row["S_NH"]
    cod_val = final_row["COD_total"]
    no3_val = final_row["S_NO"]

    recommendations = []
    
    if do_val < 2.0:
        recommendations.append(("⚠️ Low Dissolved Oxygen Warning", f"Effluent DO is {do_val:.2f} g O₂/m³ (< 2.0). Increase aeration $K_L a$ to prevent oxygen limitation during nitrification."))
    elif do_val > 4.0:
        recommendations.append(("💡 Energy Optimization Notice", f"Effluent DO is {do_val:.2f} g O₂/m³ (> 4.0). Consider reducing aeration blower speed to save energy while maintaining nitrification."))
    else:
        recommendations.append(("✅ Optimal Aeration", f"Effluent DO is in the ideal range ({do_val:.2f} g O₂/m³)."))

    if nh4_val > 5.0:
        recommendations.append(("⚠️ Elevated Effluent Ammonium", f"Ammonium concentration is {nh4_val:.2f} g/m³ (> 5.0). Nitrification is incomplete. Increase SRT or aeration."))
    else:
        recommendations.append(("✅ Excellent Nitrification", f"Effluent ammonium is well controlled ({nh4_val:.2f} g/m³)."))

    if cod_val > 60.0:
        recommendations.append(("⚠️ High Effluent COD", f"Total effluent COD is {cod_val:.2f} g/m³ (> 60.0). Check biomass activity ($X_{{BH}}$) and hydraulic overloading."))
    else:
        recommendations.append(("✅ High Organic Removal", f"COD removal efficiency is robust ({final_row['COD_removal_pct']:.1f}%)."))

    for title, desc in recommendations:
        with st.expander(title, expanded=True):
            st.write(desc)

    st.markdown("### Operational Recommendations Summary")
    st.info("AquaPINN digital twin suggests maintaining current flow rate and adjusting $K_L a$ dynamically based on real-time influent surge detection.")
