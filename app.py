import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Spin Coating Simulator", layout="wide")

st.title("Spin Coating Simulator")
st.caption("EBP Model + Meyerhofer Model evaporation and viscosity increase")

st.sidebar.header("Input Parameters")

rpm = st.sidebar.slider("RPM", 500, 3000, 3000, 100)
h_0 = st.sidebar.number_input("Initial Thickness h₀ (μm)", value=100.0, min_value=1.0)
eta_0 = st.sidebar.number_input("Initial Viscosity η₀ (Pa·s)", value=0.05, min_value=0.001)
rho = st.sidebar.number_input("Density ρ (kg/m³)", value=1000.0, min_value=1.0)
E = st.sidebar.number_input("Solvent Evaporation Rate E (μm/s)", value=0.01, min_value=0.0)
k = st.sidebar.number_input("Viscosity Growth Rate k (1/s)", value=0.03, min_value=0.0)
t = st.sidebar.number_input("Simulation Time (s)", value=60.0, min_value=1.0)
dt = st.sidebar.number_input("Time Step Δt (s)", value=0.05, min_value=0.001)

st.sidebar.markdown("---")
st.sidebar.header("Radial Profile / Uniformity")

R_mm = st.sidebar.number_input("Wafer Radius R (mm)", value=50.0, min_value=1.0)
edge_bead_width = st.sidebar.number_input("Edge Bead Width w_edge (mm)", value=5.0, min_value=0.1)
base_edge_bead = st.sidebar.slider("Initial Edge Bead Strength α₀", 0.0, 0.30, 0.05, 0.01)
edge_relaxation_rate = st.sidebar.number_input("Edge Bead Relaxation Rate β (1/s)", value=0.01, min_value=0.0)

uniformity_spec = st.sidebar.number_input("Uniformity Spec (±%)", value=2.0, min_value=0.1)
eta_gel = st.sidebar.number_input("Gel Viscosity η_gel (Pa·s)", value=0.30, min_value=0.001)

st.sidebar.markdown("---")
st.sidebar.write("Parameter Study")

rpm_values = st.sidebar.multiselect(
    "RPM cases",
    [500, 1000, 1500, 2000, 2500, 3000],
    default=[1000, 2000, 3000],
)

eta_values = st.sidebar.multiselect(
    "Viscosity cases η₀ (Pa·s)",
    [0.03, 0.05, 0.10, 0.20],
    default=[0.03, 0.05, 0.10],
)

E_values = st.sidebar.multiselect(
    "Evaporation cases E (μm/s)",
    [0.0, 0.005, 0.01, 0.03, 0.05],
    default=[0.0, 0.01, 0.03],
)


def simulate_spin_coating(
    rpm,
    h_0,
    eta_0,
    rho,
    E,
    k,
    t,
    dt,
    use_evaporation=True,
    use_viscosity_growth=True,
):
    omega = rpm * 2 * np.pi / 60
    time = np.arange(0, t + dt, dt)

    h_m = np.zeros_like(time)
    h_m[0] = h_0 * 1e-6

    eta_arr = np.zeros_like(time)
    dhdt_arr = np.zeros_like(time)

    E_m_s = E * 1e-6 if use_evaporation else 0.0

    for i in range(len(time) - 1):
        eta = eta_0 * np.exp(k * time[i]) if use_viscosity_growth else eta_0
        eta_arr[i] = eta

        dhdt = -(2 * rho * omega**2 / (3 * eta)) * h_m[i]**3 - E_m_s
        dhdt_arr[i] = dhdt

        h_m[i + 1] = max(h_m[i] + dhdt * dt, 0)

    eta_arr[-1] = eta_0 * np.exp(k * time[-1]) if use_viscosity_growth else eta_0
    dhdt_arr[-1] = dhdt_arr[-2]

    return pd.DataFrame({
        "Time (s)": time,
        "Thickness (μm)": h_m * 1e6,
        "Viscosity η (Pa·s)": eta_arr,
        "Evaporation Rate E (μm/s)": E if use_evaporation else 0.0,
        "dh/dt (μm/s)": dhdt_arr * 1e6,
    })


def calculate_alpha_t(alpha_0, beta, time_value, rpm, eta_0):
    rpm_ref = 2000.0
    eta_ref = 0.05

    rpm_factor = (rpm_ref / rpm) ** 0.5
    eta_factor = (eta_0 / eta_ref) ** 0.5
    time_factor = np.exp(-beta * time_value)

    return alpha_0 * rpm_factor * eta_factor * time_factor


def calculate_radial_profile(thickness, R_mm, edge_bead_width, alpha_t):
    r = np.linspace(0, R_mm, 300)

    distance_from_edge = R_mm - r
    edge_shape = np.exp(-distance_from_edge / edge_bead_width)

    h_r = thickness * (1 + alpha_t * edge_shape)

    h_max = np.max(h_r)
    h_min = np.min(h_r)
    h_avg = np.mean(h_r)

    if h_avg <= 1e-6:
        uniformity = np.nan
    else:
        uniformity = (h_max - h_min) / (2 * h_avg) * 100

    return r, h_r, uniformity, h_max, h_min, h_avg


def calculate_t_gel(eta_0, eta_gel, k):
    if k <= 0:
        return None
    if eta_gel <= eta_0:
        return 0
    return np.log(eta_gel / eta_0) / k


def format_uniformity(value):
    if np.isnan(value):
        return "Film depleted"
    return f"±{value:.3f} %"


def check_uniformity_pass(uniformity_value, final_thickness, spec):
    if final_thickness <= 1e-6:
        return False
    if np.isnan(uniformity_value):
        return False
    return uniformity_value <= spec


def evaluate_condition(rpm_case, eta_case):
    df = simulate_spin_coating(
        rpm_case, h_0, eta_case, rho, E, k, t, dt,
        use_evaporation=True,
        use_viscosity_growth=True,
    )

    final_thickness = df["Thickness (μm)"].iloc[-1]

    alpha_case = calculate_alpha_t(
        base_edge_bead,
        edge_relaxation_rate,
        t,
        rpm_case,
        eta_case,
    )

    _, _, uniformity_case, _, _, _ = calculate_radial_profile(
        final_thickness,
        R_mm,
        edge_bead_width,
        alpha_case,
    )

    omega_case = rpm_case * 2 * np.pi / 60
    result = "PASS" if check_uniformity_pass(uniformity_case, final_thickness, uniformity_spec) else "FAIL"

    return {
        "RPM": rpm_case,
        "ω (rad/s)": omega_case,
        "η₀ (Pa·s)": eta_case,
        "Final Thickness (μm)": final_thickness,
        "Edge Bead Strength α(t_end)": alpha_case,
        "Radial Uniformity (±%)": uniformity_case if not np.isnan(uniformity_case) else None,
        "Result": result,
    }


df_ebp = simulate_spin_coating(
    rpm, h_0, eta_0, rho, 0.0, 0.0, t, dt,
    use_evaporation=False,
    use_viscosity_growth=False,
)

df_meyer = simulate_spin_coating(
    rpm, h_0, eta_0, rho, E, k, t, dt,
    use_evaporation=True,
    use_viscosity_growth=True,
)

final_ebp = df_ebp["Thickness (μm)"].iloc[-1]
final_meyer = df_meyer["Thickness (μm)"].iloc[-1]

alpha_final = calculate_alpha_t(
    base_edge_bead,
    edge_relaxation_rate,
    t,
    rpm,
    eta_0,
)

r_profile, h_profile, uniformity, h_max, h_min, h_avg = calculate_radial_profile(
    final_meyer,
    R_mm,
    edge_bead_width,
    alpha_final,
)

t_gel = calculate_t_gel(eta_0, eta_gel, k)
uniformity_pass = check_uniformity_pass(uniformity, final_meyer, uniformity_spec)

col1, col2, col3 = st.columns(3)
col1.metric("Final Thickness: EBP", f"{final_ebp:.3f} μm")
col2.metric("Final Thickness: Meyerhofer Model", f"{final_meyer:.3f} μm")
col3.metric("Thickness Difference", f"{final_meyer - final_ebp:.3f} μm")

col4, col5, col6 = st.columns(3)
col4.metric("Input Evaporation Rate E", f"{E:.4f} μm/s")
col5.metric("Radial Uniformity", format_uniformity(uniformity))
col6.metric("Spec Result", "PASS" if uniformity_pass else "FAIL")

if final_meyer <= 1e-6:
    st.warning("The Meyerhofer Model predicts film depletion. Uniformity is not meaningful in this condition.")

if t_gel is None:
    st.info("t_gel is not defined because k = 0. Viscosity does not increase with time.")
else:
    st.info(f"Predicted gel time t_gel = {t_gel:.2f} s, based on η(t)=η₀e^(kt).")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Model Comparison",
    "RPM Effect",
    "Viscosity Effect",
    "Evaporation Effect",
    "Radial Evolution",
    "Radial Uniformity",
    "Challenge Mode",
    "Data & Insight",
])

with tab1:
    st.subheader("EBP Model vs Meyerhofer Model")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df_ebp["Time (s)"],
        df_ebp["Thickness (μm)"],
        label="EBP: no evaporation, constant viscosity",
    )
    ax.plot(
        df_meyer["Time (s)"],
        df_meyer["Thickness (μm)"],
        label="Meyerhofer Model: evaporation + η(t)",
        color="red",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Film Thickness (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.write(
        "The EBP Model considers centrifugal thinning only. "
        "The Meyerhofer Model includes solvent evaporation and viscosity increase."
    )

with tab2:
    st.subheader("Effect of Spin Speed on Meyerhofer Model")

    fig, ax = plt.subplots(figsize=(8, 5))
    summary = []

    for r_case in rpm_values:
        df = simulate_spin_coating(r_case, h_0, eta_0, rho, E, k, t, dt)
        ax.plot(df["Time (s)"], df["Thickness (μm)"], label=f"{r_case} RPM")
        summary.append([r_case, E, df["Thickness (μm)"].iloc[-1]])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Film Thickness of Meyerhofer Model (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.dataframe(
        pd.DataFrame(
            summary,
            columns=["RPM", "Input E (μm/s)", "Final Thickness (μm)"],
        )
    )

with tab3:
    st.subheader("Effect of Initial Viscosity on Meyerhofer Model")

    fig, ax = plt.subplots(figsize=(8, 5))
    summary = []

    for eta_case in eta_values:
        df = simulate_spin_coating(rpm, h_0, eta_case, rho, E, k, t, dt)
        ax.plot(df["Time (s)"], df["Thickness (μm)"], label=f"η₀={eta_case} Pa·s")
        summary.append([eta_case, df["Thickness (μm)"].iloc[-1]])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Film Thickness of Meyerhofer Model (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.dataframe(
        pd.DataFrame(
            summary,
            columns=["Initial Viscosity η₀ (Pa·s)", "Final Thickness (μm)"],
        )
    )

with tab4:
    st.subheader("Effect of Solvent Evaporation Rate on Meyerhofer Model")

    fig, ax = plt.subplots(figsize=(8, 5))
    summary = []

    for E_case in E_values:
        df = simulate_spin_coating(rpm, h_0, eta_0, rho, E_case, k, t, dt)
        ax.plot(df["Time (s)"], df["Thickness (μm)"], label=f"E={E_case} μm/s")
        summary.append([E_case, df["Thickness (μm)"].iloc[-1]])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Film Thickness of Meyerhofer Model (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.dataframe(
        pd.DataFrame(
            summary,
            columns=["Evaporation Rate E (μm/s)", "Final Thickness (μm)"],
        )
    )

with tab5:
    st.subheader("Radial Evolution of h(r,t)")

    selected_time = st.slider(
        "Select Time for Radial Profile (s)",
        min_value=0.0,
        max_value=float(t),
        value=float(t),
        step=float(dt),
    )

    idx = (df_meyer["Time (s)"] - selected_time).abs().idxmin()
    selected_time_actual = df_meyer.loc[idx, "Time (s)"]
    selected_thickness = df_meyer.loc[idx, "Thickness (μm)"]

    alpha_selected = calculate_alpha_t(
        base_edge_bead,
        edge_relaxation_rate,
        selected_time_actual,
        rpm,
        eta_0,
    )

    r_t, h_rt, u_t, hmax_t, hmin_t, havg_t = calculate_radial_profile(
        selected_thickness,
        R_mm,
        edge_bead_width,
        alpha_selected,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_t, h_rt, color="red", label=f"h(r,t) at t = {selected_time_actual:.2f} s")
    ax.axhline(havg_t, linestyle="--", label="Average thickness")
    ax.set_xlabel("Radial Position r (mm)")
    ax.set_ylabel("Film Thickness h(r,t) (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Selected Time", f"{selected_time_actual:.2f} s")
    col_b.metric("Edge Bead Strength α(t)", f"{alpha_selected:.4f}")
    col_c.metric("Radial Uniformity", format_uniformity(u_t))

    st.write(
        "This tab visualizes the time-dependent radial thickness profile h(r,t). "
        "The edge bead strength changes with RPM, initial viscosity, and time."
    )

with tab6:
    st.subheader("Final Radial Thickness Profile")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_profile, h_profile, color="red", label="Final radial thickness")
    ax.axhline(h_avg, linestyle="--", label="Average thickness")
    ax.set_xlabel("Radial Position r (mm)")
    ax.set_ylabel("Final Film Thickness h(r) (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    uniformity_df = pd.DataFrame({
        "Metric": [
            "Minimum Thickness",
            "Maximum Thickness",
            "Average Thickness",
            "Radial Uniformity",
            "Uniformity Spec",
            "Wafer Radius",
            "Edge Bead Width",
            "Initial Edge Bead Strength",
            "Final Edge Bead Strength",
            "Edge Bead Relaxation Rate",
            "Result",
        ],
        "Value": [
            f"{h_min:.4f} μm",
            f"{h_max:.4f} μm",
            f"{h_avg:.4f} μm",
            format_uniformity(uniformity),
            f"±{uniformity_spec:.2f} %",
            f"{R_mm:.2f} mm",
            f"{edge_bead_width:.2f} mm",
            f"{base_edge_bead:.4f}",
            f"{alpha_final:.4f}",
            f"{edge_relaxation_rate:.4f} 1/s",
            "PASS" if uniformity_pass else "FAIL",
        ],
    })

    st.subheader("Uniformity Evaluation")
    st.dataframe(uniformity_df)

with tab7:
    st.subheader("Challenge Mode: Find RPM and η₀ Conditions for ±2% Uniformity")

    challenge_rpm_values = st.multiselect(
        "Challenge RPM candidates",
        [1000, 1500, 2000, 2500, 3000],
        default=[1000, 1500, 2000, 2500, 3000],
    )

    challenge_eta_values = st.multiselect(
        "Challenge η₀ candidates (Pa·s)",
        [0.03, 0.04, 0.05, 0.07, 0.10],
        default=[0.03, 0.04, 0.05, 0.07, 0.10],
    )

    challenge_results = []

    for r_case in challenge_rpm_values:
        for eta_case in challenge_eta_values:
            challenge_results.append(evaluate_condition(r_case, eta_case))

    challenge_df = pd.DataFrame(challenge_results)

    if len(challenge_df) > 0:
        st.subheader("All Tested Conditions")
        st.dataframe(challenge_df)

        feasible_df = challenge_df[challenge_df["Result"] == "PASS"].copy()

        st.subheader("Feasible Conditions Satisfying Uniformity Spec")

        if len(feasible_df) > 0:
            feasible_df = feasible_df.sort_values(
                by=["Radial Uniformity (±%)", "Final Thickness (μm)"],
                ascending=[True, False],
            )
            st.dataframe(feasible_df)

            best = feasible_df.iloc[0]

            st.success(
                f"Best condition: RPM = {best['RPM']}, "
                f"ω = {best['ω (rad/s)']:.2f} rad/s, "
                f"η₀ = {best['η₀ (Pa·s)']:.3f} Pa·s, "
                f"Uniformity = ±{best['Radial Uniformity (±%)']:.3f}%"
            )
        else:
            st.warning("No tested condition satisfies the uniformity specification.")
    else:
        st.warning("Select at least one RPM and one η₀ candidate.")

with tab8:
    st.subheader("Simulation Data")

    st.dataframe(df_meyer)

    st.subheader("Physical Interpretation")

    st.markdown(
        """
        - Higher RPM increases centrifugal thinning, so the film thickness decreases faster.
        - Higher RPM also reduces the simplified edge bead factor in the radial profile model.
        - Higher initial viscosity suppresses radial flow, resulting in a thicker final film.
        - Higher initial viscosity increases the simplified edge bead factor.
        - Higher solvent evaporation rate directly decreases the film thickness.
        - Film depletion is treated as a failed condition because radial uniformity is not physically meaningful when the film thickness becomes zero.
        - Radial evolution h(r,t) is visualized using a time slider.
        - Challenge Mode searches RPM and η₀ combinations satisfying the ±2% uniformity spec.
        """
    )

    st.subheader("Governing Equation Used")

    st.latex(r"""
    \frac{dh}{dt}
    =
    -\frac{2\rho\omega^2}{3\eta(t)}h^3
    -
    E
    """)

    st.latex(r"""
    \eta(t)=\eta_0 e^{kt}
    """)

    st.latex(r"""
    t_{gel}
    =
    \frac{1}{k}
    \ln\left(\frac{\eta_{gel}}{\eta_0}\right)
    """)

    st.subheader("Simplified Radial Profile Used")

    st.latex(r"""
    h(r,t)
    =
    h(t)
    \left[
    1+\alpha(t)
    \exp\left(-\frac{R-r}{w_{edge}}\right)
    \right]
    """)

    st.latex(r"""
    \alpha(t)
    =
    \alpha_0
    \left(\frac{RPM_{ref}}{RPM}\right)^{1/2}
    \left(\frac{\eta_0}{\eta_{ref}}\right)^{1/2}
    e^{-\beta t}
    """)

    st.latex(r"""
    Uniformity(\%)
    =
    \pm
    \frac{h_{max}-h_{min}}{2h_{avg}}
    \times 100
    """)
