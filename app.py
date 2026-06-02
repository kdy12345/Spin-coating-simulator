import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Spin Coating Simulator", layout="wide")

st.title("Spin Coating Simulator")
st.caption("EBP Model + Meyerhofer Model evaporation and viscosity increase")

# -----------------------------
# Sidebar input
# -----------------------------
st.sidebar.header("Input Parameters")

rpm = st.sidebar.slider("RPM", 500, 6000, 3000, 100)
h_0 = st.sidebar.number_input("Initial Thickness h₀ (μm)", value=100.0, min_value=1.0)
mu_0 = st.sidebar.number_input("Initial Viscosity μ₀ (Pa·s)", value=0.05, min_value=0.001)
rho = st.sidebar.number_input("Density ρ (kg/m³)", value=1000.0, min_value=1.0)
E = st.sidebar.number_input("Evaporation Rate E (μm/s)", value=0.01, min_value=0.0)
k = st.sidebar.number_input("Viscosity Growth Rate k (1/s)", value=0.03, min_value=0.0)
t = st.sidebar.number_input("Simulation Time (s)", value=60.0, min_value=1.0)
dt = st.sidebar.number_input("Time Step Δt (s)", value=0.05, min_value=0.001)

st.sidebar.markdown("---")
st.sidebar.header("Radial Profile / Uniformity")

R_mm = st.sidebar.number_input("Wafer Radius R (mm)", value=50.0, min_value=1.0)
# 과제 요구사항에 맞게 초기 뭉침 기본값을 0.05로 세팅하여 합격/불합격을 유도합니다.
edge_bead_strength = st.sidebar.slider("Base Edge Bead Strength α", 0.0, 0.10, 0.05, 0.005)
edge_exponent = st.sidebar.slider("Edge Bead Exponent n", 2, 12, 6, 1)
rpm_sensitivity = st.sidebar.slider("RPM Sensitivity m", 0.0, 2.0, 0.5, 0.1)
uniformity_spec = st.sidebar.number_input("Uniformity Spec (%)", value=2.0, min_value=0.1)
mu_gel = st.sidebar.number_input("Gel Viscosity μ_gel (Pa·s)", value=0.30, min_value=0.001)

st.sidebar.markdown("---")
st.sidebar.header("Parameter Study")

rpm_values = st.sidebar.multiselect(
    "RPM cases",
    [500, 1000, 2000, 3000],
    default=[1000, 2000, 3000],
)

mu_values = st.sidebar.multiselect(
    "Viscosity cases (Pa·s)",
    [0.03, 0.05, 0.10, 0.20],
    default=[0.03, 0.05, 0.10],
)

E_values = st.sidebar.multiselect(
    "Evaporation cases (μm/s)",
    [0.0, 0.01, 0.05, 0.10],
    default=[0.0, 0.01, 0.05],
)

# -----------------------------
# Core functions
# -----------------------------
def simulate_spin_coating(
    rpm,
    h_0,
    mu_0,
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

    mu_arr = np.zeros_like(time)
    dhdt_arr = np.zeros_like(time)

    E_m_s = E * 1e-6 if use_evaporation else 0.0

    for i in range(len(time) - 1):
        if use_viscosity_growth:
            mu = mu_0 * np.exp(k * time[i])
        else:
            mu = mu_0

        mu_arr[i] = mu

        dhdt = -(2 * rho * omega**2 / (3 * mu)) * h_m[i]**3 - E_m_s
        dhdt_arr[i] = dhdt

        h_m[i + 1] = max(h_m[i] + dhdt * dt, 0)

    mu_arr[-1] = mu_0 * np.exp(k * time[-1]) if use_viscosity_growth else mu_0
    dhdt_arr[-1] = dhdt_arr[-2]

    df = pd.DataFrame({
        "Time (s)": time,
        "Thickness (μm)": h_m * 1e6,
        "Viscosity (Pa·s)": mu_arr,
        "dh/dt (μm/s)": dhdt_arr * 1e6,
    })

    return df


def calculate_effective_alpha(alpha, rpm, rpm_ref=3000, m=0.5):
    # 과제 원본 수식 그대로 복구: RPM이 올라갈수록 원심력에 의해 Edge bead 강도가 감소하는 기본 모델
    if rpm == 0:
        return alpha
    return alpha * (rpm_ref / rpm) ** m


def calculate_radial_profile(final_thickness, R_mm, alpha_eff, n):
    r = np.linspace(0, R_mm, 200)
    normalized_r = r / R_mm

    h_r = final_thickness * (1 + alpha_eff * normalized_r**n)

    h_max = np.max(h_r)
    h_min = np.min(h_r)
    h_avg = np.mean(h_r)

    if h_avg > 0:
        # [정확한 요구사항 반영] 과제 스펙(±2%)에 정합하도록 분모에 2를 곱해 균일도 편차 계산
        uniformity = (h_max - h_min) / (2 * h_avg) * 100
    else:
        uniformity = 0

    return r, h_r, uniformity, h_max, h_min, h_avg


def calculate_t_gel(mu_0, mu_gel, k):
    if k <= 0:
        return None
    if mu_gel <= mu_0:
        return 0
    return np.log(mu_gel / mu_0) / k


# -----------------------------
# Main simulation
# -----------------------------
df_ebp = simulate_spin_coating(
    rpm, h_0, mu_0, rho, 0.0, 0.0, t, dt,
    use_evaporation=False,
    use_viscosity_growth=False,
)

df_meyer = simulate_spin_coating(
    rpm, h_0, mu_0, rho, E, k, t, dt,
    use_evaporation=True,
    use_viscosity_growth=True,
)

final_ebp = df_ebp["Thickness (μm)"].iloc[-1]
final_meyer = df_meyer["Thickness (μm)"].iloc[-1]

# 오리지널 함수 호출로 복구
alpha_eff = calculate_effective_alpha(
    edge_bead_strength,
    rpm,
    rpm_ref=3000,
    m=rpm_sensitivity
)

r_profile, h_profile, uniformity, h_max, h_min, h_avg = calculate_radial_profile(
    final_meyer,
    R_mm,
    alpha_eff,
    edge_exponent
)

t_gel = calculate_t_gel(mu_0, mu_gel, k)
uniformity_pass = uniformity <= uniformity_spec

col1, col2, col3 = st.columns(3)
col1.metric("Final Thickness: EBP", f"{final_ebp:.3f} μm")
col2.metric("Final Thickness: Meyerhofer Model", f"{final_meyer:.3f} μm")
col3.metric("Thickness Difference", f"{final_meyer - final_ebp:.3f} μm")

col4, col5, col6 = st.columns(3)
col4.metric("Radial Uniformity", f"{uniformity:.3f} %")
col5.metric("Effective Edge Bead Strength", f"{alpha_eff:.4f}")
col6.metric("Uniformity Result", "PASS" if uniformity_pass else "FAIL")

if t_gel is None:
    st.info("t_gel is not defined because k = 0. Viscosity does not increase with time.")
else:
    st.info(f"Predicted gel time t_gel = {t_gel:.2f} s, based on μ(t)=μ₀e^(kt).")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Model Comparison",
    "RPM Effect",
    "Viscosity Effect",
    "Evaporation Effect",
    "Radial Uniformity",
    "Data & Insight"
])

with tab1:
    st.subheader("EBP Model vs Meyerhofer Model")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df_ebp["Time (s)"],
        df_ebp["Thickness (μm)"],
        label="EBP: no evaporation, constant viscosity"
    )
    ax.plot(
        df_meyer["Time (s)"],
        df_meyer["Thickness (μm)"],
        label="Meyerhofer Model: evaporation + μ(t)",
        color="red"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Film Thickness (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.subheader("Effect of Spin Speed on Meyerhofer Model")

    fig, ax = plt.subplots(figsize=(8, 5))
    summary = []

    for r in rpm_values:
        df = simulate_spin_coating(r, h_0, mu_0, rho, E, k, t, dt)
        ax.plot(df["Time (s)"], df["Thickness (μm)"], label=f"{r} RPM")
        summary.append([r, df["Thickness (μm)"].iloc[-1]])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Film Thickness of Meyerhofer Model (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.dataframe(pd.DataFrame(summary, columns=["RPM", "Final Thickness (μm)"]))

with tab3:
    st.subheader("Effect of Initial Viscosity on Meyerhofer Model")

    fig, ax = plt.subplots(figsize=(8, 5))
    summary = []

    for mu_case in mu_values:
        df = simulate_spin_coating(rpm, h_0, mu_case, rho, E, k, t, dt)
        ax.plot(df["Time (s)"], df["Thickness (μm)"], label=f"μ₀={mu_case} Pa·s")
        summary.append([mu_case, df["Thickness (μm)"].iloc[-1]])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Film Thickness of Meyerhofer Model (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.dataframe(pd.DataFrame(summary, columns=["Initial Viscosity (Pa·s)", "Final Thickness (μm)"]))

with tab4:
    st.subheader("Effect of Evaporation Rate on Meyerhofer Model")

    fig, ax = plt.subplots(figsize=(8, 5))
    summary = []

    for E_case in E_values:
        df = simulate_spin_coating(rpm, h_0, mu_0, rho, E_case, k, t, dt)
        ax.plot(df["Time (s)"], df["Thickness (μm)"], label=f"E={E_case} μm/s")
        summary.append([E_case, df["Thickness (μm)"].iloc[-1]])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Film Thickness of Meyerhofer Model (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.dataframe(pd.DataFrame(summary, columns=["Evaporation Rate (μm/s)", "Final Thickness (μm)"]))

with tab5:
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
            "Base Edge Bead Strength",
            "Effective Edge Bead Strength",
            "RPM Sensitivity",
            "Result"
        ],
        "Value": [
            f"{h_min:.4f} μm",
            f"{h_max:.4f} μm",
            f"{h_avg:.4f} μm",
            f"{uniformity:.4f} %",
            f"±{uniformity_spec:.2f} %",
            f"{edge_bead_strength:.4f}",
            f"{alpha_eff:.4f}",
            f"{rpm_sensitivity:.2f}",
            "PASS" if uniformity_pass else "FAIL"
        ]
    })

    st.subheader("Uniformity Evaluation")
    st.dataframe(uniformity_df)

with tab6:
    st.subheader("Simulation Data")
    st.dataframe(df_meyer)

    st.subheader("Governing Equation Used")
    st.latex(r"\frac{dh}{dt} = -\frac{2\rho\omega^2}{3\mu(t)}h^3 - E")
    st.latex(r"\mu(t)=\mu_0 e^{kt}")

    st.subheader("Radial Profile & Uniformity Formula")
    st.latex(r"h(r,t) = h(t) \left[ 1+\alpha_{eff}\left(\frac{r}{R}\right)^n \right]")
    st.latex(r"\alpha_{eff} = \alpha \left( \frac{RPM_{ref}}{RPM} \right)^m")
    st.latex(r"Uniformity (\%) = \pm \frac{h_{max}-h_{min}}{2 \times h_{avg}} \times 100")
