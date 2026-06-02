import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Spin Coating Simulator", layout="wide")

st.title("Spin Coating Simulator")
st.caption("EBP Model + Meyerhofer Model evaporation and viscosity increase")

# -----------------------------
# 1. 사이드바 입력 변수 (Inputs)
# -----------------------------
st.sidebar.header("Input Parameters")

rpm = st.sidebar.slider("RPM (ω)", 500, 5000, 3000, 100)
h_0 = st.sidebar.number_input("Initial Thickness h₀ (μm)", value=50.0, min_value=1.0)
mu_0 = st.sidebar.number_input("Initial Viscosity μ₀ (Pa·s)", value=0.03, min_value=0.001)
rho = st.sidebar.number_input("Density ρ (kg/m³)", value=1000.0, min_value=1.0)
E = st.sidebar.number_input("Evaporation Rate E (μm/s)", value=0.015, min_value=0.0)
k = st.sidebar.number_input("Viscosity Growth Rate k (1/s)", value=0.025, min_value=0.0)
t_max = st.sidebar.number_input("Simulation Time (s)", value=30.0, min_value=1.0)
dt = st.sidebar.number_input("Time Step Δt (s)", value=0.1, min_value=0.001)

st.sidebar.markdown("---")
st.sidebar.header("Radial Profile / Uniformity")

R_mm = st.sidebar.number_input("Wafer Radius R (mm)", value=50.0, min_value=1.0)
edge_bead_strength = st.sidebar.slider("Base Edge Bead Strength α", 0.0, 0.10, 0.04, 0.005)
edge_exponent = st.sidebar.slider("Edge Bead Exponent n", 2, 12, 6, 1)
uniformity_spec = st.sidebar.number_input("Uniformity Spec (%)", value=2.0, min_value=0.1)
mu_gel = st.sidebar.number_input("Gel Viscosity μ_gel (Pa·s)", value=0.30, min_value=0.001)

st.sidebar.markdown("---")
st.sidebar.header("Parameter Study Cases")
rpm_values = st.sidebar.multiselect("RPM cases", [1000, 2000, 3000, 4000, 5000], default=[1000, 3000, 5000])

# -----------------------------
# 2. 핵심 물리 함수 (Core Functions)
# -----------------------------
def simulate_spin_coating(rpm, h_0, mu_0, rho, E, k, t_max, dt, use_evaporate=True):
    omega = rpm * 2 * np.pi / 60
    time_steps = np.arange(0, t_max + dt, dt)
    
    h = np.zeros_like(time_steps)
    h[0] = h_0 * 1e-6  # μm -> m 변환
    
    mu = np.zeros_like(time_steps)
    E_m_s = E * 1e-6 if use_evaporate else 0.0
    
    for i in range(len(time_steps) - 1):
        # 시간 경과에 따른 점도 상승 (Meyerhofer Model)
        current_mu = mu_0 * np.exp(k * time_steps[i]) if use_evaporate else mu_0
        mu[i] = current_mu
        
        # 지배 방정식: dh/dt = - (2*rho*omega^2 / 3*mu) * h^3 - E
        dhdt = -(2 * rho * omega**2 / (3 * current_mu)) * h[i]**3 - E_m_s
        h[i + 1] = max(h[i] + dhdt * dt, 0)
        
    mu[-1] = mu_0 * np.exp(k * time_steps[-1]) if use_evaporate else mu_0
    return time_steps, h * 1e6, mu

def calculate_effective_alpha(alpha, rpm, rpm_ref=3000):
    # RPM이 높아짐에 따라 가장자리가 깎이거나 증발해 파고드는 현상을 부호 변환으로 묘사
    return alpha * (1.0 - (rpm / rpm_ref))

def calculate_radial_profile(final_thickness, R_mm, alpha_eff, n):
    r = np.linspace(0, R_mm, 100)
    normalized_r = r / R_mm
    h_r = final_thickness * (1 + alpha_eff * normalized_r**n)
    
    h_max, h_min, h_avg = np.max(h_r), np.min(h_r), np.mean(h_r)
    # 과제 가이드 규격: ± (max - min) / (2 * avg) * 100
    uniformity = ((h_max - h_min) / (2 * h_avg) * 100) if h_avg > 0 else 0
    return r, h_r, uniformity

# -----------------------------
# 3. 데이터 계산 및 대시보드 출력
# -----------------------------
t_steps, h_meyer, mu_meyer = simulate_spin_coating(rpm, h_0, mu_0, rho, E, k, t_max, dt, True)
_, h_ebp, _ = simulate_spin_coating(rpm, h_0, mu_0, rho, E, k, t_max, dt, False)

alpha_eff = calculate_effective_alpha(edge_bead_strength, rpm)
r_prof, h_prof, uniformity = calculate_radial_profile(h_meyer[-1], R_mm, alpha_eff, edge_exponent)

# 결과 상단 메트릭 표시
col1, col2, col3 = st.columns(3)
col1.metric("Final Thickness (Meyerhofer)", f"{h_meyer[-1]:.3f} μm")
col2.metric("Radial Uniformity", f"{uniformity:.3f} %")
status = "PASS" if uniformity <= uniformity_spec else "FAIL"
col3.metric("Uniformity Result (±2% Spec)", status, delta=None, delta_color="normal")

# 겔화 시간 예측 (t_gel)
if k > 0 and mu_gel > mu_0:
    t_gel = np.log(mu_gel / mu_0) / k
    st.info(f"💡 Predicted Gel Time (t_gel): {t_gel:.2f} seconds (Viscosity reaches {mu_gel} Pa·s)")

# -----------------------------
# 4. 탭 구성 (애니메이션, 검증, 스터디)
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Real-time Profile Animation", "Model Validation View", "RPM Parameter Study"])

with tab1:
    st.subheader("Real-time Animation of h(r, t)")
    run_anim = st.button("▶ Run Animation")
    plot_spot = st.empty()
    
    if run_anim:
        # 시간 흐름에 따른 반경 방향 프로파일 애니메이션 구현
        for idx in range(0, len(t_steps), max(1, len(t_steps)//30)):
            current_thick = h_meyer[idx]
            current_alpha = calculate_effective_alpha(edge_bead_strength, rpm) * (idx / len(t_steps))
            r_t, h_rt, _ = calculate_radial_profile(current_thick, R_mm, current_alpha, edge_exponent)
            
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(r_t, h_rt, color="crimson", lw=2.5, label=f"t = {t_steps[idx]:.1f} s")
            ax.set_ylim(0, h_0 * 1.1)
            ax.set_xlabel("Wafer Radius r (mm)")
            ax.set_ylabel("Thickness h (μm)")
            ax.grid(True, linestyle="--")
            ax.legend(loc="upper left")
            plot_spot.pyplot(fig)
            plt.close(fig)
            time.sleep(0.05)
    else:
        # 최종 프로파일 정지 화면
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(r_prof, h_prof, color="darkblue", lw=2.5, label="Final Profile")
        ax.axhline(np.mean(h_prof), color="gray", linestyle="--", label="Average")
        ax.set_xlabel("Wafer Radius r (mm)")
        ax.set_ylabel("Thickness h (μm)")
        ax.grid(True, linestyle="--")
        ax.legend()
        plot_spot.pyplot(fig)
        plt.close(fig)

with tab2:
    st.subheader("Validation View: EBP vs Meyerhofer")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t_steps, h_ebp, label="EBP Model (Centrifugal Only)", color="black", linestyle=":")
    ax.plot(t_steps, h_meyer, label="Meyerhofer Model (Evaporation + Viscosity)", color="red")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Thickness (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    st.subheader("Design Exploration: Variation by RPM cases")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for r_val in rpm_values:
        _, h_case, _ = simulate_spin_coating(r_val, h_0, mu_0, rho, E, k, t_max, dt, True)
        ax.plot(t_steps, h_case, label=f"{r_val} RPM")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Thickness (μm)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
