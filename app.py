import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("Spin Coating Simulator")

st.sidebar.header("Input Parameters")

rpm = st.sidebar.slider("Spin Speed (RPM)", 500, 6000, 3000)

h0 = st.sidebar.number_input(
    "Initial Thickness (um)",
    value=100.0
)

mu0 = st.sidebar.number_input(
    "Initial Viscosity (Pa·s)",
    value=0.05
)

E = st.sidebar.number_input(
    "Evaporation Rate (um/s)",
    value=0.5
)

k = st.sidebar.number_input(
    "Viscosity Growth Rate",
    value=0.05
)

t_end = st.sidebar.number_input(
    "Simulation Time (s)",
    value=60
)

omega = rpm * 2*np.pi / 60

rho = 1000

dt = 0.1

time = np.arange(0, t_end+dt, dt)

h = np.zeros(len(time))
h[0] = h0

for i in range(len(time)-1):

    mu = mu0*np.exp(k*time[i])

    dhdt = (
        -(2*rho*omega**2/(3*mu))*h[i]**3*1e-18
        - E
    )

    h[i+1] = max(h[i] + dhdt*dt, 0)

df = pd.DataFrame({
    "Time (s)": time,
    "Thickness (um)": h
})

st.subheader("Thickness Evolution")

fig, ax = plt.subplots()

ax.plot(time, h)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Thickness (um)")
ax.grid(True)

st.pyplot(fig)

st.subheader("Final Thickness")

st.write(f"{h[-1]:.2f} um")

st.subheader("Data")

st.dataframe(df)
