
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.linalg import solve_banded


# --------------------------------------------------
# 1. Exact mathematical solution
# --------------------------------------------------

def exact_solution(x, t, alpha):
    return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)


# --------------------------------------------------
# 2. Explicit finite difference solver
# --------------------------------------------------

def solve_explicit(alpha, N, dt_requested, T):
    dx = 1.0 / N
    steps = max(1, int(np.ceil(T / dt_requested)))
    dt = T / steps
    r = alpha * dt / dx**2

    x = np.linspace(0.0, 1.0, N + 1)
    u = np.sin(np.pi * x)
    u[0] = 0.0
    u[-1] = 0.0

    history = [u.copy()]

    start = time.perf_counter()

    for _ in range(steps):
        old = u.copy()

        u[1:-1] = (
            old[1:-1]
            + r * (old[:-2] - 2 * old[1:-1] + old[2:])
        )

        u[0] = 0.0
        u[-1] = 0.0
        history.append(u.copy())

    runtime = time.perf_counter() - start

    return x, dt, r, np.array(history), runtime


# --------------------------------------------------
# 3. Implicit backward Euler solver
# --------------------------------------------------

def solve_implicit(alpha, N, dt_requested, T):
    dx = 1.0 / N
    steps = max(1, int(np.ceil(T / dt_requested)))
    dt = T / steps
    r = alpha * dt / dx**2

    x = np.linspace(0.0, 1.0, N + 1)
    u = np.sin(np.pi * x)
    u[0] = 0.0
    u[-1] = 0.0

    n = N - 1

    # Banded representation of the tridiagonal matrix.
    ab = np.zeros((3, n))
    ab[0, 1:] = -r
    ab[1, :] = 1.0 + 2.0 * r
    ab[2, :-1] = -r

    history = [u.copy()]

    start = time.perf_counter()

    for _ in range(steps):
        u[1:-1] = solve_banded(
            (1, 1),
            ab,
            u[1:-1]
        )

        u[0] = 0.0
        u[-1] = 0.0
        history.append(u.copy())

    runtime = time.perf_counter() - start

    return x, dt, r, np.array(history), runtime


# --------------------------------------------------
# 4. Error metrics
# --------------------------------------------------

def calculate_metrics(numerical, exact):
    difference = numerical - exact

    return {
        "Maximum absolute error": float(
            np.max(np.abs(difference))
        ),
        "RMSE": float(
            np.sqrt(np.mean(difference**2))
        ),
    }


# --------------------------------------------------
# 5. Streamlit interface
# --------------------------------------------------

st.set_page_config(
    page_title="MathScope",
    page_icon="🔬",
    layout="wide",
)

st.title("MathScope")
st.subheader("An Interactive Numerical Methods Laboratory")

st.write(
    "Explore the one-dimensional heat equation and compare "
    "two numerical methods against its exact solution."
)

st.latex(
    r"\frac{\partial u}{\partial t}"
    r"=\alpha\frac{\partial^2u}{\partial x^2}"
)

with st.sidebar:
    st.header("Experiment settings")

    alpha = st.slider(
        "Diffusion coefficient α",
        min_value=0.01,
        max_value=1.0,
        value=0.1,
        step=0.01,
    )

    N = st.select_slider(
        "Number of spatial intervals",
        options=[20, 40, 80, 120, 160],
        value=40,
    )

    dt_requested = st.select_slider(
        "Requested time step",
        options=[
            0.0001,
            0.00025,
            0.0005,
            0.001,
            0.002,
            0.005,
        ],
        value=0.0005,
        format_func=lambda v: f"{v:g}",
    )

    T = st.select_slider(
        "Final simulation time",
        options=[0.01, 0.025, 0.05, 0.1, 0.2],
        value=0.05,
    )

    method = st.radio(
        "Numerical method",
        ["Both methods", "Explicit", "Implicit"],
    )

dx = 1.0 / N
steps = max(1, int(np.ceil(T / dt_requested)))
dt_actual = T / steps
r = alpha * dt_actual / dx**2

st.write(
    f"Actual time step: `{dt_actual:.6g}`  |  "
    f"Time steps: `{steps}`  |  "
    f"Stability ratio r: `{r:.4f}`"
)

if r > 0.5:
    st.warning(
        "The explicit method violates its standard stability "
        "condition r ≤ 0.5. Its results may become unstable."
    )
else:
    st.success(
        "The explicit method satisfies its standard "
        "stability condition."
    )

if steps > 5000:
    st.error(
        "This setting requires more than 5,000 time steps. "
        "Choose a larger requested time step."
    )
    st.stop()

run_button = st.button("Run experiment", type="primary")

if run_button:
    results = {}

    if method in ["Both methods", "Explicit"]:
        results["Explicit"] = solve_explicit(
            alpha, N, dt_requested, T
        )

    if method in ["Both methods", "Implicit"]:
        results["Implicit"] = solve_implicit(
            alpha, N, dt_requested, T
        )

    x = np.linspace(0.0, 1.0, N + 1)
    exact = exact_solution(x, T, alpha)

    st.header("Results at final time")

    columns = st.columns(len(results) + 1)

    columns[0].metric(
        "Grid intervals",
        N,
    )

    for i, (name, result) in enumerate(results.items(), start=1):
        _, _, _, history, runtime = result
        metrics = calculate_metrics(history[-1], exact)

        columns[i].metric(
            f"{name} RMSE",
            f"{metrics['RMSE']:.3e}",
        )

        st.caption(
            f"{name}: maximum error = "
            f"{metrics['Maximum absolute error']:.3e}; "
            f"runtime = {runtime:.4f} s"
        )

    # Compare final numerical profiles.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x, exact, label="Exact solution", linewidth=2)

    for name, result in results.items():
        history = result[3]
        ax.plot(x, history[-1], "--", label=name)

    ax.set_xlabel("Position x")
    ax.set_ylabel("Temperature u(x, T)")
    ax.set_title("Numerical solutions versus exact solution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # Plot evolution through space and time.
    st.subheader("Heat evolution")

    for name, result in results.items():
        history = result[3]

        fig, ax = plt.subplots(figsize=(9, 4.5))

        image = ax.imshow(
            history,
            aspect="auto",
            origin="lower",
            extent=[0.0, 1.0, 0.0, T],
            cmap="inferno",
        )

        ax.set_xlabel("Position x")
        ax.set_ylabel("Time t")
        ax.set_title(f"{name}: temperature evolution")
        fig.colorbar(image, ax=ax, label="Temperature")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    # Display numerical comparison table.
    rows = []

    for name, result in results.items():
        _, _, _, history, runtime = result
        metrics = calculate_metrics(history[-1], exact)

        rows.append({
            "Method": name,
            "RMSE": metrics["RMSE"],
            "Maximum error": metrics["Maximum absolute error"],
            "Runtime (seconds)": runtime,
        })

    st.subheader("Method comparison")
    st.dataframe(rows, use_container_width=True)

    st.info(
        "Runtime depends on your computer and the chosen grid. "
        "For meaningful comparisons, repeat experiments and "
        "record the settings."
    )

else:
    st.info("Choose experiment settings and click Run experiment.")

