"""Reproduce paper Figure 6 through the package's geometry optimizer."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openmc_plasma_source.stellarator_source import (
    ActionGrid,
    CoefficientLayout,
    optimize_surface,
)


# Figure 6's axisymmetric bean boundary (dimensionless paper units).
R_BOUNDARY = {(0.0, 0): 1.0, (0.0, 1): 0.6, (0.0, 2): 0.8}
Z_BOUNDARY = {(0.0, 1): 0.8, (0.0, 2): 0.1}
OMEGA_VALUES = (0.01, 0.5, 1.0)

# This is the optimizer grid; the denser plotting grid is independent.
ACTION_GRID = (7, 32, 1)
PLOT_N_S = 121
PLOT_N_THETA = 192
OUTPUT_PATH = Path(__file__).with_name("stellarator_figure6_reproduction.png")


def make_layout() -> CoefficientLayout:
    """Build the paper's M=L=2 boundary-constrained coefficient layout."""

    source_layout = CoefficientLayout.from_orders(
        n_max=0,
        m_max=2,
        l_max=2,
        r_boundary=R_BOUNDARY,
        z_boundary=Z_BOUNDARY,
    )
    # sin(0) vanishes identically, so the source's (n,m)=(0,0) Z modes do not
    # affect the paper geometry. Omit those inactive variables for this fit.
    return CoefficientLayout(
        r_modes=source_layout.r_modes,
        z_modes=tuple(
            mode
            for mode in source_layout.z_modes
            if not (mode[0] == 0.0 and mode[1] == 0)
        ),
        r_boundary=source_layout.r_boundary,
        z_boundary=source_layout.z_boundary,
    )


def optimize_for_omega(
    layout: CoefficientLayout, omega: float, initial_vector=None
):
    """Optimize the single free paper coefficient using the package code."""

    if initial_vector is None:
        initial_vector = [-0.42]
    return optimize_surface(
        layout=layout,
        nfp=1,
        grid=ActionGrid.uniform(*ACTION_GRID),
        omega=omega,
        initial_vector=initial_vector,
        r_boundary=R_BOUNDARY,
        z_boundary=Z_BOUNDARY,
        method="BFGS",
        options={"maxiter": 300, "gtol": 1.0e-9},
        use_geometry_penalty=False,
        require_valid_surface=False,
        jit=True,
        geometry_description="axisymmetric bean boundary",
    )


def evaluate_grid(surface):
    """Evaluate the package surface and normalized poloidal Jacobian."""

    s_nodes = np.linspace(0.0, 1.0, PLOT_N_S)
    theta_nodes = np.linspace(0.0, 2.0 * np.pi, PLOT_N_THETA + 1)
    radius = np.empty((PLOT_N_S, PLOT_N_THETA + 1))
    height = np.empty_like(radius)

    for i, s in enumerate(s_nodes):
        for j, theta in enumerate(theta_nodes):
            point = surface.evaluate(s, theta, 0.0)
            radius[i, j] = point.R
            height[i, j] = point.Z

    s_centers = 0.5 * (s_nodes[:-1] + s_nodes[1:])
    theta_centers = 0.5 * (theta_nodes[:-1] + theta_nodes[1:])
    jacobian_over_s = np.empty((PLOT_N_S - 1, PLOT_N_THETA))
    for i, s in enumerate(s_centers):
        for j, theta in enumerate(theta_centers):
            point = surface.evaluate(s, theta, 0.0)
            # The paper colors the normalized poloidal Jacobian sqrt(g_p)/s.
            jacobian_poloidal = point.jacobian / point.R
            jacobian_over_s[i, j] = jacobian_poloidal / s
    jacobian_over_s /= np.max(jacobian_over_s)
    return radius, height, jacobian_over_s


def minimum_jacobian(surface):
    """Sample the signed 3-D Jacobian away from the coordinate axis."""

    s_values = np.linspace(1.0 / PLOT_N_S, 1.0, 40)
    theta_values = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    return min(
        surface.evaluate(s, theta, 0.0).jacobian
        for s in s_values
        for theta in theta_values
    )


def plot_panel(axis, result, omega: float):
    radius, height, normalized_jacobian = evaluate_grid(result.surface)
    filled = axis.pcolormesh(
        radius,
        height,
        normalized_jacobian,
        shading="flat",
        cmap="plasma",
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )

    for index in range(0, PLOT_N_S, 8):
        axis.plot(
            radius[index], height[index], color="#6b7280", linewidth=0.55, alpha=0.8
        )
    for index in range(0, PLOT_N_THETA, 8):
        axis.plot(
            radius[:, index], height[:, index], color="#6b7280", linewidth=0.55, alpha=0.8
        )

    axis.plot(radius[-1], height[-1], color="#dc2626", linewidth=2.0)
    axis.plot(radius[0, 0], height[0, 0], ".", color="#2563eb", ms=7)
    axis.set_title(rf"$\omega={omega:g}$")
    axis.set_xlabel(r"$R$")
    axis.set_aspect("equal", adjustable="box")
    return filled


def main() -> None:
    layout = make_layout()
    results = []
    initial_vector = None
    for omega in OMEGA_VALUES:
        result = optimize_for_omega(layout, omega, initial_vector)
        results.append(result)
        initial_vector = result.coefficient_vector
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.7), constrained_layout=True)
    filled = None
    for axis, result, omega in zip(axes, results, OMEGA_VALUES):
        filled = plot_panel(axis, result, omega)
    axes[0].set_ylabel(r"$Z$")
    figure.colorbar(
        filled,
        ax=axes,
        location="right",
        label=r"Normalized poloidal Jacobian $\sqrt{g}_{p,N}/s$",
        ticks=np.linspace(0.0, 1.0, 6),
    )
    figure.suptitle("Figure 6 reproduction: optimized grids for the bean-shaped boundary")
    figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {OUTPUT_PATH}")
    for omega, result in zip(OMEGA_VALUES, results):
        min_jacobian = minimum_jacobian(result.surface)
        print(
            f"omega={omega:g}: r_20={result.coefficient_vector[0]:.8f}, "
            f"action={result.diagnostics['final_objective']:.8f}, "
            f"valid={bool(result.diagnostics['valid_surface'])}, "
            f"sampled minJ={min_jacobian:.3e}"
        )
    plt.show()


if __name__ == "__main__":
    main()
