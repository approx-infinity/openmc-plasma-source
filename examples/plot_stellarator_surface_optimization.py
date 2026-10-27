"""Compare analytical and optimized stellarator coordinate grids.

The comparison is made at one fixed physical toroidal angle. Such a cut is a
poloidal R-Z section. Each panel shows constant-s and constant-theta lines,
so changes to the optimized interior mapping are visible even though the
external boundary at s=1 is constrained to remain unchanged.

The example uses Matplotlib only and does not construct an OpenMC source mesh.
"""

from math import pi

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from openmc_plasma_source.stellarator_source import (
    ActionGrid,
    CoefficientLayout,
    FourierZernikeSurface,
    Helias5BScaleCase,
    optimize_surface,
    rotating_ellipse_boundary_coefficients,
    rotating_ellipse_initial_guess,
)


TOROIDAL_ANGLE = 45.0  # physical toroidal angle [rad]
GRID_S = 21
GRID_THETA = 32
BASIS_ORDER = (1, 2, 3)
OPTIMIZATION_GRID = (3, 8, 4)


def make_initial_and_optimized_surfaces() -> tuple[
    FourierZernikeSurface, FourierZernikeSurface
]:
    """Build the analytical and optimized Fourier--Zernike surfaces."""

    case = Helias5BScaleCase(
        # Illustrative rotating-ellipse geometry at the Helias-5B scale.
        elongation=1.6,
        rotation=1,
        # A small asymmetric perturbation makes the interior optimization
        # visible; the perfectly symmetric ellipse is already stationary for
        # this action.
        asymmetry=0.05,
    )
    r_boundary, z_boundary = rotating_ellipse_boundary_coefficients(
        major_radius=case.major_radius,
        minor_radius=case.minor_radius,
        nfp=case.nfp,
        elongation=case.elongation,
        rotation=case.rotation,
        asymmetry=case.asymmetry,
    )
    r_initial, z_initial = rotating_ellipse_initial_guess(
        major_radius=case.major_radius,
        minor_radius=case.minor_radius,
        nfp=case.nfp,
        elongation=case.elongation,
        rotation=case.rotation,
        asymmetry=case.asymmetry,
    )
    layout = CoefficientLayout.from_orders(
        n_max=BASIS_ORDER[0],
        m_max=BASIS_ORDER[1],
        l_max=BASIS_ORDER[2],
        r_boundary=r_boundary,
        z_boundary=z_boundary,
        twisted_rotation=case.rotation,
    )
    initial_vector = layout.initial_vector(r_initial, z_initial)
    initial_r, initial_z = layout.coefficients_from_vector(initial_vector)
    initial_surface = FourierZernikeSurface(
        initial_r,
        initial_z,
        case.nfp,
        field_period_poloidal_shift=case.rotation * pi,
    )

    optimization_result = optimize_surface(
        layout=layout,
        nfp=case.nfp,
        grid=ActionGrid.uniform(*OPTIMIZATION_GRID),
        omega=0.5,
        initial_vector=initial_vector,
        r_boundary=r_boundary,
        z_boundary=z_boundary,
        options={"maxiter": 20, "gtol": 1.0e-8},
        use_geometry_penalty=False,
        jit=True,
        field_period_poloidal_shift=case.rotation * pi,
        geometry_description=(
            f"R_0={case.major_radius / 100:g}m, "
            f"a={case.minor_radius / 100:g}m"
        ),
    )
    return initial_surface, optimization_result.surface


def coordinate_grid(
    surface: FourierZernikeSurface,
    toroidal_angle: float,
    n_s: int = GRID_S,
    n_theta: int = GRID_THETA,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a fixed-zeta R-Z coordinate grid.

    ``R[i, :]`` and ``Z[i, :]`` are constant-s flux surfaces, while
    ``R[:, j]`` and ``Z[:, j]`` are constant-theta radial spokes.
    """

    s_values = np.linspace(0.0, 1.0, n_s)
    theta_values = np.linspace(0.0, 2.0 * pi, n_theta + 1)
    radius = np.empty((n_s, n_theta + 1))
    height = np.empty_like(radius)
    for s_index, normalized_radius in enumerate(s_values):
        for theta_index, theta in enumerate(theta_values):
            point = surface.evaluate(normalized_radius, theta, toroidal_angle)
            radius[s_index, theta_index] = point.R
            height[s_index, theta_index] = point.Z

    # The geometry coefficients use centimetres; plot in metres.
    return radius / 100.0, height / 100.0


def plot_coordinate_grid(
    axis: Axes,
    surface: FourierZernikeSurface,
    title: str,
    toroidal_angle: float,
) -> None:
    """Plot constant-s surfaces and constant-theta spokes on one axis."""

    radius, height = coordinate_grid(surface, toroidal_angle)

    for s_index in range(radius.shape[0]):
        axis.plot(
            radius[s_index, :],
            height[s_index, :],
            color="#dc2626" if s_index == radius.shape[0] - 1 else "#64748b",
            linewidth=0.8 if s_index < radius.shape[0] - 1 else 2.0,
            alpha=0.75 if s_index < radius.shape[0] - 1 else 1.0,
            label=(
                "External boundary (s=1)"
                if s_index == radius.shape[0] - 1
                else None
            ),
        )

    for theta_index in range(radius.shape[1]):
        axis.plot(
            radius[:, theta_index],
            height[:, theta_index],
            color="#94a3b8",
            linewidth=0.7,
            alpha=0.65,
        )

    axis.scatter(
        radius[0, 0],
        height[0, 0],
        color="#2563eb",
        s=35,
        zorder=3,
        label="Coordinate axis (s=0)",
    )
    axis.set_title(title)
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(fontsize=8, loc="best")


def main() -> None:
    initial_surface, optimized_surface = make_initial_and_optimized_surfaces()
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    plot_coordinate_grid(
        axes[0],
        initial_surface,
        "Analytical initial coordinate grid",
        TOROIDAL_ANGLE,
    )
    plot_coordinate_grid(
        axes[1],
        optimized_surface,
        "After JAX/SciPy optimization",
        TOROIDAL_ANGLE,
    )
    figure.suptitle(
        "Stellarator R-Z coordinate grids at "
        f"ζ={TOROIDAL_ANGLE:g} rad (Nfp={initial_surface.nfp})"
    )
    plt.show()


if __name__ == "__main__":
    main()
