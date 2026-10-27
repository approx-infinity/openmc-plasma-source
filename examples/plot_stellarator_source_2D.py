"""Plot fixed X-Z and Z=0 X-Y cuts of a stellarator flux surface.

This example uses Matplotlib only. It evaluates the continuous
Fourier--Zernike surface directly and does not construct or plot an OpenMC
source mesh.
"""

from math import pi

import matplotlib.pyplot as plt
import numpy as np

from openmc_plasma_source.stellarator_source import (
    FourierZernikeSurface,
    Helias5BScaleCase,
    rotating_ellipse_initial_guess,
)


SURFACE_RADIUS = 1.0
TOROIDAL_ANGLE = 36.0  # physical toroidal angle [rad] for the X-Z cut
N_THETA = 256
N_ZETA = 512
N_ROOT_ITERATIONS = 60


def poloidal_cross_section(
    surface: FourierZernikeSurface,
    normalized_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one complete poloidal cross-section at fixed toroidal angle."""

    theta_values = np.linspace(0.0, 2.0 * pi, N_THETA + 1)
    radius = np.empty_like(theta_values)
    height = np.empty_like(theta_values)
    for index, theta in enumerate(theta_values):
        point = surface.evaluate(normalized_radius, theta, TOROIDAL_ANGLE)
        radius[index] = point.R
        height[index] = point.Z

    # At TOROIDAL_ANGLE=0 this is the physical X-Z plane. For another
    # toroidal angle, X is the radial coordinate projected into that plane.
    x = radius * np.cos(TOROIDAL_ANGLE)
    return x, height


def _bisect_zero(
    surface: FourierZernikeSurface,
    normalized_radius: float,
    zeta: float,
    theta_left: float,
    theta_right: float,
) -> float:
    """Find one scalar poloidal root of Z(s, theta, zeta)=0."""

    left = theta_left
    right = theta_right
    left_value = surface.evaluate(normalized_radius, left, zeta).Z
    for _ in range(N_ROOT_ITERATIONS):
        middle = 0.5 * (left + right)
        middle_value = surface.evaluate(normalized_radius, middle, zeta).Z
        if left_value * middle_value <= 0.0:
            right = middle
        else:
            left = middle
            left_value = middle_value
    return 0.5 * (left + right)


def midplane_intersection(
    surface: FourierZernikeSurface,
    normalized_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all X-Y intersections of the surface with the plane Z=0."""

    zeta_values = np.linspace(0.0, 2.0 * pi, N_ZETA + 1)
    theta_scan = np.linspace(0.0, 2.0 * pi, N_THETA + 1)
    intersections: list[list[tuple[float, float]]] = []

    for zeta in zeta_values:
        heights = [
            surface.evaluate(normalized_radius, theta, zeta).Z
            for theta in theta_scan
        ]
        roots = []
        for index in range(N_THETA):
            theta_left = theta_scan[index]
            theta_right = theta_scan[index + 1]
            left_value = heights[index]
            right_value = heights[index + 1]
            if left_value == 0.0:
                roots.append(theta_left)
            elif left_value * right_value < 0.0:
                roots.append(
                    _bisect_zero(
                        surface,
                        normalized_radius,
                        zeta,
                        theta_left,
                        theta_right,
                    )
                )

        # Remove a duplicate root at the periodic theta seam.
        unique_roots = []
        for root in roots:
            if not unique_roots or abs(root - unique_roots[-1]) > 1.0e-8:
                unique_roots.append(root)

        intersections.append(
            [
                (
                    surface.evaluate(normalized_radius, theta, zeta).R
                    * np.cos(zeta),
                    surface.evaluate(normalized_radius, theta, zeta).R
                    * np.sin(zeta),
                )
                for theta in unique_roots
            ]
        )

    maximum_intersections = max((len(points) for points in intersections), default=0)
    x_branches = np.full((maximum_intersections, len(zeta_values)), np.nan)
    y_branches = np.full_like(x_branches, np.nan)
    for zeta_index, points in enumerate(intersections):
        for branch, (x, y) in enumerate(points):
            x_branches[branch, zeta_index] = x
            y_branches[branch, zeta_index] = y
    return x_branches, y_branches


def main() -> None:
    case = Helias5BScaleCase(
        # Illustrative rotating-ellipse geometry at the Helias-5B scale.
        elongation=1.6,
        rotation=1,
        asymmetry=0.05,
    )
    r_coefficients, z_coefficients = rotating_ellipse_initial_guess(
        major_radius=case.major_radius,
        minor_radius=case.minor_radius,
        nfp=case.nfp,
        elongation=case.elongation,
        rotation=case.rotation,
        asymmetry=case.asymmetry,
    )
    surface = FourierZernikeSurface(
        r_coefficients,
        z_coefficients,
        case.nfp,
        field_period_poloidal_shift=case.rotation * pi,
    )

    x_xz, z_xz = poloidal_cross_section(surface, SURFACE_RADIUS)
    x_xy, y_xy = midplane_intersection(surface, SURFACE_RADIUS)
    x_xz_m = x_xz / 100.0
    z_xz_m = z_xz / 100.0
    x_xy_m = x_xy / 100.0
    y_xy_m = y_xy / 100.0

    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    axes[0].plot(x_xz_m, z_xz_m, color="#147d92", linewidth=2.0)
    axes[0].fill(x_xz_m, z_xz_m, color="#67c5d8", alpha=0.18)
    axes[0].set_title(f"Poloidal X-Z cut at ζ={TOROIDAL_ANGLE:g} rad")
    axes[0].set_xlabel("X [m]")
    axes[0].set_ylabel("Z [m]")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, alpha=0.25)

    for x_branch, y_branch in zip(x_xy_m, y_xy_m):
        axes[1].plot(x_branch, y_branch, color="#d97706", linewidth=1.8)
    axes[1].set_title("X-Y intersection at Z=0")
    axes[1].set_xlabel("X [m]")
    axes[1].set_ylabel("Y [m]")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True, alpha=0.25)

    figure.suptitle(
        "Continuous Fourier--Zernike Stellarator Flux Surface "
        f"(s={SURFACE_RADIUS:g}, Nfp={case.nfp}, rotation={case.rotation})"
    )
    plt.show()


if __name__ == "__main__":
    main()
