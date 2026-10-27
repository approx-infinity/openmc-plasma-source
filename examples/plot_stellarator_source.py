"""Plot continuous Fourier–Zernike stellarator flux surfaces in 3-D.

The example uses the supplied Helias-5B dimensional scale with an illustrative
rotating-ellipse Fourier–Zernike geometry. It plots the continuous coordinate
mapping directly; no OpenMC mesh or source voxelization is involved.

A physically faithful Helias-5B plot requires the actual non-axisymmetric
Fourier boundary coefficients.
"""

from math import pi

import numpy as np
import pyvista as pv

from openmc_plasma_source.stellarator_source import (
    FourierZernikeSurface,
    Helias5BScaleCase,
    rotating_ellipse_initial_guess,
)


SURFACE_RADIUS = 1.00
N_THETA = 96
N_ZETA = 160


def continuous_surface_coordinates(
    surface: FourierZernikeSurface,
    normalized_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate one continuous flux surface on a periodic angular grid."""

    theta_values = np.linspace(0.0, 2.0 * pi, N_THETA + 1)
    zeta_values = np.linspace(0.0, 2.0 * pi, N_ZETA + 1)
    R = np.empty((len(theta_values), len(zeta_values)))
    Z = np.empty_like(R)

    rotation = surface.field_period_poloidal_shift / pi
    for i, theta_parameter in enumerate(theta_values):
        for j, zeta in enumerate(zeta_values):
            # Follow the twisted poloidal seam introduced by the visible
            # ellipse-rotation convention. This keeps the final toroidal
            # column connected to the first one in the physical surface.
            theta = theta_parameter + 0.5 * rotation * surface.nfp * zeta
            point = surface.evaluate(normalized_radius, theta, zeta)
            R[i, j] = point.R
            Z[i, j] = point.Z

    _, zeta_grid = np.meshgrid(theta_values, zeta_values, indexing="ij")
    X = R * np.cos(zeta_grid)
    Y = R * np.sin(zeta_grid)
    return X, Y, Z


def main() -> None:
    case = Helias5BScaleCase(
        # Illustrative shape parameters for the current rotating-ellipse model,
        # not a physical Helias-5B equilibrium.
        elongation=1.6,
        rotation=1,
        # Exaggerated for visual clarity: the production example uses 0.05.
        asymmetry=0,
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

    plotter = pv.Plotter(window_size=(1200, 900))
    X, Y, Z = continuous_surface_coordinates(surface, SURFACE_RADIUS)
    flux_surface = pv.StructuredGrid(X, Y, Z)
    plotter.add_mesh(
        flux_surface,
        color="#b7358e",
        opacity=1.0,
        show_edges=False,
        smooth_shading=True,
        name="flux_surface",
    )

    plotter.add_text(
        "Continuous Fourier–Zernike Stellarator Flux Surface "
        f"(s={SURFACE_RADIUS:g})\n"
        f"R₀ = {case.major_radius / 100.0:g} m, "
        f"a = {case.minor_radius / 100.0:g} m, Nfp = {case.nfp}",
        font_size=12,
    )
    plotter.add_axes()
    plotter.show_grid(
        xtitle="x [cm]",
        ytitle="y [cm]",
        ztitle="z [cm]",
    )
    plotter.show()


if __name__ == "__main__":
    main()
