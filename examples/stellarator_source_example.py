"""Run a three-dimensional stellarator neutron source in OpenMC.

Reference design point (approximate):
    major radius             R0  = 22 m
    effective minor radius   a   = 1.8 m  (V = 2*pi^2*R*a^2 ~ 1400 m^3)
    field periods            nfp = 5
    B on axis                ~5 T
    plasma volume            ~1400 m^3
    fusion power             ~3000 MW
    operation                ISS04-type confinement, no H-mode pedestal

The current public geometry model is a rotating ellipse represented in the
Fourier–Zernike basis. These dimensions therefore provide a scale benchmark,
not a physical HELIAS 5-B equilibrium. A faithful HELIAS source requires its
non-axisymmetric boundary coefficients.
"""

from pathlib import Path
from math import sqrt

import openmc

from openmc_plasma_source import stellarator_source

# Use the example's local cross-section library; replace this path with your
# own library or comment out the assignment to use OpenMC's configured path.
openmc.config["cross_sections"] = (Path(__file__).parent.resolve() / "cross_sections.xml")

ELONGATION = 1.6
# The model's minor_radius is the ellipse semi-minor axis (semi-major axis is
# ELONGATION * minor_radius), so choose it such that the effective radius
# sqrt(a_minor * a_major) equals the Helias-5B value of 180 cm.
MINOR_RADIUS = 180.0 / sqrt(ELONGATION)  # ~142.3 cm


def main() -> None:
    result = stellarator_source(
        # Geometry inputs are in centimetres.
        major_radius=2200.0,
        minor_radius=MINOR_RADIUS,
        nfp=5,
        elongation=ELONGATION,
        rotation=1.0,
        asymmetry=0.05,
        # HELIAS has no H-mode edge barrier, so use L-mode profiles.
        mode="L",
        ion_density_centre=2.123e20,           # m^-3
        ion_density_peaking_factor=0.5,
        ion_density_pedestal=2.0e20,         # unused in L-mode
        ion_density_separatrix=3.0e19,
        ion_temperature_centre=15.2e3,       # eV
        ion_temperature_peaking_factor=1.0,
        ion_temperature_beta=2.0,
        ion_temperature_pedestal=1.0e3,      # eV, unused in L-mode
        ion_temperature_separatrix=0.1e3,    # eV
        pedestal_radius=0.8 * MINOR_RADIUS,  # cm, unused in L-mode
        fuel={"D": 0.5, "T": 0.5},
        # Scalar quadrature resolution: (n_s, n_theta, n_zeta).
        sample_size=(32, 64, 64),
        # OpenMC cylindrical mesh resolution: (n_R, n_phi, n_Z).
        mesh_resolution=(80, 40, 80),
        # The default is the analytical Zernike–Fourier ellipse extension.
        # For higher-order interior optimization, use for example:
        # basis_order=(1, 1, 3),
        # optimize_geometry=True,
        # optimization_grid=(4, 16, 8),
    )

    print(f"Mesh source strength: {result.source.strength:.6e} n/s")
    print(f"Plasma volume: {result.plasma_volume:.6e} m^3")
    print(f"Fusion power: {result.fusion_power:.6e} MW")

    # Minimal vacuum-bounded geometry for a fixed-source OpenMC run.
    sphere_surface = openmc.Sphere(r=100000.0, boundary_type="vacuum")
    cell = openmc.Cell(region=-sphere_surface)
    geometry = openmc.Geometry([cell])

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.batches = 10
    settings.particles = 1000
    settings.source = [result.source]

    model = openmc.model.Model(
        materials=None,
        geometry=geometry,
        settings=settings,
    )
    model.run()

    # Optional visualization of sampled source positions.
    from openmc_source_plotter import plot_source_position

    plot = plot_source_position(this=settings, n_samples=200)
    plot.update_layout(scene={"aspectmode": "data"})
    plot.show()


if __name__ == "__main__":
    main()
