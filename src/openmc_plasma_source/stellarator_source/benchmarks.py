"""Production benchmark cases and convergence studies."""

from dataclasses import dataclass
from math import pi
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from .boundary import rotating_ellipse_initial_guess
from .geometry import FourierZernikeSurface
from .source import stellarator_source


@dataclass(frozen=True)
class Helias5BScaleCase:
    """Helias-5B dimensional scale reference.

    The supplied major/minor radii and five field periods are represented
    exactly. The default circular cross-section is deliberately a numerical
    scale benchmark, not a physical Helias-5B equilibrium. A faithful Helias
    case requires its non-axisymmetric Fourier boundary coefficients and any
    additional shape/profile data.

    Attributes
    ----------
    major_radius, minor_radius : float
        Reference radii in centimetres.
    nfp : int
        Number of field periods.
    elongation : float
        Cross-section elongation used by the rotating-ellipse approximation.
    rotation : int
        Rotating-ellipse orientation index.
    asymmetry : float
        Dimensionless second-harmonic boundary amplitude.
    """

    major_radius: float = 2200.0  # cm, 22 m
    minor_radius: float = 180.0  # cm, 1.8 m
    nfp: int = 5
    elongation: float = 1.0
    rotation: int = 0
    asymmetry: float = 0.0

    @property
    def analytic_volume_m3(self) -> float:
        """Return the closed-form volume for the symmetric case.

        The closed form applies only when ``asymmetry == 0``. For an
        asymmetric boundary, use :attr:`reference_volume_m3`, which evaluates
        the cylindrical Jacobian on an independently refined scalar grid.

        Returns
        -------
        float
            Analytic torus volume in cubic metres.

        Raises
        ------
        ValueError
            If ``asymmetry`` is nonzero.
        """

        if self.asymmetry != 0.0:
            raise ValueError(
                "analytic_volume_m3 is only defined for asymmetry=0; use "
                "reference_volume_m3 for an asymmetric boundary"
            )

        return (
            2.0
            * pi**2
            * self.major_radius
            * self.elongation
            * self.minor_radius**2
            * 1.0e-6
        )

    @property
    def reference_volume_m3(self) -> float:
        """Return the reference volume for this scale case.

        Symmetric cases use the analytic elliptical-torus formula. Asymmetric
        cases use a fixed 16-by-64-by-64 midpoint quadrature of the cylindrical
        Jacobian; this reference is independent of the source-sampling grid but
        is still a numerical approximation.

        Returns
        -------
        float
            Reference volume in cubic metres.
        """

        if self.asymmetry == 0.0:
            return self.analytic_volume_m3

        r_coefficients, z_coefficients = rotating_ellipse_initial_guess(
            self.major_radius,
            self.minor_radius,
            self.nfp,
            self.elongation,
            self.rotation,
            self.asymmetry,
        )
        surface = FourierZernikeSurface(r_coefficients, z_coefficients, self.nfp)
        n_s = 16
        n_theta = 64
        n_zeta = 64
        ds = 1.0 / n_s
        dtheta = 2.0 * pi / n_theta
        dzeta = 2.0 * pi / n_zeta
        volume_cm3 = 0.0
        for i in range(n_s):
            s = (i + 0.5) * ds
            for j in range(n_theta):
                theta = (j + 0.5) * dtheta
                for k in range(n_zeta):
                    zeta = (k + 0.5) * dzeta
                    volume_cm3 += (
                        surface.evaluate(s, theta, zeta).jacobian
                        * ds
                        * dtheta
                        * dzeta
                    )
        return volume_cm3 * 1.0e-6


def helias5b_uniform_profile(
    *,
    ion_density: float = 4.8e20,
    ion_temperature: float = 12.0e3,
    fuel: Mapping[str, float] | None = None,
) -> Dict[str, object]:
    """Create flat L-mode plasma profiles for source-scale benchmarks.

    Parameters
    ----------
    ion_density : float, default=4.8e20
        Constant density in m^-3.
    ion_temperature : float, default=12e3
        Constant temperature in eV.
    fuel : mapping of str to float, optional
        Species fractions. Defaults to equal D/T fractions.

    Returns
    -------
    dict
        Keyword arguments accepted by :func:`stellarator_source`, including
        profile mode, density/temperature levels, pedestal location, and fuel.
    """

    return {
        "mode": "L",
        "ion_density_centre": ion_density,
        "ion_density_peaking_factor": 0.0,
        "ion_density_pedestal": ion_density,
        "ion_density_separatrix": ion_density,
        "ion_temperature_centre": ion_temperature,
        "ion_temperature_peaking_factor": 0.0,
        "ion_temperature_beta": 1.0,
        "ion_temperature_pedestal": ion_temperature,
        "ion_temperature_separatrix": ion_temperature,
        "pedestal_radius": 0.8 * 180.0,
        "fuel": dict(fuel or {"D": 0.5, "T": 0.5}),
    }


def helias5b_convergence_study(
    sample_sizes: Sequence[Tuple[int, int, int]],
    *,
    mesh_resolution: Tuple[int, int, int] = (24, 24, 24),
    case: Helias5BScaleCase = Helias5BScaleCase(),
    profile_parameters: Mapping[str, object] | None = None,
) -> Tuple[Dict[str, object], ...]:
    """Run the source sampler at several resolutions for a scale benchmark.

    The returned records contain the input resolution, integrated volume,
    source strength, fusion power, and relative volume error against the
    analytical elliptical-torus reference.

    Parameters
    ----------
    sample_sizes : sequence of 3-tuples of int
        Quadrature resolutions ``(n_s, n_theta, n_zeta)`` to evaluate. At least
        one is required.
    mesh_resolution : tuple of int, default=(24, 24, 24)
        Cylindrical OpenMC mesh dimensions ``(n_R, n_phi, n_Z)`` shared by all
        runs.
    case : Helias5BScaleCase
        Geometry scale and boundary parameters. Its reference volume is used
        for the relative-error column.
    profile_parameters : mapping, optional
        Overrides for :func:`helias5b_uniform_profile`; the pedestal radius is
        capped at 80% of the case minor radius.

    Returns
    -------
    tuple of dict
        One record per resolution, containing ``sample_size``, integrated
        volume, total source strength, fusion power, relative volume error,
        and the full ``StellaratorSource`` result.

    Raises
    ------
    ValueError
        If ``sample_sizes`` is empty or any source-construction input is
        invalid.
    """

    if not sample_sizes:
        raise ValueError("sample_sizes must contain at least one resolution")
    profiles = dict(profile_parameters or helias5b_uniform_profile())
    profiles["pedestal_radius"] = min(
        float(profiles["pedestal_radius"]), 0.8 * case.minor_radius
    )
    records = []
    for sample_size in sample_sizes:
        result = stellarator_source(
            major_radius=case.major_radius,
            minor_radius=case.minor_radius,
            nfp=case.nfp,
            elongation=case.elongation,
            rotation=case.rotation,
            asymmetry=case.asymmetry,
            sample_size=sample_size,
            mesh_resolution=mesh_resolution,
            **profiles,
        )
        reference_volume = case.reference_volume_m3
        volume_error = (result.plasma_volume - reference_volume) / reference_volume
        records.append(
            {
                "sample_size": tuple(sample_size),
                "plasma_volume_m3": result.plasma_volume,
                "source_strength_n_per_s": result.source.strength,
                "fusion_power_mw": result.fusion_power,
                "relative_volume_error": volume_error,
                "result": result,
            }
        )
    return tuple(records)
