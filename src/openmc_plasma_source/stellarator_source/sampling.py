"""Scalar Jacobian-weighted stellarator source sampling."""

from bisect import bisect_right
from dataclasses import dataclass
from math import isclose, isfinite, pi
from typing import Dict, Tuple

import numpy as np
import openmc
from openmc import IndependentSource

from ..fuel_types import get_neutron_energy_distribution
from .emission import fusion_emission_densities
from .geometry import FourierZernikeSurface
from .profiles import ion_density_profile, ion_temperature_profile


@dataclass
class StellaratorSource:
    """Products of integrating a stellarator plasma and building its source.

    Attributes
    ----------
    source : openmc.MeshSource
        Cylindrical mesh source whose cell strengths are neutron production
        rates in neutrons/s.
    plasma_volume : float
        Integrated plasma volume in m^3.
    fusion_power : float
        Integrated fusion power in MW.
    surface : FourierZernikeSurface
        Coordinate surface used for the quadrature and source construction.
    optimization_result : object or None
        Optional optimization result attached by the high-level
        ``stellarator_source`` factory; direct calls to the sampler leave it
        unset.
    """

    source: openmc.MeshSource
    plasma_volume: float
    fusion_power: float
    surface: FourierZernikeSurface
    optimization_result: object | None = None


def _validate_resolution(name: str, resolution: Tuple[int, ...], dimensions: int) -> None:
    """Require a resolution tuple with the requested number of positive ints.

    Parameters
    ----------
    name : str
        Parameter name used in any error message.
    resolution : tuple of int
        Number of cells along each coordinate axis.
    dimensions : int
        Required tuple length.

    Raises
    ------
    ValueError
        If the length differs from ``dimensions`` or a value is non-positive,
        non-integral, or boolean.
    """

    if len(resolution) != dimensions:
        raise ValueError(f"{name} must contain {dimensions} positive integers")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in resolution
    ):
        raise ValueError(f"{name} must contain {dimensions} positive integers")


def _mesh_bin(edges: Tuple[float, ...], value: float) -> int:
    """Return the half-open mesh interval containing one coordinate value.

    Bins are ``[edges[i], edges[i+1])`` except that a value equal (or within
    the endpoint tolerance) to the final edge is assigned to the last bin.

    Parameters
    ----------
    edges : tuple of float
        Strictly increasing mesh edges.
    value : float
        Coordinate to locate.

    Returns
    -------
    int
        Zero-based bin index.

    Raises
    ------
    ValueError
        If ``value`` lies outside the edge range.
    """

    index = bisect_right(edges, value) - 1
    if index == len(edges) - 1 and value == edges[-1]:
        index -= 1
    if index == len(edges) - 1 and isclose(value, edges[-1], rel_tol=1.0e-12, abs_tol=1.0e-12):
        index -= 1
    if index < 0 and isclose(value, edges[0], rel_tol=1.0e-12, abs_tol=1.0e-12):
        index = 0
    if index < 0 or index >= len(edges) - 1:
        raise ValueError("sampled point lies outside the cylindrical mesh bounds")
    return index


def sample_stellarator_source(
    surface: FourierZernikeSurface,
    minor_radius: float,
    mode: str,
    ion_density_centre: float,
    ion_density_peaking_factor: float,
    ion_density_pedestal: float,
    ion_density_separatrix: float,
    ion_temperature_centre: float,
    ion_temperature_peaking_factor: float,
    ion_temperature_beta: float,
    ion_temperature_pedestal: float,
    ion_temperature_separatrix: float,
    pedestal_radius: float,
    fuel: Dict[str, float] = {"D": 0.5, "T": 0.5},
    sample_size: Tuple[int, int, int] = (100, 100, 100),
    mesh_resolution: Tuple[int, int, int] = (100, 100, 100),
) -> StellaratorSource:
    """Integrate fusion emission over a surface and bin it into an OpenMC mesh.

    Geometry inputs use centimetres. Density inputs use m^-3 and temperature
    inputs use eV, matching the public tokamak source convention. The returned
    plasma volume is in m^3 and fusion power is in MW.

    The quadrature is a scalar midpoint rule in normalized radius, poloidal
    angle, and toroidal angle. Each contribution is weighted by

    ``sqrt(g) * ds * dtheta * dzeta``

    in cm^3 and converted to m^3 exactly once when applying the emission and
    power densities.

    Parameters
    ----------
    surface : FourierZernikeSurface
        Coordinate mapping to integrate. Every quadrature point must have
        positive cylindrical radius and Jacobian.
    minor_radius : float
        Plasma minor radius in centimetres, used to convert normalized ``s``
        to the profile radius ``r=s*minor_radius``.
    mode : str
        Density and temperature profile mode (L, H, or A).
    ion_density_centre, ion_density_pedestal, ion_density_separatrix : float
        Density profile levels in m^-3.
    ion_density_peaking_factor : float
        Density core-profile exponent.
    ion_temperature_centre, ion_temperature_pedestal,
    ion_temperature_separatrix : float
        Temperature profile levels in eV.
    ion_temperature_peaking_factor, ion_temperature_beta : float
        Temperature core-profile exponents.
    pedestal_radius : float
        Pedestal location in centimetres; must be positive and less than the
        minor radius.
    fuel : mapping of str to float
        Fuel species fractions, for example deuterium/tritium fractions.
    sample_size : tuple of int
        Midpoint quadrature counts ``(n_s, n_theta, n_zeta)``.
    mesh_resolution : tuple of int
        Cylindrical source mesh counts ``(n_R, n_phi, n_Z)``.

    Returns
    -------
    StellaratorSource
        Mesh source plus the integrated plasma volume, fusion power, and
        surface. Each mesh source strength is the integrated neutron rate for
        samples that landed in that cell.

    Raises
    ------
    ValueError
        If dimensions/resolutions are invalid, any sampled point has a
        non-positive Jacobian or radius, total neutron production is zero, or
        the sampled geometry has degenerate mesh bounds.
    """

    if minor_radius <= 0.0 or not isfinite(float(minor_radius)):
        raise ValueError("minor_radius must be positive and finite")
    if pedestal_radius <= 0.0 or pedestal_radius >= minor_radius:
        raise ValueError("pedestal_radius must be positive and less than minor_radius")
    _validate_resolution("sample_size", sample_size, 3)
    _validate_resolution("mesh_resolution", mesh_resolution, 3)

    n_s, n_theta, n_zeta = sample_size
    n_r, n_phi, n_z = mesh_resolution
    ds = 1.0 / n_s
    dtheta = 2.0 * pi / n_theta
    dzeta = 2.0 * pi / n_zeta

    samples = []
    plasma_volume_cm3 = 0.0
    total_source_rate = 0.0
    total_fusion_power_w = 0.0
    r_min = float("inf")
    r_max = float("-inf")
    z_min = float("inf")
    z_max = float("-inf")

    for i in range(n_s):
        s = (i + 0.5) * ds
        minor_radius_at_sample = s * minor_radius
        density = float(
            ion_density_profile(
                mode=mode,
                ion_density_centre=ion_density_centre,
                ion_density_peaking_factor=ion_density_peaking_factor,
                ion_density_pedestal=ion_density_pedestal,
                minor_radius=minor_radius,
                pedestal_radius=pedestal_radius,
                ion_density_separatrix=ion_density_separatrix,
                r=minor_radius_at_sample,
            )
        )
        temperature = float(
            ion_temperature_profile(
                r=minor_radius_at_sample,
                mode=mode,
                pedestal_radius=pedestal_radius,
                ion_temperature_pedestal=ion_temperature_pedestal / 1.0e3,
                ion_temperature_centre=ion_temperature_centre / 1.0e3,
                ion_temperature_beta=ion_temperature_beta,
                ion_temperature_peaking_factor=ion_temperature_peaking_factor,
                ion_temperature_separatrix=ion_temperature_separatrix / 1.0e3,
                minor_radius=minor_radius,
            )
        )
        source_density, power_density = fusion_emission_densities(
            density, temperature, fuel
        )
        source_density = float(source_density)
        power_density = float(power_density)

        for j in range(n_theta):
            theta = (j + 0.5) * dtheta
            for k in range(n_zeta):
                zeta = (k + 0.5) * dzeta
                evaluation = surface.evaluate(s, theta, zeta)
                jacobian = evaluation.jacobian
                if not isfinite(jacobian) or jacobian <= 0.0:
                    raise ValueError(
                        "stellarator surface has a non-positive Jacobian at a "
                        "sampled quadrature point"
                    )
                if evaluation.R <= 0.0 or not isfinite(evaluation.R):
                    raise ValueError(
                        "stellarator surface has a non-positive cylindrical radius"
                    )

                cell_volume_cm3 = jacobian * ds * dtheta * dzeta
                cell_volume_m3 = cell_volume_cm3 * 1.0e-6
                source_rate = source_density * cell_volume_m3
                fusion_power_w = power_density * cell_volume_m3
                phi = zeta % (2.0 * pi)

                samples.append(
                    (evaluation.R, phi, evaluation.Z, source_rate, temperature)
                )
                plasma_volume_cm3 += cell_volume_cm3
                total_source_rate += source_rate
                total_fusion_power_w += fusion_power_w
                r_min = min(r_min, evaluation.R)
                r_max = max(r_max, evaluation.R)
                z_min = min(z_min, evaluation.Z)
                z_max = max(z_max, evaluation.Z)

    if total_source_rate <= 0.0 or not isfinite(total_source_rate):
        raise ValueError(
            "Total neutron source density is zero. This may be caused by ion "
            "temperatures or densities that are too low to produce fusion reactions."
        )
    if r_min >= r_max or z_min >= z_max:
        raise ValueError("sampled stellarator geometry has degenerate mesh bounds")

    r_grid = tuple(r_min + index * (r_max - r_min) / n_r for index in range(n_r + 1))
    phi_grid = tuple(index * 2.0 * pi / n_phi for index in range(n_phi + 1))
    z_grid = tuple(z_min + index * (z_max - z_min) / n_z for index in range(n_z + 1))
    mesh = openmc.CylindricalMesh(
        r_grid=r_grid,
        phi_grid=phi_grid,
        z_grid=z_grid,
    )

    strengths = [
        [[0.0 for _ in range(n_z)] for _ in range(n_phi)] for _ in range(n_r)
    ]
    temperature_weights = [
        [[0.0 for _ in range(n_z)] for _ in range(n_phi)] for _ in range(n_r)
    ]
    for R, phi, Z, source_rate, temperature in samples:
        i = _mesh_bin(r_grid, R)
        j = _mesh_bin(phi_grid, phi)
        k = _mesh_bin(z_grid, Z)
        strengths[i][j][k] += source_rate
        temperature_weights[i][j][k] += source_rate * temperature

    sources = np.empty((n_r, n_phi, n_z), dtype=object)
    for i in range(n_r):
        for j in range(n_phi):
            for k in range(n_z):
                source = IndependentSource()
                strength = strengths[i][j][k]
                if strength > 0.0:
                    temperature = temperature_weights[i][j][k] / strength
                    source.energy = get_neutron_energy_distribution(
                        ion_temperature=temperature,
                        fuel=fuel,
                    )
                    source.angle = openmc.stats.Isotropic()
                source.strength = strength
                sources[i, j, k] = source

    return StellaratorSource(
        source=openmc.MeshSource(mesh, sources),
        plasma_volume=plasma_volume_cm3 * 1.0e-6,
        fusion_power=total_fusion_power_w * 1.0e-6,
        surface=surface,
    )
