import math

import numpy as np
import openmc
import pytest

from openmc_plasma_source.stellarator_source import (
    FourierZernikeSurface,
    fusion_emission_densities,
    rotating_ellipse_initial_guess,
    sample_stellarator_source,
)


def _constant_profile_source():
    major_radius = 550.0
    minor_radius = 50.0
    nfp = 4
    elongation = 1.6
    rotation = 1
    r_coefficients, z_coefficients = rotating_ellipse_initial_guess(
        major_radius, minor_radius, nfp, elongation, rotation
    )
    surface = FourierZernikeSurface(r_coefficients, z_coefficients, nfp)
    result = sample_stellarator_source(
        surface=surface,
        minor_radius=minor_radius,
        mode="L",
        ion_density_centre=4.8e20,
        ion_density_peaking_factor=0.0,
        ion_density_pedestal=4.8e20,
        ion_density_separatrix=4.8e20,
        ion_temperature_centre=12.0e3,
        ion_temperature_peaking_factor=0.0,
        ion_temperature_beta=1.0,
        ion_temperature_pedestal=12.0e3,
        ion_temperature_separatrix=12.0e3,
        pedestal_radius=0.8 * minor_radius,
        fuel={"D": 0.5, "T": 0.5},
        sample_size=(8, 16, 8),
        mesh_resolution=(10, 12, 10),
    )
    return result, surface, major_radius, minor_radius, elongation


def test_stellarator_sampling_returns_three_dimensional_mesh_source():
    result, _surface, _major_radius, _minor_radius, _elongation = (
        _constant_profile_source()
    )
    assert isinstance(result.source, openmc.MeshSource)
    assert len(result.source.mesh.r_grid) == 11
    assert len(result.source.mesh.phi_grid) == 13
    assert len(result.source.mesh.z_grid) == 11
    assert result.source.strength > 0.0
    assert result.plasma_volume > 0.0
    assert result.fusion_power > 0.0
    assert sum(source.strength for source in result.source.sources.flat) == pytest.approx(
        result.source.strength
    )


def test_uniform_stellarator_volume_source_and_power():
    result, _surface, major_radius, minor_radius, elongation = (
        _constant_profile_source()
    )
    expected_volume_m3 = (
        2.0
        * math.pi**2
        * major_radius
        * elongation
        * minor_radius**2
        * 1.0e-6
    )
    source_density, power_density = fusion_emission_densities(
        4.8e20, 12.0e3, {"D": 0.5, "T": 0.5}
    )
    assert result.plasma_volume == pytest.approx(expected_volume_m3, rel=2.0e-3)
    assert result.source.strength == pytest.approx(
        expected_volume_m3 * float(source_density), rel=2.0e-3
    )
    assert result.fusion_power == pytest.approx(
        expected_volume_m3 * float(power_density) * 1.0e-6, rel=2.0e-3
    )
