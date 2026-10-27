import numpy as np
import pytest

from openmc_plasma_source.stellarator_source import (
    fusion_emission_densities,
    ion_density_profile,
    ion_temperature_profile,
    reaction_rate_density,
)
from openmc_plasma_source.tokamak_source import (
    tokamak_ion_density,
    tokamak_ion_temperature,
    tokamak_reaction_rate,
)


def _profile_parameters():
    return {
        "ion_density_centre": 1.09e20,
        "ion_density_peaking_factor": 1.0,
        "ion_density_pedestal": 1.00e20,
        "ion_density_separatrix": 3.0e19,
        "ion_temperature_pedestal": 6.09,
        "ion_temperature_centre": 45.9,
        "ion_temperature_beta": 6.0,
        "ion_temperature_peaking_factor": 8.06,
        "ion_temperature_separatrix": 0.1,
        "minor_radius": 292.258,
        "pedestal_radius": 0.8 * 292.258,
    }


@pytest.mark.parametrize("mode", ["L", "H", "A"])
def test_shared_profiles_match_tokamak_wrappers(mode):
    parameters = _profile_parameters()
    radii = np.linspace(0.0, parameters["minor_radius"], 31)
    density_kwargs = {
        "mode": mode,
        "r": radii,
        **{
            key: parameters[key]
            for key in (
                "ion_density_centre",
                "ion_density_peaking_factor",
                "ion_density_pedestal",
                "minor_radius",
                "pedestal_radius",
                "ion_density_separatrix",
            )
        },
    }
    temperature_kwargs = {
        "r": radii,
        "mode": mode,
        **{
            key: parameters[key]
            for key in (
                "pedestal_radius",
                "ion_temperature_pedestal",
                "ion_temperature_centre",
                "ion_temperature_beta",
                "ion_temperature_peaking_factor",
                "ion_temperature_separatrix",
                "minor_radius",
            )
        },
    }
    assert np.array_equal(
        ion_density_profile(**density_kwargs), tokamak_ion_density(**density_kwargs)
    )
    assert np.array_equal(
        ion_temperature_profile(**temperature_kwargs),
        tokamak_ion_temperature(**temperature_kwargs),
    )


@pytest.mark.parametrize("reaction", ["DD", "DT", "TT"])
def test_shared_reaction_rate_matches_tokamak_wrapper(reaction):
    density = np.array([1.0e19, 2.0e19, 3.0e19])
    temperature = np.array([1.0e4, 1.5e4, 2.0e4])
    assert np.array_equal(
        reaction_rate_density(density, temperature, reaction),
        tokamak_reaction_rate(density, temperature, reaction),
    )


@pytest.mark.parametrize("fuel", [{"D": 1.0}, {"T": 1.0}, {"D": 0.5, "T": 0.5}])
def test_fusion_emission_densities_are_finite_and_nonnegative(fuel):
    density = np.array([1.0e19, 2.0e19])
    temperature = np.array([1.0e4, 2.0e4])
    source_density, power_density = fusion_emission_densities(
        density, temperature, fuel
    )
    assert source_density.shape == density.shape
    assert power_density.shape == density.shape
    assert np.all(np.isfinite(source_density))
    assert np.all(np.isfinite(power_density))
    assert np.all(source_density >= 0.0)
    assert np.all(power_density >= 0.0)
