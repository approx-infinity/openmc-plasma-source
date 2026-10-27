import openmc

from openmc_plasma_source import stellarator_source


def test_public_stellarator_source_api():
    result = stellarator_source(
        major_radius=550.0,
        minor_radius=50.0,
        nfp=4,
        elongation=1.6,
        rotation=1.0,
        asymmetry=0.05,
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
        pedestal_radius=40.0,
        fuel={"D": 0.5, "T": 0.5},
        sample_size=(4, 8, 4),
        mesh_resolution=(6, 8, 6),
    )
    assert isinstance(result.source, openmc.MeshSource)
    assert result.source.strength > 0.0
    assert result.plasma_volume > 0.0
    assert result.fusion_power > 0.0
    assert result.surface.nfp == 4
    assert result.optimization_result is None
