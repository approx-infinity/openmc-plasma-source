import math

import pytest

from openmc_plasma_source.stellarator_source import (
    Helias5BScaleCase,
    helias5b_convergence_study,
)


def test_helias5b_scale_case_constants():
    case = Helias5BScaleCase()
    assert case.major_radius == pytest.approx(2200.0)
    assert case.minor_radius == pytest.approx(180.0)
    assert case.nfp == 5
    assert case.analytic_volume_m3 == pytest.approx(
        2.0 * math.pi**2 * 22.0 * 1.8**2
    )


def test_helias5b_uniform_convergence_benchmark():
    records = helias5b_convergence_study(
        sample_sizes=((4, 8, 8), (8, 16, 16)),
        mesh_resolution=(8, 10, 8),
    )
    assert len(records) == 2
    assert records[0]["plasma_volume_m3"] == pytest.approx(
        records[1]["plasma_volume_m3"], rel=1.0e-12
    )
    assert records[0]["relative_volume_error"] == pytest.approx(0.0, abs=1.0e-12)
    assert records[1]["relative_volume_error"] == pytest.approx(0.0, abs=1.0e-12)
    assert records[0]["source_strength_n_per_s"] > 0.0
    assert records[0]["fusion_power_mw"] > 0.0


def test_asymmetric_helia_scale_volume_reference():
    case = Helias5BScaleCase(elongation=1.6, rotation=1, asymmetry=0.05)
    records = helias5b_convergence_study(
        sample_sizes=((4, 16, 16), (8, 32, 32)),
        mesh_resolution=(8, 10, 8),
        case=case,
    )
    assert abs(records[0]["relative_volume_error"]) < 1.0e-3
    assert abs(records[1]["relative_volume_error"]) < abs(
        records[0]["relative_volume_error"]
    )
