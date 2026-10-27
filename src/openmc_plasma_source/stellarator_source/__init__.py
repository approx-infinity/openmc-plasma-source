"""Geometry primitives for the stellarator source implementation."""

from .basis import (
    ZernikeMode,
    valid_modes,
    zernike_radial,
    zernike_radial_derivative,
)
from .boundary import (
    rotating_ellipse_boundary_coefficients,
    rotating_ellipse_coordinates,
    rotating_ellipse_initial_guess,
)
from .benchmarks import (
    Helias5BScaleCase,
    helias5b_convergence_study,
    helias5b_uniform_profile,
)
from .geometry import (
    FourierZernikeSurface,
    SurfaceEvaluation,
)
from ..emission import (
    fusion_emission_densities,
    reaction_fuel_density,
    reaction_rate_density,
)
from .optimization import (
    ActionGrid,
    CoefficientLayout,
    jax_value_and_gradient,
    make_action,
    scalar_action_reference,
)
from .optimizer import (
    StellaratorOptimizationResult,
    make_geometry_penalty,
    make_optimization_objective,
    optimize_surface,
)
from ..profiles import ion_density_profile, ion_temperature_profile
from .sampling import StellaratorSource, sample_stellarator_source
from .source import stellarator_source
from .validation import (
    boundary_error,
    periodicity_error,
    radial_spacing_diagnostics,
    stellarator_symmetry_error,
    validate_surface,
)

__all__ = [
    "FourierZernikeSurface",
    "SurfaceEvaluation",
    "ZernikeMode",
    "ActionGrid",
    "CoefficientLayout",
    "Helias5BScaleCase",
    "StellaratorOptimizationResult",
    "StellaratorSource",
    "boundary_error",
    "fusion_emission_densities",
    "helias5b_convergence_study",
    "helias5b_uniform_profile",
    "ion_density_profile",
    "ion_temperature_profile",
    "jax_value_and_gradient",
    "make_geometry_penalty",
    "make_action",
    "make_optimization_objective",
    "optimize_surface",
    "periodicity_error",
    "radial_spacing_diagnostics",
    "reaction_fuel_density",
    "reaction_rate_density",
    "rotating_ellipse_boundary_coefficients",
    "rotating_ellipse_coordinates",
    "rotating_ellipse_initial_guess",
    "stellarator_symmetry_error",
    "scalar_action_reference",
    "sample_stellarator_source",
    "stellarator_source",
    "validate_surface",
    "valid_modes",
    "zernike_radial",
    "zernike_radial_derivative",
]
