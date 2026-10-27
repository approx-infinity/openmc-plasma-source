"""Public stellarator source construction API."""

from math import pi
from typing import Dict, Mapping, Sequence, Tuple

from .boundary import (
    rotating_ellipse_boundary_coefficients,
    rotating_ellipse_initial_guess,
)
from .geometry import FourierZernikeSurface
from .optimization import ActionGrid, CoefficientLayout
from .optimizer import optimize_surface
from .sampling import StellaratorSource, sample_stellarator_source


def _validate_basis_order(basis_order: Tuple[int, int, int]) -> None:
    """Validate the public ``(n_max, m_max, l_max)`` basis-order tuple.

    Raises ``ValueError`` unless it has three non-negative integer entries and
    ``l_max >= m_max``. Booleans are not accepted as integers.
    """

    if len(basis_order) != 3:
        raise ValueError("basis_order must be (n_max, m_max, l_max)")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in basis_order
    ):
        raise ValueError("basis_order must contain non-negative integers")
    if basis_order[2] < basis_order[1]:
        raise ValueError("basis_order requires l_max >= m_max")


def stellarator_source(
    major_radius: float,
    minor_radius: float,
    nfp: int,
    elongation: float,
    rotation: float,
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
    *,
    basis_order: Tuple[int, int, int] = (2, 2, 2),
    optimize_geometry: bool = False,
    optimization_grid: Tuple[int, int, int] = (4, 16, 8),
    omega: float = 0.01,
    optimizer_options: Mapping[str, float | int] | None = None,
    optimizer_jit: bool = True,
    asymmetry: float = 0.0,
) -> StellaratorSource:
    """Build an OpenMC neutron source for a rotating-ellipse stellarator.

    Geometry inputs use centimetres. Density inputs use m^-3 and temperature
    inputs use eV, matching the existing tokamak source API. The returned
    plasma volume is in m^3 and fusion power is in MW.

    The rotating-ellipse parameters define the external boundary. The
    boundary is represented in the Fourier–Zernike basis, and the analytical
    ``l=m=1`` extension is used by default. Set ``optimize_geometry=True`` and
    increase ``basis_order`` to optimize higher-order interior coefficients
    using the optional JAX/SciPy workflow. The standard source calculation
    uses NumPy and does not require JAX.

    Parameters
    ----------
    major_radius, minor_radius:
        Major and minor radii in cm.
    nfp:
        Number of field periods.
    elongation:
        Elliptical elongation of the rotating-ellipse boundary.
    rotation:
        Integer number of visible poloidal ellipse rotations per field period.
        ``rotation=1`` advances the ellipse orientation by pi over one field
        period, which is one complete ellipse orientation cycle because an
        ellipse is invariant under a pi rotation.
    mode and profile parameters:
        The complete tokamak-compatible profile parameter set. Temperatures are
        public inputs in eV.
    sample_size:
        Scalar quadrature resolution ``(n_s, n_theta, n_zeta)``.
    mesh_resolution:
        OpenMC cylindrical mesh resolution ``(n_R, n_phi, n_Z)``.
    basis_order:
        Fourier–Zernike order ``(n_max, m_max, l_max)``.
    optimize_geometry:
        If true, optimize free higher-order interior coefficients; install the
        ``stellarator-optimization`` extra to enable this option. The default
        false path uses the analytical Zernike–Fourier rotating-ellipse map.
    optimization_grid:
        Scalar action grid ``(n_s, n_theta, n_zeta)`` used by SciPy/JAX.
    asymmetry:
        Dimensionless amplitude of the field-period-compatible ``m=2,
        n=rotation`` perturbation. Zero recovers the symmetric ellipse.
    omega:
        Straightness weighting in the discrete theory action.

    Returns
    -------
    StellaratorSource
        Contains the OpenMC mesh source, integrated plasma volume in m^3,
        fusion power in MW, the Fourier–Zernike surface, and (when geometry
        optimization ran) its optimization result.

    Raises
    ------
    ValueError
        If geometry, profile, resolution, or basis parameters are inconsistent
        or invalid. Optimization may also raise if it fails its configured
        convergence or surface-validity requirements.
    """

    _validate_basis_order(basis_order)
    if asymmetry != 0.0 and (basis_order[1] < 2 or basis_order[2] < 2):
        raise ValueError(
            "nonzero asymmetry requires basis_order m_max >= 2 and l_max >= 2"
        )
    if len(optimization_grid) != 3 or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in optimization_grid
    ):
        raise ValueError(
            "optimization_grid must contain three positive integers"
        )

    r_boundary, z_boundary = rotating_ellipse_boundary_coefficients(
        major_radius=major_radius,
        minor_radius=minor_radius,
        nfp=nfp,
        elongation=elongation,
        rotation=rotation,
        asymmetry=asymmetry,
    )
    r_initial, z_initial = rotating_ellipse_initial_guess(
        major_radius=major_radius,
        minor_radius=minor_radius,
        nfp=nfp,
        elongation=elongation,
        rotation=rotation,
        asymmetry=asymmetry,
    )
    rotation_index = int(float(rotation))
    if basis_order[0] < abs(rotation_index):
        raise ValueError(
            "basis_order n_max must be at least abs(rotation) for the "
            "rotating-ellipse boundary"
        )
    layout = CoefficientLayout.from_orders(
        n_max=basis_order[0],
        m_max=basis_order[1],
        l_max=basis_order[2],
        r_boundary=r_boundary,
        z_boundary=z_boundary,
        twisted_rotation=rotation_index,
    )
    initial_vector = layout.initial_vector(r_initial, z_initial)

    optimization_result = None
    if optimize_geometry and layout.n_free > 0:
        try:
            optimization_result = optimize_surface(
                layout=layout,
                nfp=nfp,
                grid=ActionGrid.uniform(*optimization_grid),
                omega=omega,
                initial_vector=initial_vector,
                r_boundary=r_boundary,
                z_boundary=z_boundary,
                options=optimizer_options,
                jit=optimizer_jit,
                field_period_poloidal_shift=rotation_index * pi,
                geometry_description=(
                    f"R_0={major_radius / 100:g}m, a={minor_radius / 100:g}m"
                ),
            )
        except ModuleNotFoundError as exc:
            if exc.name in {"jax", "jaxlib", "scipy"}:
                raise ImportError(
                    "Geometry optimization requires JAX and SciPy. Install "
                    'them with `pip install "openmc-plasma-source'
                    '[stellarator-optimization]"`.'
                ) from exc
            raise
        surface = optimization_result.surface
    else:
        r_coefficients, z_coefficients = layout.coefficients_from_vector(
            initial_vector
        )
        surface = FourierZernikeSurface(
            r_coefficients,
            z_coefficients,
            nfp,
            field_period_poloidal_shift=rotation_index * pi,
        )

    result = sample_stellarator_source(
        surface=surface,
        minor_radius=minor_radius,
        mode=mode,
        ion_density_centre=ion_density_centre,
        ion_density_peaking_factor=ion_density_peaking_factor,
        ion_density_pedestal=ion_density_pedestal,
        ion_density_separatrix=ion_density_separatrix,
        ion_temperature_centre=ion_temperature_centre,
        ion_temperature_peaking_factor=ion_temperature_peaking_factor,
        ion_temperature_beta=ion_temperature_beta,
        ion_temperature_pedestal=ion_temperature_pedestal,
        ion_temperature_separatrix=ion_temperature_separatrix,
        pedestal_radius=pedestal_radius,
        fuel=fuel,
        sample_size=sample_size,
        mesh_resolution=mesh_resolution,
    )
    result.optimization_result = optimization_result
    return result
