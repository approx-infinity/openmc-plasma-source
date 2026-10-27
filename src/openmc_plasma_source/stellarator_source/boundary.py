"""Analytical boundary and initial coefficients for the rotating ellipse."""

from math import cos, isfinite, sin
from typing import Dict, Tuple

ModeKey = Tuple[float, int, int]
BoundaryKey = Tuple[float, int]


def _validate_geometry_inputs(
    major_radius: float,
    minor_radius: float,
    nfp: int,
    elongation: float,
    rotation: float,
    asymmetry: float,
) -> int:
    """Validate ellipse parameters and return the integer rotation index.

    Lengths are expressed in centimetres. The rotation must be integral
    because the Fourier representation stores toroidal frequencies as
    ``n = +/- rotation/2`` (and the optional asymmetry as ``n=rotation``).

    Raises
    ------
    ValueError
        If a radius is non-finite or non-positive, the minor radius is not
        smaller than the major radius, ``nfp`` is not a positive integer,
        elongation is non-positive, rotation is non-integral, or asymmetry is
        non-finite.
    """

    if not isfinite(float(major_radius)) or major_radius <= 0.0:
        raise ValueError("major_radius must be positive and finite")
    if not isfinite(float(minor_radius)) or minor_radius <= 0.0:
        raise ValueError("minor_radius must be positive and finite")
    if minor_radius >= major_radius:
        raise ValueError("minor_radius must be less than major_radius")
    if not isinstance(nfp, int) or isinstance(nfp, bool) or nfp <= 0:
        raise ValueError("nfp must be a positive integer")
    if not isfinite(float(elongation)) or elongation <= 0.0:
        raise ValueError("elongation must be positive and finite")
    if not isfinite(float(rotation)) or not float(rotation).is_integer():
        raise ValueError("rotation must be an integer for the Fourier basis")
    if not isfinite(float(asymmetry)):
        raise ValueError("asymmetry must be finite")
    return int(rotation)


def _add(mapping: Dict, key, value: float) -> None:
    """Add a Fourier contribution to an existing coefficient dictionary.

    Contributions sharing a key are summed. Values are converted to Python
    floats so the dictionaries can be passed directly to the geometry layer.
    """

    mapping[key] = mapping.get(key, 0.0) + float(value)


def rotating_ellipse_boundary_coefficients(
    major_radius: float,
    minor_radius: float,
    nfp: int,
    elongation: float,
    rotation: float,
    asymmetry: float = 0.0,
) -> Tuple[Dict[BoundaryKey, float], Dict[BoundaryKey, float]]:
    """Construct Fourier coefficients for a rotating elliptical boundary.

    Geometry inputs use centimetres. The Fourier convention is
    ``cos(m*theta - n*nfp*zeta)`` for ``R`` and
    ``sin(m*theta - n*nfp*zeta)`` for ``Z``. Here ``rotation`` counts visible
    poloidal orientation rotations of the ellipse per field period, so
    ``delta = rotation*nfp*zeta/2``. The resulting m=1 ellipse modes have
    toroidal frequencies ``n=+/-rotation/2``. The optional asymmetry is the
    field-period-compatible ``m=2, n=rotation`` perturbation.

    Parameters
    ----------
    major_radius, minor_radius : float
        Major and minor radii in centimetres, with ``0 < minor_radius <
        major_radius``.
    nfp : int
        Number of identical field periods in a full toroidal turn.
    elongation : float
        Ratio of vertical to horizontal ellipse half-width; must be positive.
    rotation : float
        Integer number of visible poloidal orientation rotations per field
        period. Fractional values are rejected by the Fourier representation.
    asymmetry : float, default=0
        Dimensionless amplitude of the field-period-compatible second-harmonic
        perturbation. Zero gives a symmetric rotating ellipse.

    Returns
    -------
    tuple of dict
        ``(r_boundary, z_boundary)``. Each dictionary maps ``(n, m)`` to the
        cosine coefficient for R or sine coefficient for Z, using phase
        ``m*theta - n*nfp*zeta``.

    Raises
    ------
    ValueError
        If the geometry parameters violate the conditions described above.
    """

    q = _validate_geometry_inputs(
        major_radius, minor_radius, nfp, elongation, rotation, asymmetry
    )
    r_boundary: Dict[BoundaryKey, float] = {(0, 0): float(major_radius)}
    z_boundary: Dict[BoundaryKey, float] = {}

    toroidal_frequency = 0.5 * q
    _add(
        r_boundary,
        (-toroidal_frequency, 1),
        0.5 * minor_radius * (1.0 + elongation),
    )
    _add(
        r_boundary,
        (toroidal_frequency, 1),
        0.5 * minor_radius * (1.0 - elongation),
    )
    _add(
        z_boundary,
        (-toroidal_frequency, 1),
        0.5 * minor_radius * (1.0 + elongation),
    )
    _add(
        z_boundary,
        (toroidal_frequency, 1),
        0.5 * minor_radius * (elongation - 1.0),
    )
    if asymmetry != 0.0:
        _add(r_boundary, (float(q), 2), asymmetry * minor_radius)
        _add(z_boundary, (float(q), 2), asymmetry * minor_radius)
    return r_boundary, z_boundary


def rotating_ellipse_initial_guess(
    major_radius: float,
    minor_radius: float,
    nfp: int,
    elongation: float,
    rotation: float,
    asymmetry: float = 0.0,
) -> Tuple[Dict[ModeKey, float], Dict[ModeKey, float]]:
    """Build the regular lowest-radial-order extension of the boundary.

    Each non-axisymmetric boundary coefficient is assigned its lowest
    admissible Zernike order ``l=m``. The axisymmetric major-radius term is
    stored as ``(n,m,l)=(0,0,0)``. This constructs a smooth analytical initial
    surface but introduces no higher-order interior optimization modes.

    Parameters are identical to :func:`rotating_ellipse_boundary_coefficients`.

    Returns
    -------
    tuple of dict
        ``(r_coefficients, z_coefficients)``, keyed by ``(n, m, l)`` and ready
        for :class:`FourierZernikeSurface` or :class:`CoefficientLayout`.
    """

    r_boundary, z_boundary = rotating_ellipse_boundary_coefficients(
        major_radius, minor_radius, nfp, elongation, rotation, asymmetry
    )
    r_coefficients: Dict[ModeKey, float] = {(0, 0, 0): r_boundary[(0, 0)]}
    z_coefficients: Dict[ModeKey, float] = {}
    for (n, m), coefficient in r_boundary.items():
        if m != 0:
            r_coefficients[(n, m, m)] = coefficient
    for (n, m), coefficient in z_boundary.items():
        if m != 0:
            z_coefficients[(n, m, m)] = coefficient
    return r_coefficients, z_coefficients


def rotating_ellipse_coordinates(
    s: float,
    theta: float,
    zeta: float,
    major_radius: float,
    minor_radius: float,
    nfp: int,
    elongation: float,
    rotation: float,
    asymmetry: float = 0.0,
) -> Tuple[float, float]:
    """Evaluate the closed-form rotating-ellipse boundary and interior map.

    This function provides an independent coordinate formula for checking the
    Fourier coefficients. The ellipse orientation is
    ``delta = rotation*nfp*zeta/2``; the cross-section size scales linearly
    with ``s`` and the optional second harmonic scales as ``s**2``.

    Parameters
    ----------
    s : float
        Normalized minor-radius coordinate; ``s=0`` is the axis and ``s=1`` is
        the external boundary.
    theta, zeta : float
        Poloidal and toroidal angles in radians.
    major_radius, minor_radius : float
        Radii in centimetres.
    nfp : int
        Number of field periods.
    elongation : float
        Vertical-to-horizontal ellipse elongation.
    rotation : float
        Integer orientation rotation per field period.
    asymmetry : float, default=0
        Dimensionless amplitude of the second-harmonic perturbation.

    Returns
    -------
    tuple of float
        Cylindrical coordinates ``(R, Z)`` in centimetres.

    Raises
    ------
    ValueError
        If the geometry parameters are invalid.
    """

    q = _validate_geometry_inputs(
        major_radius, minor_radius, nfp, elongation, rotation, asymmetry
    )
    delta = 0.5 * q * nfp * zeta
    radial = minor_radius * s
    r_offset = radial * cos(theta) * cos(delta) - (
        elongation * radial * sin(theta) * sin(delta)
    )
    z = radial * cos(theta) * sin(delta) + (
        elongation * radial * sin(theta) * cos(delta)
    )
    asymmetry_phase = 2.0 * theta - q * nfp * zeta
    local_asymmetry = asymmetry * minor_radius * s**2
    r_offset += local_asymmetry * cos(asymmetry_phase)
    z += local_asymmetry * sin(asymmetry_phase)
    return major_radius + r_offset, z
