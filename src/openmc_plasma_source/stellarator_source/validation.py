"""Numerical validation routines for Fourier–Zernike surfaces."""

from math import isfinite, pi
from typing import Dict, Iterable, Mapping, Tuple

from .geometry import FourierZernikeSurface


def boundary_error(
    surface: FourierZernikeSurface,
    r_boundary: Mapping[Tuple[int, int], float],
    z_boundary: Mapping[Tuple[int, int], float],
) -> float:
    """Measure mismatch between a surface boundary and prescribed coefficients.

    Coefficients with missing keys are treated as zero, so this comparison
    checks both extra surface modes and missing requested modes.

    Parameters
    ----------
    surface : FourierZernikeSurface
        Surface whose ``s=1`` Fourier coefficients are inspected.
    r_boundary, z_boundary : mapping of (n, m) to float
        Expected cosine coefficients for R and sine coefficients for Z.

    Returns
    -------
    float
        Largest absolute coefficient difference across R and Z.
    """

    actual_r, actual_z = surface.boundary_coefficients()
    keys_r = set(actual_r) | set(r_boundary)
    keys_z = set(actual_z) | set(z_boundary)
    error = 0.0
    for key in keys_r:
        error = max(error, abs(actual_r.get(key, 0.0) - r_boundary.get(key, 0.0)))
    for key in keys_z:
        error = max(error, abs(actual_z.get(key, 0.0) - z_boundary.get(key, 0.0)))
    return error


def periodicity_error(
    surface: FourierZernikeSurface,
    s_values: Iterable[float],
    theta_values: Iterable[float],
    zeta_values: Iterable[float],
) -> float:
    """Return the maximum error after one field-period toroidal shift.

    A twisted surface compares the shifted toroidal section after applying its
    poloidal seam shift. This permits ``rotation=1`` to use the half-angle
    ellipse representation while retaining exact field-period closure as a
    physical surface.

    Parameters
    ----------
    surface : FourierZernikeSurface
        Surface and its field-period count/seam shift.
    s_values, theta_values, zeta_values : iterable of float
        Coordinates at which to compare positions. Angles are in radians.

    Returns
    -------
    float
        Maximum absolute difference in R or Z between the two equivalent
        representations of a field-period-shifted point.
    """

    error = 0.0
    field_period = 2.0 * pi / surface.nfp
    for s in s_values:
        for theta in theta_values:
            for zeta in zeta_values:
                first = surface.evaluate(
                    s,
                    theta + surface.field_period_poloidal_shift,
                    zeta,
                )
                second = surface.evaluate(s, theta, zeta + field_period)
                error = max(
                    error,
                    abs(first.R - second.R),
                    abs(first.Z - second.Z),
                )
    return error


def stellarator_symmetry_error(
    surface: FourierZernikeSurface,
    s_values: Iterable[float],
    theta_values: Iterable[float],
    zeta_values: Iterable[float],
) -> float:
    """Measure deviation from the package's stellarator-symmetry identities.

    The tested identities are ``R(s, theta, zeta)=R(s, -theta, -zeta)`` and
    ``Z(s, theta, zeta)=-Z(s, -theta, -zeta)``.

    Parameters
    ----------
    surface : FourierZernikeSurface
        Surface to inspect.
    s_values, theta_values, zeta_values : iterable of float
        Sample coordinates; angular values are in radians.

    Returns
    -------
    float
        Maximum absolute residual in either symmetry identity.
    """

    error = 0.0
    for s in s_values:
        for theta in theta_values:
            for zeta in zeta_values:
                first = surface.evaluate(s, theta, zeta)
                reflected = surface.evaluate(s, -theta, -zeta)
                error = max(
                    error,
                    abs(first.R - reflected.R),
                    abs(first.Z + reflected.Z),
                )
    return error


def radial_spacing_diagnostics(
    surface: FourierZernikeSurface,
    s_values: Iterable[float],
    theta_values: Iterable[float],
    zeta_values: Iterable[float],
) -> Dict[str, float]:
    """Measure spacing between consecutive sampled radial surfaces.

    Positive Jacobian provides local invertibility. This additional check
    ensures that adjacent sampled surfaces do not collapse onto one another
    along the same coordinate line. It is a numerical diagnostic, not a
    mathematical proof of global non-intersection.

    Parameters
    ----------
    surface : FourierZernikeSurface
        Mapping to sample.
    s_values : iterable of float
        At least two strictly increasing radial values.
    theta_values, zeta_values : iterable of float
        Fixed angular locations at which to compare adjacent radial samples.

    Returns
    -------
    dict of str to float
        Contains ``minimum_radial_spacing`` in the same length units as the
        surface coefficients.

    Raises
    ------
    ValueError
        If fewer than two radii are supplied, radii are not strictly
        increasing, any spacing is non-finite, or adjacent points coincide.
    """

    radii = list(s_values)
    if len(radii) < 2:
        raise ValueError("at least two radial values are required")
    if any(radii[index] >= radii[index + 1] for index in range(len(radii) - 1)):
        raise ValueError("s_values must be strictly increasing")

    minimum_spacing = float("inf")
    for theta in theta_values:
        for zeta in zeta_values:
            previous = surface.evaluate(radii[0], theta, zeta)
            for s in radii[1:]:
                current = surface.evaluate(s, theta, zeta)
                spacing = ((current.R - previous.R) ** 2 + (current.Z - previous.Z) ** 2) ** 0.5
                if not isfinite(spacing):
                    raise ValueError("radial surface spacing is not finite")
                minimum_spacing = min(minimum_spacing, spacing)
                previous = current

    if minimum_spacing <= 0.0:
        raise ValueError("adjacent sampled surfaces collapse onto one another")
    return {"minimum_radial_spacing": minimum_spacing}


def validate_surface(
    surface: FourierZernikeSurface,
    s_values: Iterable[float],
    theta_values: Iterable[float],
    zeta_values: Iterable[float],
    *,
    r_boundary: Mapping[Tuple[int, int], float] | None = None,
    z_boundary: Mapping[Tuple[int, int], float] | None = None,
    tolerance: float = 1.0e-10,
) -> Dict[str, float]:
    """Run sampled geometry, boundary, symmetry, and periodicity checks.

    The checks include positive cylindrical radius and Jacobian through
    :meth:`FourierZernikeSurface.validate`, nonzero spacing of adjacent radial
    samples, field-period closure, stellarator symmetry, and optional boundary
    coefficient agreement. This is a finite-grid diagnostic; passing it does
    not prove global injectivity between sample points.

    Parameters
    ----------
    surface : FourierZernikeSurface
        Mapping to validate.
    s_values, theta_values, zeta_values : iterable of float
        Sample coordinates. ``s_values`` must be strictly increasing for the
        radial-spacing check; angles are in radians.
    r_boundary, z_boundary : mapping, optional
        Expected boundary coefficients. Supply both or neither.
    tolerance : float, default=1e-10
        Maximum accepted residual for periodicity, symmetry, and boundary
        checks.

    Returns
    -------
    dict of str to float
        Jacobian/radius extrema, minimum radial spacing, symmetry and
        periodicity residuals, and (if requested) boundary error.

    Raises
    ------
    ValueError
        If any geometry check fails, a residual exceeds ``tolerance``, or
        only one of the two boundary maps is supplied.
    """

    radii = list(s_values)
    thetas = list(theta_values)
    zetas = list(zeta_values)
    diagnostics = surface.validate(radii, thetas, zetas)
    diagnostics.update(radial_spacing_diagnostics(surface, radii, thetas, zetas))
    diagnostics["periodicity_error"] = periodicity_error(
        surface, radii, thetas, zetas
    )
    diagnostics["stellarator_symmetry_error"] = stellarator_symmetry_error(
        surface, radii, thetas, zetas
    )

    if diagnostics["periodicity_error"] > tolerance:
        raise ValueError("surface does not satisfy field-period periodicity")
    if diagnostics["stellarator_symmetry_error"] > tolerance:
        raise ValueError("surface does not satisfy stellarator symmetry")

    if (r_boundary is None) != (z_boundary is None):
        raise ValueError("both R and Z boundary coefficients are required")
    if r_boundary is not None and z_boundary is not None:
        diagnostics["boundary_error"] = boundary_error(
            surface, r_boundary, z_boundary
        )
        if diagnostics["boundary_error"] > tolerance:
            raise ValueError("surface does not satisfy the supplied boundary")

    return diagnostics
