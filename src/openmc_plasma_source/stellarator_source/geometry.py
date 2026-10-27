"""Scalar Fourier–Zernike stellarator coordinate mapping."""

from dataclasses import dataclass
from math import cos, isfinite, sin
from typing import Dict, Iterable, Mapping, Tuple

from .basis import zernike_radial, zernike_radial_derivative

ModeKey = Tuple[float, int, int]


@dataclass(frozen=True)
class SurfaceEvaluation:
    """Coordinates and first derivatives at one flux-coordinate point.

    ``R`` and ``Z`` are cylindrical position coordinates. Derivatives are
    with respect to the normalized radius ``s`` and angles ``theta`` and
    ``zeta``. Position and derivative units follow the coefficient units;
    angles are in radians.
    """

    R: float
    Z: float
    R_s: float      # partial derivative of R with respect to s
    R_theta: float  # partial derivative of R with respect to theta
    R_zeta: float   # partial derivative of R with respect to zeta
    Z_s: float
    Z_theta: float
    Z_zeta: float

    @property
    def jacobian(self) -> float:
        """Return the signed cylindrical volume Jacobian.

        Uses the stellarator convention
        ``sqrt(g) = R * (R_s*Z_theta - R_theta*Z_s)``. A positive value is
        required by :meth:`FourierZernikeSurface.validate` away from the
        coordinate axis.
        """

        return self.R * (self.R_s * self.Z_theta - self.R_theta * self.Z_s)

    def cartesian_coordinates(self, zeta: float) -> Tuple[float, float, float]:
        """Convert this cylindrical point to Cartesian coordinates.

        Parameters
        ----------
        zeta : float
            Toroidal angle in radians. It is supplied explicitly because the
            evaluation record stores ``R`` and ``Z``, not its evaluation angle.

        Returns
        -------
        tuple of float
            ``(x, y, z) = (R*cos(zeta), R*sin(zeta), Z)`` in the same length
            units as ``R`` and ``Z``.
        """

        return (
            self.R * cos(zeta),
            self.R * sin(zeta),
            self.Z,
        )

    def cartesian_jacobian(self, zeta: float) -> float:
        """Compute the signed volume Jacobian from Cartesian basis vectors.

        The positive volume convention used by the source is
        ``e_s dot (e_zeta cross e_theta)``. The raw determinant
        ``e_s dot (e_theta cross e_zeta)`` has the opposite sign for the
        standard counter-clockwise poloidal angle used here.

        Parameters
        ----------
        zeta : float
            Toroidal angle in radians at which the Cartesian basis is formed.

        Returns
        -------
        float
            ``e_s dot (e_zeta cross e_theta)`` using the package's positive
            volume convention. It should agree with :attr:`jacobian` up to
            floating-point error.
        """

        cos_zeta = cos(zeta)
        sin_zeta = sin(zeta)
        e_s = (
            self.R_s * cos_zeta,
            self.R_s * sin_zeta,
            self.Z_s,
        )
        e_theta = (
            self.R_theta * cos_zeta,
            self.R_theta * sin_zeta,
            self.Z_theta,
        )
        e_zeta = (
            self.R_zeta * cos_zeta - self.R * sin_zeta,
            self.R_zeta * sin_zeta + self.R * cos_zeta,
            self.Z_zeta,
        )
        cross = (
            e_zeta[1] * e_theta[2] - e_zeta[2] * e_theta[1],
            e_zeta[2] * e_theta[0] - e_zeta[0] * e_theta[2],
            e_zeta[0] * e_theta[1] - e_zeta[1] * e_theta[0],
        )
        return e_s[0] * cross[0] + e_s[1] * cross[1] + e_s[2] * cross[2]


def _validate_coefficients(coefficients: Mapping[ModeKey, float]) -> Dict[ModeKey, float]:
    """Validate and copy a ``(n, m, l) -> coefficient`` mapping.

    Toroidal indices and coefficients must be finite; ``m`` and ``l`` must be
    integers with ``m >= 0``, ``l >= m``, and even ``l - m``. The returned
    dictionary contains normalized Python floats and tuple keys.

    Raises
    ------
    ValueError
        For malformed keys, non-finite numbers, or inadmissible radial modes.
    TypeError
        If ``m`` or ``l`` is not an integer.
    """
    result: Dict[ModeKey, float] = {}
    for key, coefficient in coefficients.items():
        if len(key) != 3:
            raise ValueError("coefficient keys must be (n, m, l) tuples")
        n, m, l = key
        if not isfinite(float(n)):
            raise ValueError("toroidal mode numbers must be finite")
        if not all(
            isinstance(index, int) and not isinstance(index, bool)
            for index in (m, l)
        ):
            raise TypeError("poloidal and radial mode indices must be integers")
        if m < 0 or l < m or (l - m) % 2:
            raise ValueError("coefficient mode must satisfy l >= m and even l-m")
        if not isfinite(float(coefficient)):
            raise ValueError("coefficients must be finite")
        result[(n, m, l)] = float(coefficient)
    return result


@dataclass
class FourierZernikeSurface:
    """Scalar Fourier–Zernike representation of nested coordinates.

    Coefficients are stored as mappings keyed by ``(n, m, l)``. The basis
    phase is ``m*theta - n*nfp*zeta``; R uses cosine terms and Z uses sine
    terms. The radial coordinate is normalized so ``s=1`` is the external
    boundary. Coefficients and returned lengths use the same physical unit.
    """

    r_coefficients: Mapping[ModeKey, float]
    z_coefficients: Mapping[ModeKey, float]
    nfp: int
    field_period_poloidal_shift: float = 0.0

    def __post_init__(self) -> None:
        """Validate field-period settings and normalize coefficient maps.

        Raises ``TypeError`` or ``ValueError`` when ``nfp``, the optional
        poloidal seam shift, or either coefficient mapping is invalid.
        """

        if not isinstance(self.nfp, int) or isinstance(self.nfp, bool):
            raise TypeError("nfp must be an integer")
        if self.nfp <= 0:
            raise ValueError("nfp must be positive")
        if not isfinite(float(self.field_period_poloidal_shift)):
            raise ValueError("field_period_poloidal_shift must be finite")
        self.r_coefficients = _validate_coefficients(self.r_coefficients)
        self.z_coefficients = _validate_coefficients(self.z_coefficients)

    def evaluate(self, s: float, theta: float, zeta: float) -> SurfaceEvaluation:
        """Evaluate position and first derivatives at ``(s, theta, zeta)``.

        Parameters
        ----------
        s : float
            Normalized radial coordinate; the usual domain is ``[0, 1]``.
        theta, zeta : float
            Poloidal and toroidal angles in radians.

        Returns
        -------
        SurfaceEvaluation
            Cylindrical position ``(R, Z)`` and derivatives ``R_s``,
            ``R_theta``, ``R_zeta``, ``Z_s``, ``Z_theta``, and ``Z_zeta``.

        Notes
        -----
        The coordinate values are evaluated directly from the stored
        Fourier–Zernike coefficients; this method does not check that the
        resulting surface is nested or has positive Jacobian.
        """

        R = 0.0
        R_s = 0.0
        R_theta = 0.0
        R_zeta = 0.0
        for (n, m, l), coefficient in self.r_coefficients.items():
            radial = zernike_radial(s, l, m)
            radial_s = zernike_radial_derivative(s, l, m)
            phase = m * theta - n * self.nfp * zeta
            R += coefficient * radial * cos(phase)
            R_s += coefficient * radial_s * cos(phase)
            R_theta -= coefficient * radial * m * sin(phase)
            R_zeta += coefficient * radial * n * self.nfp * sin(phase)

        Z = 0.0
        Z_s = 0.0
        Z_theta = 0.0
        Z_zeta = 0.0
        for (n, m, l), coefficient in self.z_coefficients.items():
            radial = zernike_radial(s, l, m)
            radial_s = zernike_radial_derivative(s, l, m)
            phase = m * theta - n * self.nfp * zeta
            Z += coefficient * radial * sin(phase)
            Z_s += coefficient * radial_s * sin(phase)
            Z_theta += coefficient * radial * m * cos(phase)
            Z_zeta -= coefficient * radial * n * self.nfp * cos(phase)

        return SurfaceEvaluation(
            R=R,
            Z=Z,
            R_s=R_s,
            R_theta=R_theta,
            R_zeta=R_zeta,
            Z_s=Z_s,
            Z_theta=Z_theta,
            Z_zeta=Z_zeta,
        )

    def boundary_coefficients(self) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
        """Sum radial modes to recover Fourier coefficients at ``s=1``.

        Since every supported radial polynomial satisfies ``R_l^m(1)=1``,
        the boundary coefficient for each ``(n,m)`` is the sum of its stored
        radial coefficients.

        Returns
        -------
        tuple of dict
            ``(r_boundary, z_boundary)`` mappings keyed by ``(n,m)``.
        """

        r_boundary: Dict[Tuple[int, int], float] = {}
        z_boundary: Dict[Tuple[int, int], float] = {}
        for (n, m, _l), coefficient in self.r_coefficients.items():
            r_boundary[(n, m)] = r_boundary.get((n, m), 0.0) + coefficient
        for (n, m, _l), coefficient in self.z_coefficients.items():
            z_boundary[(n, m)] = z_boundary.get((n, m), 0.0) + coefficient
        return r_boundary, z_boundary

    def validate(
        self,
        s_values: Iterable[float],
        theta_values: Iterable[float],
        zeta_values: Iterable[float],
        *,
        require_positive_jacobian: bool = True,
    ) -> Dict[str, float]:
        """Check radius and Jacobian signs on a supplied coordinate grid.

        The magnetic axis at ``s=0`` is a coordinate singularity, so positive
        Jacobian is enforced only for non-axis samples.

        Parameters
        ----------
        s_values, theta_values, zeta_values : iterable of float
            Radial coordinates and poloidal/toroidal angles to sample. The
            iterables are traversed directly; pass reusable sequences if they
            must be consumed more than once.
        require_positive_jacobian : bool, default=True
            If true, raise when any non-axis sample has Jacobian ``<= 0``.

        Returns
        -------
        dict of str to float
            Minimum radius, minimum and maximum Jacobian, and minimum
            non-axis Jacobian over the sampled points.

        Raises
        ------
        ValueError
            If coordinates are non-finite, cylindrical radius is non-positive,
            or a requested positive-Jacobian condition fails.
        """

        minimum_radius = float("inf")
        minimum_jacobian = float("inf")
        minimum_nonaxis_jacobian = float("inf")
        maximum_jacobian = float("-inf")
        for s in s_values:
            for theta in theta_values:
                for zeta in zeta_values:
                    evaluation = self.evaluate(s, theta, zeta)
                    if not isfinite(evaluation.R) or not isfinite(evaluation.Z):
                        raise ValueError("surface evaluation is not finite")
                    if evaluation.R <= 0.0:
                        raise ValueError("surface has non-positive cylindrical radius")
                    jacobian = evaluation.jacobian
                    minimum_radius = min(minimum_radius, evaluation.R)
                    minimum_jacobian = min(minimum_jacobian, jacobian)
                    if s > 0.0:
                        minimum_nonaxis_jacobian = min(
                            minimum_nonaxis_jacobian, jacobian
                        )
                    maximum_jacobian = max(maximum_jacobian, jacobian)
                    # The magnetic axis is a coordinate singularity. Its
                    # Jacobian is expected to vanish at s=0; strict
                    # positivity is required only on non-axis surfaces.
                    if require_positive_jacobian and s > 0.0 and jacobian <= 0.0:
                        raise ValueError("surface Jacobian is not strictly positive")
        return {
            "minimum_radius": minimum_radius,
            "minimum_jacobian": minimum_jacobian,
            "minimum_nonaxis_jacobian": minimum_nonaxis_jacobian,
            "maximum_jacobian": maximum_jacobian,
        }
