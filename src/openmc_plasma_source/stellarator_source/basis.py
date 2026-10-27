"""Scalar Zernike basis functions used by the stellarator geometry layer."""

from dataclasses import dataclass
from math import factorial
from typing import Iterable, Iterator


@dataclass(frozen=True)
class ZernikeMode:
    """Indices identifying one Fourier–Zernike basis mode.

    The angular phase is ``m*theta - n*nfp*zeta``. ``n`` is the toroidal
    Fourier index and may be half-integer for a twisted field period; ``m`` is
    the non-negative poloidal index; ``l`` is the radial polynomial order.
    Valid radial orders satisfy ``l >= m`` and even ``l - m``.
    """

    n: float
    m: int
    l: int


def _validate_radial_indices(l: int, m: int) -> None:
    """Raise if ``l`` and ``m`` do not define an admissible radial mode.

    Booleans are rejected even though Python considers them integers. Raises
    ``TypeError`` for non-integer indices and ``ValueError`` when ``m`` is
    negative, ``l < m``, or ``l - m`` is odd.
    """

    if not isinstance(l, int) or isinstance(l, bool):
        raise TypeError("l must be an integer")
    if not isinstance(m, int) or isinstance(m, bool):
        raise TypeError("m must be an integer")
    if m < 0:
        raise ValueError("m must be non-negative")
    if l < m:
        raise ValueError("l must be greater than or equal to m")
    if (l - m) % 2:
        raise ValueError("l-m must be even for a Zernike radial polynomial")


def zernike_radial(s: float, l: int, m: int) -> float:
    r"""Evaluate the Zernike radial polynomial for one radial coordinate.

    Uses the finite factorial expansion of the unnormalised radial polynomial
    R_l^m(s). The caller supplies a scalar ``s``; this routine does not impose
    a radial-domain limit, although stellarator coordinates normally use
    ``0 <= s <= 1``.

    Parameters
    ----------
    s : float
        Radial coordinate at which to evaluate the polynomial.
    l, m : int
        Radial and poloidal orders. They must satisfy ``l >= m >= 0`` and
        ``l - m`` even.

    Returns
    -------
    float
        The value of R_l^m(s).

    Raises
    ------
    TypeError, ValueError
        If ``l`` or ``m`` is not an admissible radial mode.
    """

    _validate_radial_indices(l, m)
    value = 0.0
    for k in range((l - m) // 2 + 1):
        numerator = (-1) ** k * factorial(l - k)
        denominator = (
            factorial(k)
            * factorial((l + m) // 2 - k)
            * factorial((l - m) // 2 - k)
        )
        value += numerator / denominator * s ** (l - 2 * k)
    return float(value)


def zernike_radial_derivative(s: float, l: int, m: int) -> float:
    r"""Evaluate the derivative of the radial polynomial with respect to s.

    Differentiates the finite factorial expansion term by term; terms with
    zero exponent contribute zero.

    Parameters
    ----------
    s : float
        Radial coordinate at which to evaluate the derivative.
    l, m : int
        Radial and poloidal orders satisfying ``l >= m >= 0`` and even
        ``l - m``.

    Returns
    -------
    float
        The value of d R_l^m(s) / ds.

    Raises
    ------
    TypeError, ValueError
        If ``l`` or ``m`` is not an admissible radial mode.
    """

    _validate_radial_indices(l, m)
    value = 0.0
    for k in range((l - m) // 2 + 1):
        exponent = l - 2 * k
        if exponent == 0:
            continue
        numerator = (-1) ** k * factorial(l - k)
        denominator = (
            factorial(k)
            * factorial((l + m) // 2 - k)
            * factorial((l - m) // 2 - k)
        )
        value += numerator / denominator * exponent * s ** (exponent - 1)
    return float(value)


def valid_modes(
    n_max: int,
    m_max: int,
    l_max: int,
    *,
    n_values: Iterable[float] | None = None,
    twisted_rotation: int | None = None,
) -> Iterator[ZernikeMode]:
    """Yield every admissible mode up to the requested basis orders.

    ``twisted_rotation`` enables the half-integer toroidal grid and retains
    only modes satisfying ``n - m*twisted_rotation/2 in Z``. This is the
    Fourier representation of a poloidal-coordinate seam twist. Modes are
    yielded in increasing ``m``, then ``n``, then ``l`` order.

    Parameters
    ----------
    n_max, m_max, l_max : int
        Non-negative maximum toroidal, poloidal, and radial orders. The
        toroidal range is symmetric about zero.
    n_values : iterable of float, optional
        Explicit toroidal indices. If omitted, integer indices are generated,
        or half-integer indices when ``twisted_rotation`` is set.
    twisted_rotation : int, optional
        Rotation index defining the toroidal half-grid and parity condition.

    Yields
    ------
    ZernikeMode
        A mode satisfying ``l >= m``, even ``l - m``, and the optional twisted
        seam condition.

    Raises
    ------
    ValueError
        If any maximum basis order is negative.
    """

    if n_max < 0 or m_max < 0 or l_max < 0:
        raise ValueError("basis orders must be non-negative")

    if n_values is None:
        if twisted_rotation is None:
            candidate_n_values = tuple(
                float(n) for n in range(-n_max, n_max + 1)
            )
        else:
            candidate_n_values = tuple(
                0.5 * n for n in range(-2 * n_max, 2 * n_max + 1)
            )
    else:
        candidate_n_values = tuple(float(n) for n in n_values)

    for m in range(m_max + 1):
        for n in candidate_n_values:
            if twisted_rotation is not None:
                half_grid_index = round(2.0 * n)
                if abs(2.0 * n - half_grid_index) > 1.0e-12:
                    continue
                if (half_grid_index - m * twisted_rotation) % 2:
                    continue
            for l in range(m, l_max + 1):
                if (l - m) % 2 == 0:
                    yield ZernikeMode(n=n, m=m, l=l)
