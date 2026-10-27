"""JAX discrete action and coefficient handling for stellarator surfaces.

This module deliberately uses scalar evaluations and explicit Python loops in
the first implementation. The loops have fixed bounds, so JAX can still trace
and differentiate the resulting action. Vectorized evaluation is deferred
until the scalar reference implementation is validated.
"""

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from .basis import valid_modes, zernike_radial, zernike_radial_derivative

ModeKey = Tuple[float, int, int]
BoundaryKey = Tuple[float, int]


@dataclass(frozen=True)
class ActionGrid:
    """Radial nodes and angular cell counts for action quadrature.

    Radial nodes include both endpoints and define ``len(s_nodes)-1`` radial
    cells. Poloidal and toroidal cells are uniform over one full ``2*pi``
    period; radial cells may be nonuniform.
    """

    s_nodes: Tuple[float, ...]
    n_theta: int
    n_zeta: int

    def __post_init__(self) -> None:
        """Validate radial ordering, radial bounds, and angular cell counts.

        Raises ``ValueError`` if fewer than two radial nodes are supplied, if
        nodes are not strictly increasing or leave ``[0, 1]``, or if either
        angular cell count is less than one.
        """

        if len(self.s_nodes) < 2:
            raise ValueError("s_nodes must contain at least two radial nodes")
        if any(self.s_nodes[i] >= self.s_nodes[i + 1] for i in range(len(self.s_nodes) - 1)):
            raise ValueError("s_nodes must be strictly increasing")
        if self.s_nodes[0] < 0.0 or self.s_nodes[-1] > 1.0:
            raise ValueError("s_nodes must lie within [0, 1]")
        if self.n_theta < 1 or self.n_zeta < 1:
            raise ValueError("n_theta and n_zeta must be positive")

    @classmethod
    def uniform(cls, n_s: int, n_theta: int, n_zeta: int) -> "ActionGrid":
        """Create a uniform tensor grid including ``s=0`` and ``s=1``.

        Parameters
        ----------
        n_s : int
            Number of radial nodes; at least two are required.
        n_theta, n_zeta : int
            Numbers of poloidal and toroidal cells.

        Returns
        -------
        ActionGrid
            Radial nodes ``i/(n_s-1)`` with the requested uniform angular
            resolutions.

        Raises
        ------
        ValueError
            If ``n_s < 2``. Invalid angular resolutions are rejected by the
            constructed instance.
        """

        if n_s < 2:
            raise ValueError("n_s must be at least two")
        s_nodes = tuple(index / (n_s - 1) for index in range(n_s))
        return cls(s_nodes=s_nodes, n_theta=n_theta, n_zeta=n_zeta)

    @property
    def n_radial_cells(self) -> int:
        """Number of intervals between consecutive radial nodes."""

        return len(self.s_nodes) - 1


@dataclass(frozen=True)
class CoefficientLayout:
    """Free and boundary-constrained Fourier–Zernike coefficients.

    For each ``(n, m)`` pair, the lowest radial coefficient with ``l=m`` is
    eliminated using the external boundary constraint. All higher valid radial
    coefficients are optimization variables, packed with R entries first and
    Z entries second.
    """

    r_modes: Tuple[ModeKey, ...]
    z_modes: Tuple[ModeKey, ...]
    r_boundary: Mapping[BoundaryKey, float]
    z_boundary: Mapping[BoundaryKey, float]

    @classmethod
    def from_orders(
        cls,
        n_max: int,
        m_max: int,
        l_max: int,
        r_boundary: Mapping[BoundaryKey, float],
        z_boundary: Mapping[BoundaryKey, float],
        *,
        twisted_rotation: int | None = None,
    ) -> "CoefficientLayout":
        """Construct admissible modes and enforce the external boundary.

        Parameters
        ----------
        n_max, m_max, l_max : int
            Maximum toroidal, poloidal, and radial basis orders.
        r_boundary, z_boundary : mapping of (n, m) to float
            Prescribed cosine coefficients for R and sine coefficients for Z
            at ``s=1``.
        twisted_rotation : int, optional
            Restricts toroidal modes to the parity class compatible with a
            poloidal seam rotation.

        Returns
        -------
        CoefficientLayout
            Mode lists and copied boundary maps. The lowest radial order for
            each boundary key is reconstructed from the boundary sum.
        """

        modes = tuple(
            (mode.n, mode.m, mode.l)
            for mode in valid_modes(
                n_max,
                m_max,
                l_max,
                twisted_rotation=twisted_rotation,
            )
        )
        z_modes = tuple(
            mode for mode in modes if not (mode[0] == 0.0 and mode[1] == 0 and mode[2] == 0)
        )
        return cls(
            r_modes=modes,
            z_modes=z_modes,
            r_boundary=dict(r_boundary),
            z_boundary=dict(z_boundary),
        )

    @staticmethod
    def _free_modes(modes: Sequence[ModeKey]) -> Tuple[ModeKey, ...]:
        """Select modes with ``l > m`` whose coefficients are independent."""

        return tuple(mode for mode in modes if mode[2] > mode[1])

    @property
    def r_free_modes(self) -> Tuple[ModeKey, ...]:
        """Free higher-radial-order modes in the R expansion."""

        return self._free_modes(self.r_modes)

    @property
    def z_free_modes(self) -> Tuple[ModeKey, ...]:
        """Free higher-radial-order modes in the Z expansion."""

        return self._free_modes(self.z_modes)

    @property
    def n_free(self) -> int:
        """Total number of scalar optimization variables."""

        return len(self.r_free_modes) + len(self.z_free_modes)

    def _free_indices(self, modes: Sequence[ModeKey], offset: int) -> Dict[ModeKey, int]:
        """Map each free mode to its packed-vector index plus ``offset``."""

        return {
            mode: offset + index
            for index, mode in enumerate(self._free_modes(modes))
        }

    @staticmethod
    def _assemble_component(
        x,
        modes: Sequence[ModeKey],
        boundary: Mapping[BoundaryKey, float],
        offset: int,
    ):
        """Assemble one coordinate's coefficients with boundary sums.

        Higher radial orders read from ``x``. For every ``(n, m)``, the
        coefficient at ``l=m`` is the prescribed boundary value minus the sum
        of free higher-order coefficients sharing that key. The returned
        sequence preserves ``modes`` order.
        """

        free_modes = tuple(mode for mode in modes if mode[2] > mode[1])
        free_indices = {
            mode: offset + index for index, mode in enumerate(free_modes)
        }
        free_by_boundary: Dict[BoundaryKey, Tuple[int, ...]] = {}
        for mode in free_modes:
            key = (mode[0], mode[1])
            indices = list(free_by_boundary.get(key, ()))
            indices.append(free_indices[mode])
            free_by_boundary[key] = tuple(indices)

        coefficients = []
        for mode in modes:
            if mode[2] > mode[1]:
                coefficient = x[free_indices[mode]]
            else:
                key = (mode[0], mode[1])
                coefficient = boundary.get(key, 0.0)
                for index in free_by_boundary.get(key, ()):
                    coefficient = coefficient - x[index]
            coefficients.append((mode, coefficient))
        return tuple(coefficients)

    def coefficients(self, x):
        """Build JAX-compatible R and Z term sequences from a free vector.

        Parameters
        ----------
        x : one-dimensional array-like
            Packed free coefficients of length :attr:`n_free`; R entries
            precede Z entries.

        Returns
        -------
        tuple
            ``(r_terms, z_terms)``. Each sequence contains
            ``((n, m, l), coefficient)`` pairs, including values reconstructed
            from the boundary constraint.

        Raises
        ------
        ValueError
            If the vector length differs from :attr:`n_free`.
        """

        if len(x) != self.n_free:
            raise ValueError(f"expected {self.n_free} free coefficients, got {len(x)}")
        r_terms = self._assemble_component(
            x, self.r_modes, self.r_boundary, offset=0
        )
        z_terms = self._assemble_component(
            x,
            self.z_modes,
            self.z_boundary,
            offset=len(self.r_free_modes),
        )
        return r_terms, z_terms

    def initial_vector(
        self,
        r_coefficients: Mapping[ModeKey, float],
        z_coefficients: Mapping[ModeKey, float],
    ) -> Tuple[float, ...]:
        """Pack initial higher-order coefficients in layout order.

        Missing free-mode keys default to zero. Boundary-order coefficients
        are ignored because they are reconstructed from the boundary sums.

        Parameters
        ----------
        r_coefficients, z_coefficients : mapping of ModeKey to float
            Initial R and Z coefficients keyed by ``(n, m, l)``.

        Returns
        -------
        tuple of float
            R free coefficients followed by Z free coefficients.
        """

        values = []
        for mode in self.r_free_modes:
            values.append(float(r_coefficients.get(mode, 0.0)))
        for mode in self.z_free_modes:
            values.append(float(z_coefficients.get(mode, 0.0)))
        return tuple(values)

    def coefficients_from_vector(
        self,
        x: Sequence[float],
    ) -> Tuple[Dict[ModeKey, float], Dict[ModeKey, float]]:
        """Convert a free vector to complete Python coefficient dictionaries.

        This returns ordinary dictionaries for logging, surface construction,
        and diagnostics; :meth:`coefficients` instead returns ordered term
        sequences suitable for JAX tracing.

        Parameters
        ----------
        x : sequence of float
            Packed free vector of length :attr:`n_free`.

        Returns
        -------
        tuple of dict
            ``(r_coefficients, z_coefficients)`` including reconstructed
            boundary-order values.

        Raises
        ------
        ValueError
            If the vector length differs from :attr:`n_free`.
        """

        if len(x) != self.n_free:
            raise ValueError(f"expected {self.n_free} free coefficients, got {len(x)}")
        values = tuple(float(value) for value in x)
        r_free = {
            mode: values[index] for index, mode in enumerate(self.r_free_modes)
        }
        z_offset = len(self.r_free_modes)
        z_free = {
            mode: values[z_offset + index]
            for index, mode in enumerate(self.z_free_modes)
        }
        r_result: Dict[ModeKey, float] = dict(r_free)
        z_result: Dict[ModeKey, float] = dict(z_free)
        for modes, boundary, result in (
            (self.r_modes, self.r_boundary, r_result),
            (self.z_modes, self.z_boundary, z_result),
        ):
            for mode in modes:
                if mode[2] != mode[1]:
                    continue
                key = (mode[0], mode[1])
                result[mode] = boundary.get(key, 0.0)
                for free_mode, value in result.items():
                    if free_mode[0] == mode[0] and free_mode[1] == mode[1] and free_mode[2] > mode[1]:
                        result[mode] -= value
        return r_result, z_result


def _zernike_radial_jax(s, l: int, m: int):
    """Evaluate the Zernike radial polynomial using JAX scalar operations.

    The finite polynomial is expanded by a Python loop during tracing, so JAX
    can differentiate the result with respect to ``s`` and coefficient inputs.
    Radial-index validity is expected to have been checked by the caller.
    """

    import jax.numpy as jnp
    from math import factorial

    value = jnp.asarray(0.0)
    for k in range((l - m) // 2 + 1):
        numerator = (-1) ** k * factorial(l - k)
        denominator = (
            factorial(k)
            * factorial((l + m) // 2 - k)
            * factorial((l - m) // 2 - k)
        )
        value = value + numerator / denominator * s ** (l - 2 * k)
    return value


def _zernike_radial_derivative_jax(s, l: int, m: int):
    """Evaluate d R_l^m / ds using JAX-compatible scalar arithmetic.

    Terms are differentiated explicitly, with constant terms omitted. This is
    used by the geometry penalty when forming the Jacobian.
    """

    import jax.numpy as jnp
    from math import factorial

    value = jnp.asarray(0.0)
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
        value = value + numerator / denominator * exponent * s ** (exponent - 1)
    return value


def _evaluate_point(r_terms, z_terms, nfp: int, s, theta, zeta):
    """Evaluate scalar cylindrical position from preassembled Fourier terms.

    R terms use cosine and Z terms use sine of ``m*theta - n*nfp*zeta``. The
    arguments and return values are JAX-compatible scalars. This helper does
    not validate the surface or its coefficients.
    """

    import jax.numpy as jnp

    R = jnp.asarray(0.0)
    Z = jnp.asarray(0.0)
    for (n, m, l), coefficient in r_terms:
        phase = m * theta - n * nfp * zeta
        R = R + coefficient * _zernike_radial_jax(s, l, m) * jnp.cos(phase)
    for (n, m, l), coefficient in z_terms:
        phase = m * theta - n * nfp * zeta
        Z = Z + coefficient * _zernike_radial_jax(s, l, m) * jnp.sin(phase)
    return R, Z


def _cartesian_point(r_terms, z_terms, nfp: int, s, theta, zeta):
    """Convert a mapped point to Cartesian XYZ using JAX scalar operations.

    Used to measure the action's radial edge length between adjacent ``s``
    values at fixed angles.
    """

    import jax.numpy as jnp

    R, Z = _evaluate_point(r_terms, z_terms, nfp, s, theta, zeta)
    return R * jnp.cos(zeta), R * jnp.sin(zeta), Z


def make_action(
    layout: CoefficientLayout,
    nfp: int,
    grid: ActionGrid,
    omega: float,
    *,
    radial_length_epsilon: float = 0.0,
):
    """Build a callable for the cell-quadrature stellarator action.

    ``A`` is the signed poloidal quadrilateral area in the ``(R, Z)`` plane,
    and ``L`` is the three-dimensional radial edge length. They approximate
    ``J_p * ds * dtheta`` and ``|e_s| * ds``, respectively, so the cell action
    includes the coordinate-space quadrature measures. No absolute value is
    applied to the area or Jacobian; invalid orientation must be detected by
    validation rather than hidden by an absolute value.

    Parameters
    ----------
    layout : CoefficientLayout
        Basis modes, boundary constraints, and free-variable ordering.
    nfp : int
        Number of field periods; phases use ``n*nfp*zeta``.
    grid : ActionGrid
        Radial nodes and angular cell counts for the quadrature.
    omega : float
        Non-negative weight of radial line length in the action.
    radial_length_epsilon : float, default=0
        Non-negative regularizer added under the square root of each radial
        edge length. Use a positive value only to smooth a zero-length edge.

    Returns
    -------
    callable
        Function ``action(x)`` returning a scalar JAX value for the packed free
        coefficient vector ``x``. It can be passed to
        :func:`jax_value_and_gradient`.

    Raises
    ------
    ValueError
        If ``nfp`` is not a positive integer, ``omega`` is negative, or the
        length regularizer is negative.
    """

    if not isinstance(nfp, int) or isinstance(nfp, bool) or nfp <= 0:
        raise ValueError("nfp must be a positive integer")
    if omega < 0.0:
        raise ValueError("omega must be non-negative")
    if radial_length_epsilon < 0.0:
        raise ValueError("radial_length_epsilon must be non-negative")

    def action(x):
        """Evaluate the discrete action at one packed coefficient vector."""

        import jax.numpy as jnp

        r_terms, z_terms = layout.coefficients(x)
        total = jnp.asarray(0.0)
        dtheta = 2.0 * pi / grid.n_theta
        dzeta = 2.0 * pi / grid.n_zeta

        for i in range(grid.n_radial_cells):
            s0 = grid.s_nodes[i]
            s1 = grid.s_nodes[i + 1]
            ds = s1 - s0
            s_mid = 0.5 * (s0 + s1)
            for j in range(grid.n_theta):
                theta0 = j * dtheta
                theta1 = (j + 1) * dtheta
                for k in range(grid.n_zeta):
                    zeta = k * dzeta
                    p00 = _evaluate_point(r_terms, z_terms, nfp, s0, theta0, zeta)
                    p10 = _evaluate_point(r_terms, z_terms, nfp, s1, theta0, zeta)
                    p11 = _evaluate_point(r_terms, z_terms, nfp, s1, theta1, zeta)
                    p01 = _evaluate_point(r_terms, z_terms, nfp, s0, theta1, zeta)

                    area = 0.5 * (
                        p00[0] * p10[1]
                        + p10[0] * p11[1]
                        + p11[0] * p01[1]
                        + p01[0] * p00[1]
                        - p10[0] * p00[1]
                        - p11[0] * p10[1]
                        - p01[0] * p11[1]
                        - p00[0] * p01[1]
                    )

                    x0, y0, z0 = _cartesian_point(
                        r_terms, z_terms, nfp, s0, theta0, zeta
                    )
                    x1, y1, z1 = _cartesian_point(
                        r_terms, z_terms, nfp, s1, theta0, zeta
                    )
                    radial_length = jnp.sqrt(
                        (x1 - x0) ** 2
                        + (y1 - y0) ** 2
                        + (z1 - z0) ** 2
                        + radial_length_epsilon**2
                    )
                    # A approximates J_p * ds * dtheta and radial_length
                    # approximates |e_s| * ds. Include the coordinate-cell
                    # measures when discretizing the continuum action.
                    cell_action = (
                        0.5 * area**2 / (s_mid * ds * dtheta)
                        + omega * radial_length * dtheta
                    )
                    total = total + cell_action * dzeta

        return total

    return action


def jax_value_and_gradient(action, *, jit: bool = True):
    """Differentiate an action with JAX and optionally compile it.

    Parameters
    ----------
    action : callable
        Scalar-valued function of a one-dimensional coefficient array.
    jit : bool, default=True
        If true, JIT-compile the value-and-gradient function. Compilation may
        take appreciable time for large statically unrolled grids.

    Returns
    -------
    callable
        Function mapping a coefficient vector to ``(value, gradient)``.
        JAX 64-bit mode is enabled before constructing the derivative.
    """

    import jax

    jax.config.update("jax_enable_x64", True)
    value_and_gradient = jax.value_and_grad(action)
    if jit:
        return jax.jit(value_and_gradient)
    return value_and_gradient


def scalar_action_reference(
    r_coefficients: Mapping[ModeKey, float],
    z_coefficients: Mapping[ModeKey, float],
    nfp: int,
    grid: ActionGrid,
    omega: float,
) -> float:
    """Evaluate the same cell action as :func:`make_action` in Python floats.

    This implementation avoids JAX and is intended as an independent numerical
    reference for small grids. Coefficient maps must already include both the
    free and boundary-reconstructed modes.

    Parameters
    ----------
    r_coefficients, z_coefficients : mapping of ModeKey to float
        Complete Fourier–Zernike coefficients for R and Z.
    nfp : int
        Number of field periods.
    grid : ActionGrid
        Quadrature grid used by the action.
    omega : float
        Non-negative radial-length weight.

    Returns
    -------
    float
        Scalar action value.
    """

    total = 0.0
    dtheta = 2.0 * pi / grid.n_theta
    dzeta = 2.0 * pi / grid.n_zeta

    def point(s: float, theta: float, zeta: float) -> Tuple[float, float]:
        """Evaluate the Python-float R,Z mapping at one quadrature vertex."""

        R = 0.0
        Z = 0.0
        for (n, m, l), coefficient in r_coefficients.items():
            R += coefficient * zernike_radial(s, l, m) * cos(m * theta - n * nfp * zeta)
        for (n, m, l), coefficient in z_coefficients.items():
            Z += coefficient * zernike_radial(s, l, m) * sin(m * theta - n * nfp * zeta)
        return R, Z

    for i in range(grid.n_radial_cells):
        s0 = grid.s_nodes[i]
        s1 = grid.s_nodes[i + 1]
        ds = s1 - s0
        s_mid = 0.5 * (s0 + s1)
        for j in range(grid.n_theta):
            theta0 = j * dtheta
            theta1 = (j + 1) * dtheta
            for k in range(grid.n_zeta):
                zeta = k * dzeta
                p00 = point(s0, theta0, zeta)
                p10 = point(s1, theta0, zeta)
                p11 = point(s1, theta1, zeta)
                p01 = point(s0, theta1, zeta)
                area = 0.5 * (
                    p00[0] * p10[1]
                    + p10[0] * p11[1]
                    + p11[0] * p01[1]
                    + p01[0] * p00[1]
                    - p10[0] * p00[1]
                    - p11[0] * p10[1]
                    - p01[0] * p11[1]
                    - p00[0] * p01[1]
                )
                x0 = p00[0] * cos(zeta)
                y0 = p00[0] * sin(zeta)
                x1 = p10[0] * cos(zeta)
                y1 = p10[0] * sin(zeta)
                radial_length = sqrt(
                    (x1 - x0) ** 2
                    + (y1 - y0) ** 2
                    + (p10[1] - p00[1]) ** 2
                )
                total += (
                    0.5 * area**2 / (s_mid * ds * dtheta)
                    + omega * radial_length * dtheta
                ) * dzeta
    return total
