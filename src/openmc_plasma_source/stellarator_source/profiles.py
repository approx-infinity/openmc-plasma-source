"""Shared tokamak-compatible density and temperature profiles."""

from typing import Union

import numpy as np
from numpy.typing import NDArray


def ion_density_profile(
    mode: str,
    ion_density_centre: float,
    ion_density_peaking_factor: float,
    ion_density_pedestal: float,
    minor_radius: float,
    pedestal_radius: float,
    ion_density_separatrix: float,
    r: Union[float, NDArray],
) -> NDArray:
    """Evaluate the L-mode or pedestal-shaped ion density profile.

    L mode uses ``n0 * (1 - (r/a)**2)**alpha``. H and A modes use the same
    core-plus-linear-pedestal formula: the core runs from the central density
    to the pedestal value at ``pedestal_radius``, then density varies linearly
    to the separatrix value at ``minor_radius``.

    Parameters
    ----------
    mode : str
        Profile selector: ``"L"``, ``"H"``, or ``"A"``. H and A currently
        share the same implementation.
    ion_density_centre, ion_density_pedestal, ion_density_separatrix : float
        Density levels in m^-3.
    ion_density_peaking_factor : float
        Exponent controlling the core-profile curvature.
    minor_radius, pedestal_radius : float
        Plasma minor radius and pedestal location in centimetres.
    r : float or array-like
        Minor-radius sample locations in centimetres.

    Returns
    -------
    numpy.ndarray
        Density values in m^-3, with the shape of ``r`` (a scalar input
        produces a zero-dimensional array).

    Raises
    ------
    ValueError
        If any requested radius is negative or ``mode`` is unsupported.
    """

    r = np.asarray(r)
    if np.any(r < 0):
        raise ValueError("Minor radius must not be negative")

    if mode == "L":
        density = (
            ion_density_centre
            * (1 - (r / minor_radius) ** 2) ** ion_density_peaking_factor
        )
    elif mode in ["H", "A"]:
        r_core = np.minimum(r, pedestal_radius)
        density = np.where(
            r < pedestal_radius,
            (
                (ion_density_centre - ion_density_pedestal)
                * (1 - (r_core / pedestal_radius) ** 2) ** ion_density_peaking_factor
                + ion_density_pedestal
            ),
            (
                (ion_density_pedestal - ion_density_separatrix)
                * (minor_radius - r)
                / (minor_radius - pedestal_radius)
                + ion_density_separatrix
            ),
        )
    else:
        raise ValueError(f'Mode {mode} not in available options ["L", "H", "A"]')
    return density


def ion_temperature_profile(
    r: Union[float, NDArray],
    mode: str,
    pedestal_radius: float,
    ion_temperature_pedestal: float,
    ion_temperature_centre: float,
    ion_temperature_beta: float,
    ion_temperature_peaking_factor: float,
    ion_temperature_separatrix: float,
    minor_radius: float,
) -> NDArray:
    """Evaluate the existing tokamak ion-temperature profile.

    L mode uses ``T0 * (1 - (r/a)**2)**alpha``. H and A modes use a shaped core
    profile up to the pedestal, followed by a linear interpolation to the
    separatrix. The temperature-level arguments accepted by this low-level
    helper are in keV, matching the historical internal tokamak helper; the
    result is converted to eV. Public source APIs convert their eV inputs to
    keV before calling this function.

    Parameters
    ----------
    r : float or array-like
        Minor-radius sample locations in centimetres.
    mode : str
        Profile selector: ``"L"``, ``"H"``, or ``"A"``. H and A currently
        share the same implementation.
    pedestal_radius, minor_radius : float
        Pedestal location and plasma minor radius in centimetres.
    ion_temperature_pedestal, ion_temperature_centre,
    ion_temperature_separatrix : float
        Temperatures in keV.
    ion_temperature_beta, ion_temperature_peaking_factor : float
        Exponents controlling the shaped core profile.

    Returns
    -------
    numpy.ndarray
        Ion temperature in eV, with the shape of ``r``.

    Raises
    ------
    ValueError
        If any requested radius is negative or ``mode`` is unsupported.
    """

    r = np.asarray(r)
    if np.any(r < 0):
        raise ValueError("Minor radius must not be negative")

    if mode == "L":
        temperature = (
            ion_temperature_centre
            * (1 - (r / minor_radius) ** 2) ** ion_temperature_peaking_factor
        )
    elif mode in ["H", "A"]:
        r_core = np.minimum(r, pedestal_radius)
        temperature = np.where(
            r < pedestal_radius,
            (
                ion_temperature_pedestal
                + (ion_temperature_centre - ion_temperature_pedestal)
                * (1 - (r_core / pedestal_radius) ** ion_temperature_beta)
                ** ion_temperature_peaking_factor
            ),
            (
                ion_temperature_separatrix
                + (ion_temperature_pedestal - ion_temperature_separatrix)
                * (minor_radius - r)
                / (minor_radius - pedestal_radius)
            ),
        )
    else:
        raise ValueError(f'Mode {mode} not in available options ["L", "H", "A"]')
    return temperature * 1e3
