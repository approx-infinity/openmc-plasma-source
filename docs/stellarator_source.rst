Stellarator source
==================

The stellarator source models a three-dimensional plasma using a scalar
Fourier--Zernike flux-surface representation. Existing profile, reaction,
Ballabio energy-broadening, and volume-weighting physics are shared with the
tokamak source. The new part is the three-dimensional geometry and scalar
spatial quadrature.

The current convenience geometry is a rotating elliptical boundary. A real
equilibrium can be represented by supplying or optimizing higher-order
Fourier--Zernike coefficients.

API example
-----------

.. code-block:: python

   from openmc_plasma_source import stellarator_source

   result = stellarator_source(
       major_radius=550.0,       # cm
       minor_radius=50.0,         # cm
       nfp=4,
       elongation=1.6,
       rotation=1.0,              # visible poloidal rotations per field period
       asymmetry=0.05,
       mode="H",
       ion_density_centre=4.8e20,
       ion_density_peaking_factor=1.0,
       ion_density_pedestal=4.8e20,
       ion_density_separatrix=3.0e19,
       ion_temperature_centre=12.0e3,      # eV
       ion_temperature_peaking_factor=2.0,
       ion_temperature_beta=6.0,
       ion_temperature_pedestal=1.0e3,     # eV
       ion_temperature_separatrix=0.1e3,   # eV
       pedestal_radius=40.0,               # cm
       fuel={"D": 0.5, "T": 0.5},
       sample_size=(64, 128, 128),
       mesh_resolution=(100, 128, 100),
   )

   source = result.source
   print(result.plasma_volume)  # m^3
   print(result.fusion_power)    # MW

Units and resolution
--------------------

Geometry inputs use cm. Temperatures use eV, densities use m :sup:`-3`, and
``pedestal_radius`` uses cm. The returned plasma volume is in m :sup:`3` and
fusion power is in MW.

``sample_size`` is the scalar midpoint quadrature resolution
``(n_s, n_theta, n_zeta)``. ``mesh_resolution`` is the OpenMC cylindrical
mesh resolution ``(n_R, n_phi, n_Z)``.

Poloidal rotation convention
----------------------------

``rotation`` is an integer number of visible poloidal ellipse rotations per
field period. The ellipse angle is

.. math::

   \delta(\zeta) = \frac{q N_{\mathrm{fp}} \zeta}{2},

where ``q = rotation``. Thus ``rotation=1`` advances the ellipse orientation
by :math:`\pi` over one field period. Because an ellipse is invariant under a
:math:`\pi` rotation, this is one visible poloidal rotation per field period.

For odd ``rotation``, the m=1 terms use half-integer toroidal frequencies. The
implementation treats the field-period seam as a twisted Fourier--Zernike
coordinate seam, with the corresponding poloidal shift. This keeps the
physical surface field-periodic without incorrectly displaying two poloidal
rotations per field period.

Asymmetry
---------

``asymmetry`` adds a regular m=2, n=rotation perturbation. It removes the
ellipse's :math:`\pi`-rotation degeneracy while retaining field-period closure
as a physical surface and preserving stellarator symmetry.

Geometry optimization
---------------------

The default geometry uses the analytical Zernike--Fourier interior extension.
Higher-order interior coefficients can be optimized with the JAX/SciPy action
workflow:

.. code-block:: python

   result = stellarator_source(
       # ... geometry and profile parameters ...
       basis_order=(2, 2, 4),
       optimize_geometry=True,
       optimization_grid=(4, 16, 8),
   )

Helias-5B scale benchmark
-------------------------

The supplied scale case uses:

* major radius: 22 m;
* minor radius: 1.8 m;
* field periods: 5.

This is a dimensional scale benchmark, not a physical Helias-5B equilibrium.
A faithful equilibrium requires the actual non-axisymmetric boundary
coefficients and associated shape data.

.. code-block:: python

   from openmc_plasma_source.stellarator_source import (
       helias5b_convergence_study,
   )

   records = helias5b_convergence_study(
       sample_sizes=((16, 32, 32), (32, 64, 64), (64, 128, 128)),
   )

Live flux-surface plotting
--------------------------

With the stellarator optional dependencies installed, run:

.. code-block:: console

   python examples/plot_stellarator_source.py

This opens a live PyVista view of one continuous flux surface. It does not
plot the OpenMC source mesh.

For 2-D projections without PyVista, run:

.. code-block:: console

   python examples/plot_stellarator_source_2D.py

This displays one complete poloidal X-Z cross-section at the configured
toroidal angle and the flux-surface intersection with Z=0 in the X-Y plane,
using Matplotlib.

To compare the analytical surface with the JAX/SciPy optimized surface at a
fixed toroidal angle, run:

.. code-block:: console

   python examples/plot_stellarator_surface_optimization.py

The optimization example plots the complete poloidal X-Z cross-section before
and after optimization.

Complete example
----------------

.. literalinclude:: ../examples/stellarator_source_example.py
   :language: python
