Tokamak source
==============

The tokamak source models a plasma with radial density and temperature
profiles in an elongated, triangular, Shafranov-shifted tokamak cross-section.
It maps the plasma onto an OpenMC ``CylindricalMesh``. Each mesh voxel receives
its own neutron source strength and local neutron energy distribution.

The profile model follows Fausser et al., *Tokamak D-T neutron source models
for different plasma physics confinement modes*, Fusion Engineering and
Design, `doi:10.1016/j.fusengdes.2012.02.025
<https://doi.org/10.1016/j.fusengdes.2012.02.025>`_.

API example
-----------

.. code-block:: python

   from openmc_plasma_source import tokamak_source

   result = tokamak_source(
       major_radius=906.0,
       minor_radius=292.258,
       elongation=1.557,
       triangularity=0.270,
       shafranov_factor=0.44789,
       mode="H",
       ion_density_centre=1.09e20,
       ion_density_peaking_factor=1.0,
       ion_density_pedestal=1.09e20,
       ion_density_separatrix=3.0e19,
       ion_temperature_centre=45.9e3,
       ion_temperature_peaking_factor=8.06,
       ion_temperature_beta=6.0,
       ion_temperature_pedestal=6.09e3,
       ion_temperature_separatrix=0.1e3,
       pedestal_radius=0.8 * 292.258,
       fuel={"D": 0.5, "T": 0.5},
   )

   source = result.source
   print(result.plasma_volume)  # m^3
   print(result.fusion_power)    # MW

Geometry and units
------------------

``major_radius`` and ``minor_radius`` are in cm. Temperatures are in eV,
densities are in m :sup:`-3`, and ``pedestal_radius`` and
``shafranov_factor`` are in cm.

The optional ``start_angle`` and ``rotation_angle`` parameters select a
toroidal sector in radians. ``rotation_angle`` can be negative and may cross
the ``0``/``2*pi`` seam. The default is a full torus.

The optional ``mesh_resolution`` is the number of bins in the ``(r, z)``
cross-section. ``grid_density`` controls the internal forward-mapping grid.
The axisymmetric toroidal direction is handled internally.

Returned quantities
-------------------

The returned ``TokamakSource`` contains:

``source``
   An ``openmc.MeshSource`` backed by a cylindrical mesh.

``plasma_volume``
   Plasma volume in m :sup:`3`.

``fusion_power``
   Total fusion power in MW.

The mesh-source strength is the total neutron emission rate in neutrons per
second and can be used to normalize OpenMC tallies.

Toroidal-sector example
-----------------------

.. code-block:: python

   result = tokamak_source(
       # ... plasma parameters ...
       start_angle=0.0,
       rotation_angle=2.0 * 3.141592653589793,
       mesh_resolution=(100, 100),
       grid_density=500,
   )

Complete example
----------------

.. literalinclude:: ../examples/tokamak_source_example.py
   :language: python
