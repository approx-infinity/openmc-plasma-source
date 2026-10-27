Point source
============

The point source represents a localized fusion neutron source. It is useful
for sealed-tube neutron generators, inertial-confinement source models, and
other cases where the emission location is approximated by a single point.

API
---

.. code-block:: python

   from openmc_plasma_source import fusion_point_source

   source = fusion_point_source(
       coordinate=(0.0, 0.0, 0.0),
       temperature=20000.0,
       fuel={"D": 0.5, "T": 0.5},
   )

Parameters
----------

``coordinate``
   Three-dimensional source location in metres.

``temperature``
   Plasma ion temperature in eV. The default is 20000 eV.

``fuel``
   Mapping of fuel isotopes to atom fractions. The default is an equal D-T
   mixture.

The function returns a list containing one ``openmc.IndependentSource``. The
source uses an isotropic angular distribution and a fuel-dependent neutron
energy distribution with Ballabio broadening.

Complete example
----------------

.. literalinclude:: ../examples/point_source_example.py
   :language: python
