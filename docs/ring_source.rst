Ring source
===========

The ring source represents a uniform circular source at a fixed major radius
and axial position. It is useful when detailed plasma profiles are unavailable
but a toroidal source approximation is sufficient.

API
---

.. code-block:: python

   import math
   from openmc_plasma_source import fusion_ring_source

   source = fusion_ring_source(
       radius=700.0,
       start_angle=0.0,
       rotation_angle=2.0 * math.pi,
       z_placement=0.0,
       temperature=20000.0,
       fuel={"D": 0.5, "T": 0.5},
   )

Parameters
----------

``radius``
   Ring radius in metres.

``start_angle``
   Toroidal starting angle in radians. It must lie between ``-2*pi`` and
   ``2*pi``.

``rotation_angle``
   Toroidal angular extent in radians. Negative values extend in the opposite
   direction from ``start_angle``. It must lie between ``-2*pi`` and ``2*pi``.

``z_placement``
   Fixed axial location in metres.

``temperature``
   Plasma ion temperature in eV.

``fuel``
   Mapping of fuel isotopes to atom fractions.

The function returns a list containing one ``openmc.IndependentSource``. Its
spatial distribution is cylindrical-independent, its angular distribution is
isotropic, and its energy distribution is determined by the fuel mixture and
temperature.

Complete example
----------------

.. literalinclude:: ../examples/ring_source_example.py
   :language: python
