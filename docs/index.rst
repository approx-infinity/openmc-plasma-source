OpenMC Plasma Source
====================

``openmc-plasma-source`` provides pre-built neutron sources for fusion
applications using OpenMC. The package includes point, ring, tokamak, and
stellarator source models.

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   point_source
   ring_source
   tokamak_source
   stellarator_source
   testing

Common result quantities
------------------------

The tokamak and stellarator constructors return result objects containing an
OpenMC mesh source, plasma volume, and fusion power. Plasma volume is returned
in m :sup:`3`, fusion power is returned in MW, and mesh-source strength is the
total neutron emission rate in neutrons per second.

For the point and ring sources, the constructors return a list containing one
``openmc.IndependentSource`` instance.

Project links
-------------

* `Source repository <https://github.com/fusion-energy/openmc-plasma-source>`_
* `OpenMC documentation <https://docs.openmc.org>`_
* `Issue tracker <https://github.com/fusion-energy/openmc-plasma-source/issues>`_
