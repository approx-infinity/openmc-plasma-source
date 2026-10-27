Installation
============

Requirements
------------

OpenMC is required to use ``openmc-plasma-source``. Install OpenMC according
to the `official OpenMC installation instructions
<https://docs.openmc.org/en/stable/quickinstall.html>`_.

Install from PyPI
------------------

Install the package with pip:

.. code-block:: console

   pip install openmc-plasma-source

The standard stellarator source calculation uses NumPy. JAX and SciPy are only
needed when enabling optional geometry optimization. Install those packages
with:

.. code-block:: console

   pip install "openmc-plasma-source[stellarator-optimization]"

PyVista is only needed by the live 3-D flux-surface plotting example:

.. code-block:: console

   pip install "openmc-plasma-source[stellarator-plotting]"

The combined ``stellarator`` extra remains available to install both sets of
optional dependencies.

Install from a local checkout
-----------------------------

From the repository root:

.. code-block:: console

   pip install .

For editable development installation:

.. code-block:: console

   pip install -e .

Running examples
----------------

The example scripts are in the ``examples/`` directory. For example:

.. code-block:: console

   python examples/point_source_example.py
   python examples/ring_source_example.py
   python examples/tokamak_source_example.py
   python examples/stellarator_source_example.py

The point, ring, and tokamak examples use a local ``cross_sections.xml`` file
for demonstration. Replace it with the cross-section data configured for your
OpenMC installation.
