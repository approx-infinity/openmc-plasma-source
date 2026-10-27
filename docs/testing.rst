Testing
=======

Run the complete test suite from the repository root with:

.. code-block:: console

   python -m pytest

The stellarator tests cover the scalar Fourier--Zernike basis, rotating and
twisted field-period geometry, Jacobian validation, profile and emission
reuse, source sampling, JAX gradients, SciPy optimization, and Helias-5B
scale benchmarks.
