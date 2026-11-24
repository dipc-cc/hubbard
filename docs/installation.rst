.. _installation:

Installation
============

Required dependencies
---------------------

   * Python_ 3.10 or newer

   * NumPy_ 2.0 or newer

   * SciPy_ 1.14 or newer

   * sisl_ 0.15 or newer; sisl 0.15 needs SciPy older than 1.17, which
     removed the ``scipy.special.sph_harm`` that sisl 0.15 imports, so
     SciPy 1.17 or newer needs a newer sisl

   * Matplotlib_ 3.9 or newer (plotting)

Optional:

   * netCDF4_ 1.7 or newer (reading and writing ``HU.nc`` files)

.. _Python: https://www.python.org/
.. _NumPy: https://docs.scipy.org/doc/numpy/reference/
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/
.. _sisl : https://zerothi.github.io/sisl
.. _Matplotlib: https://matplotlib.org/
.. _netCDF4: https://github.com/Unidata/netcdf4-python


Installation from source
------------------------

Simply download as zip or clone the project from the `git repository <https://github.com/dipc-cc/hubbard>`_.

Manual installation is performed with the command

.. code-block:: bash

   python3 -m pip install . --prefix=<prefix>

One may also wish to set the following environment variables

.. code-block:: bash

   export PYTHONPATH=<my-python-home>/lib/python/
   export PATH=$PATH:<my-python-home>/bin/
