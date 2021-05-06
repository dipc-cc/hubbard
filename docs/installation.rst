.. _installation:

Installation
============

Required dependencies
---------------------

* Python_ 2.7, 3.4 or newer
* NumPy_ 1.10 or newer
* SciPy_ 0.17 or newer
* sisl_ 0.11.0 or newer

Optional:

* Matplotlib_ 2.0.0 or newer (plotting)

.. _Python: https://www.python.org/
.. _NumPy: https://docs.scipy.org/doc/numpy/reference/
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/
.. _sisl : https://sisl.readthedocs.io/en/latest/installation.html
.. _Matplotlib: https://matplotlib.org/


Installation from source
------------------------

Simply download as zip or clone the project from the `git repository <https://github.com/dipc-cc/hubbard>`_.

Manual installation is performed with the command

.. code-block:: bash

   python setup.py install --prefix=<prefix>

for Python 2, and

.. code-block:: bash

   python3 setup.py install --prefix=<prefix>

for Python 3.

One can also use the auxiliary files `install-py2.sh` and `install-py3.sh`

One may also wish to set the following environment variables

.. code-block:: bash

   export PYTHONPATH=<my-python-home>/lib/python/
   export PATH=$PATH:<my-python-home>/bin/
