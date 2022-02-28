"""
=========================================
Mean-field Hubbard model (:mod:`hubbard`)
=========================================

.. module:: hubbard
    :noindex:

Package for mean-field Hubbard approximation

Self Consistent field class
===========================

.. autosummary::
   :toctree:

    HubbardHamiltonian
    NEGF
    calc_n

Read and write in binary files
==============================

.. autosummary::
   :toctree:

   ncSileHubbard

Build the specific TB Hamiltonian for a sp2 system
==================================================

.. autosummary::
   :toctree:

   sp2

Create real space grid for a geometry and supercell
===================================================

.. autosummary::
   :toctree:

   real_space_grid

"""

# Add version information
from . import _version
__version__ = _version.version
__version_tuple__ = _version.version_tuple

from .hamiltonian import *
from . import plot
from .sp2 import *
from .ncsile import *
from .density import *
from .negf import *
from .grid import *
