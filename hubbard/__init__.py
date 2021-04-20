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
    occ
    NEGF

Read and write in binary files
==============================

.. autosummary::
   :toctree:

   ncSilehubbard

Build the specific TB Hamiltonian for a sp2 system
==================================================

.. autosummary::
   :toctree:

   sp2


"""

from .hamiltonian import *
from . import plot
from .sp2 import *
from .ncsile import *
from .density import *
from .negf import *

__all__ = [s for s in dir() if not s.startswith('_')]
