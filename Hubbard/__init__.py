"""
=========================================
Mean Field Hubbard model (:mod:`Hubbard`)
=========================================

.. module:: Hubbard
    :noindex:

Package for mean-field Hubbard approximation

Self Consistent Field class
===========================

.. autosummary::
   :toctree:

    HubbardHamiltonian
    dm
    NEGF

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


"""

from .hamiltonian import *
from . import plot
from .sp2 import *
from .ncsile import *
from .density import *
from .negf import *
from . import geometry

__all__ = [s for s in dir() if not s.startswith('_')]
