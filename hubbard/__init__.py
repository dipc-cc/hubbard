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

from . import info
from .info import (
    git_revision as __git_revision__,
    version as __version__,
    major as __major__,
    minor as __minor__,
    micro as __micro__,
)

__all__ = [s for s in dir() if not s.startswith('_')]
__all__ += [f'__{s}__' for s in ['version', 'major', 'minor', 'micro']]
__all__ += [f'__{s}__' for s in ['git_revision']]
