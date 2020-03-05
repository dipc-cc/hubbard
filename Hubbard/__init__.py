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
    NEGF

Read and write in binary files
==============================

.. autosummary::
   :toctree:

   ncSileHubbard

"""

from .info import git_revision as __git_revision__
from .info import version as __version__

from .hamiltonian import *
from . import plot
from .sp2 import *
from .ncsile import *
from .density import *
from . import geometry
