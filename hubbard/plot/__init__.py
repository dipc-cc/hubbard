"""
=======================================================
Visualizing electronic properties (:mod:`hubbard.plot`)
=======================================================

.. module:: hubbard.plot
    :noindex:

Functionality for plotting different physical properties of the
self-consistent solution for the system.

Plot spin-resolved quantities
=============================

.. autosummary::
   :toctree:

    Charge
    ChargeDifference
    SpinPolarization

Plot spectra
============

.. autosummary::
   :toctree:

    LDOS
    LDOS_from_eigenstate
    DOSmap
    Spectrum

Plot wavefunctions for each spin-channel
========================================

.. autosummary::
   :toctree:

   Wavefunction

Plot geometrical functions
==========================

.. autosummary::
   :toctree:

    BondOrder
    BondHoppings
    Bonds

Plot bandstructure of the system
================================

.. autosummary::
   :toctree:

    Bandstructure

"""

from .plot import *
from .charge import *
from .wavefunction import *
from .spectrum import *
from .bandstructure import *
from .bonds import *

__all__ = [s for s in dir() if not s.startswith('_')]
