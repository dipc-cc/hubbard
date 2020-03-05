"""
=======================================================
Visualizing electronic properties (:mod:`Hubbard.plot`)
=======================================================

.. module:: Hubbard.plot
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

    DOS
    DOS_distribution
    LDOSmap
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
