"""
===========================================
Common geometries (:mod:`Hubbard.geometry`)
===========================================

.. module:: Hubbard.geometry
    :noindex:

Functions to create particular geometries
=========================================

.. autosummary::
   :toctree:

    agnr
    zgnr
    cgnr
    agnr2B
    ssh

Saturate edge atoms of sp2 systems with H-atoms
===============================================

.. autosummary::
   :toctree:

    add_Hatoms

"""

from .graphene_ribbon import *
from .ssh import *
from .add_Hatoms import *

__all__ = [s for s in dir() if not s.startswith('_')]
