"""

:mod:`Hubbard.ncsile`
==========================

.. currentmodule:: Hubbard.ncsile

"""

from __future__ import print_function
import numpy as np
import sisl

class ncSileHubbard(sisl.SileCDF):

    def read_density(self, group):
        # Find group
        g = self.groups[group]

        # Read densities
        nup = g.variables['nup'][:]
        ndn = g.variables['ndn'][:]
        return nup, ndn

    def write_density(self, infolabel, group, nup, ndn):
        # Create group
        g = self._crt_grp(self, group)
        g.info = infolabel

        # Create dimensions
        self._crt_dim(self, 'norb', len(nup))

        # Write variable nup
        v = self._crt_var(g, 'nup', 'f8', ('norb', ))
        v.info = 'Density spin-up'
        g.variables['nup'][:] = nup

        # Write variable ndn
        v = self._crt_var(g, 'ndn', 'f8', ('norb', ))
        v.info = 'Density spin-down'
        g.variables['ndn'][:] = ndn
