import numpy as np
import sisl

__all__ = ['ncSilehubbard']


class ncSilehubbard(sisl.SileCDF):
    """ Read and write `hubbard.HubbardHamiltonian` object in binary files (netCDF4 support)

    See Also
    ------------
    sisl.io.SileCDF : sisl class
    """

    def read_density(self, group):
        # Find group
        g = self.groups[group]

        # Read densities
        n = g.variables['n'][:]
        return n

    def write_density(self, infolabel, group, n):
        # Create group
        g = self._crt_grp(self, group)
        g.info = infolabel

        # Create dimensions
        self._crt_dim(self, 'ncomp', n.shape[0])
        self._crt_dim(self, 'norb', n.shape[1])

        # Write variable n
        v = self._crt_var(g, 'n', ('f8', 'f8'), ('ncomp', 'norb'))
        v.info = 'Densities'
        g.variables['n'][:] = n
