import numpy as np
import sisl

__all__ = ['ncSilehubbard']


class ncSilehubbard(sisl.SileCDF):
    """ Read and write `hubbard.HubbardHamiltonian` object in binary files (netCDF4 support)

    See Also
    --------
    `sisl.SileCDF` : sisl class
    """

    def read_density(self, group):
        # Find group
        g = self.groups[group]

        # Read densities
        occ = g.variables['occ'][:]
        return occ

    def write_density(self, infolabel, group, occ):
        # Create group
        g = self._crt_grp(self, group)
        g.info = infolabel

        # Create dimensions
        self._crt_dim(self, 'ncomp', occ.shape[0])
        self._crt_dim(self, 'norb', occ.shape[1])

        # Write variable occ
        v = self._crt_var(g, 'occ', ('f8', 'f8'), ('ncomp', 'norb'))
        v.info = 'Densities'
        g.variables['occ'][:] = occ
