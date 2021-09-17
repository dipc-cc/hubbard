import numpy as np
import sisl
import warnings

__all__ = ['ncSilehubbard']


def get_sile(file, *args, **kwargs):
    # returns a sile object
    return ncSilehubbard(file, *args, **kwargs)

class ncSilehubbard(sisl.SileCDF):
    """ Read and write `hubbard.HubbardHamiltonian` object in binary files (netCDF4 support)

    See Also
    ------------
    sisl.io.SileCDF : sisl class
    """
    def read_U(self, group=None):
        # Find group
        if group is not None:
            if group in self.groups:
                g = self.groups[group]
            else:
                warnings.warn(f'group {group} does not exist in file {file}')
        else:
            if self.groups:
                for k in self.groups:
                    # If there are any groups read the density from the first one
                    g = self.groups[k]
                    break
            else:
                g = self
        # Read U
        U = g.variables['U'][:]
        return U

    def read_kT(self, group=None):
        # Find group
        if group is not None:
            if group in self.groups:
                g = self.groups[group]
            else:
                warnings.warn(f'group {group} does not exist in file {file}')
        else:
            if self.groups:
                for k in self.groups:
                    # If there are any groups read the density from the first one
                    g = self.groups[k]
                    break
            else:
                g = self
        # Read kT
        kT = g.variables['kT'][:]
        return kT

    def read_density(self, group=None):
        # Find group
        if group is not None:
            if group in self.groups:
                g = self.groups[group]
            else:
                warnings.warn(f'group {group} does not exist in file {file}')
        else:
            if self.groups:
                for k in self.groups:
                    # If there are any groups read the density from the first one
                    g = self.groups[k]
                    break
            else:
                g = self

        # Read densities
        n = g.variables['n'][:]
        return n

    def write_density(self, n, U, kT, group=None):
        # Create group
        if group is not None:
            g = self._crt_grp(self, group)
        else:
            g = self

        # Create dimensions
        self._crt_dim(self, 'ncomp', n.shape[0])
        self._crt_dim(self, 'norb', n.shape[1])

        # Write variable n
        v = self._crt_var(g, 'n', ('f8', 'f8'), ('ncomp', 'norb'))
        v = self._crt_var(g, 'U', 'f8')
        v = self._crt_var(g, 'kT', 'f8')
        v.info = 'Densities'
        g.variables['n'][:] = n
        g.variables['U'][:] = U
        g.variables['kT'][:] = kT
