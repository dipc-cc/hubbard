import numpy as np
import sisl

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
        """
        Read Coulomb repulsion U parameter from netcdf file

        Parameters:
        -----------
        group: str, optional
            netcdf group

        Returns:
        --------
        Float or list of floats containing the temperature for each netcdf group
        """
        # Find group
        if group is not None:
            if group in self.groups:
                g = self.groups[group]
                U = np.array(g.variables['U'][:])
            else:
                raise ValueError(f'group {group} does not exist in file {file}')
        else:
            if self.groups:
                U = []
                for k in self.groups:
                    # Read Coulomb repulsion U parameter from all groups and append them in a list
                    g = self.groups[k]
                    _U = np.array(g.variables['U'][:])
                    U.append(_U)
            else:
                U = np.array(self.variables['U'][:])
        return U

    def read_kT(self, group=None):
        """
        Read temperature from netcdf file

        Parameters:
        -----------
        group: str, optional
            netcdf group

        Returns:
        --------
        Float or list of floats containing the temperature times the Boltzmann constant (k) for each netcdf group
        """
        # Find group
        if group is not None:
            if group in self.groups:
                g = self.groups[group]
                # Read kT
                kT = np.array(g.variables['kT'][:])
            else:
                raise ValueError(f'group {group} does not exist in file {file}')
        else:
            if self.groups:
                kT = []
                for k in self.groups:
                    g = self.groups[k]
                    # Read temperatures from all groups and append them in a list
                    _kT = np.array(g.variables['kT'][:])
                    kT.append(_kT)
            else:
                kT = np.array(self.variables['kT'][:])
        return kT

    def read_density(self, group=None):
        """
        Read density from netcdf file

        Parameters:
        -----------
        group: str, optional
           netcdf group. If there are groups in the file and no group is found then it reads the n variable from all groups found

        Return:
        -------
        numpy.ndarray or list of numpy.ndarrays
        """
        # Find group
        if group is not None:
            if group in self.groups:
                g = self.groups[group]
                n = np.array(g.variables['n'][:])
            else:
                raise ValueError(f'group {group} does not exist in file {file}')
        else:
            if self.groups:
                n = []
                for k in self.groups:
                    g = self.groups[k]
                    # Read densities from all groups and append them in a list
                    _n = np.array(g.variables['n'][:])
                    n.append(_n)
            else:
                # Read densities
                n = np.array(self.variables['n'][:])
        return n

    def write_density(self, n, U, kT, group=None):
        # Create group
        if group is not None:
            g = self._crt_grp(self, group)
        else:
            g = self

        # Create dimensions
        self._crt_dim(self, 'nspin', n.shape[0])
        self._crt_dim(self, 'norb', n.shape[1])

        # Write variable n
        v = self._crt_var(g, 'n', ('f8', 'f8'), ('nspin', 'norb'))
        v = self._crt_var(g, 'U', 'f8')
        v = self._crt_var(g, 'kT', 'f8')
        v.info = 'Mean Field Hubbard calculation'
        g.variables['n'][:] = n
        g.variables['U'][:] = U
        g.variables['kT'][:] = kT
