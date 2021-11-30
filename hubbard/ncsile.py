import numpy as np
import sisl

__all__ = ['ncSileHubbard']


class ncSileHubbard(sisl.SileCDF):
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
                raise ValueError(f'group {group} does not exist in file {self._file}')
        else:
            # Try reading U saved without group
            try:
                U = np.array(self.variables['U'][:])
            except:
                # Read from all groups, append all U
                if self.groups:
                    U  = []
                    for k in self.groups:
                        g = self.groups[k]
                        # Read U from all groups and append them in a list
                        _U = np.array(g.variables['U'][:])
                        U.append(_U)
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
                raise ValueError(f'group {group} does not exist in file {self._file}')
        else:
            # Try reading kT saved without group
            try:
                kT = np.array(self.variables['kT'][:])
            except:
                # Read from all groups, append all kT
                if self.groups:
                    kT  = []
                    for k in self.groups:
                        g = self.groups[k]
                        # Read kT from all groups and append them in a list
                        _kT = np.array(g.variables['kT'][:])
                        kT.append(_kT)
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
                raise ValueError(f'group {group} does not exist in file {self._file}')
        else:
            # Try reading densities saved without group
            try:
                n = np.array(self.variables['n'][:])
            except:
                # Read from all groups, append all densities
                if self.groups:
                    n  = []
                    for k in self.groups:
                        g = self.groups[k]
                        # Read densities from all groups and append them in a list
                        _n = np.array(g.variables['n'][:])
                        n.append(_n)
        return n

    def write_density(self, n, U, kT, units, group=None):
        # Create group
        if group is not None:
            g = self._crt_grp(self, group)
        else:
            g = self

        # Create dimensions
        self._crt_dim(self, 'nspin', n.shape[0])
        self._crt_dim(self, 'norb', n.shape[1])

        # Write variable n
        v1 = self._crt_var(g, 'n', ('f8', 'f8'), ('nspin', 'norb'))
        v2 = self._crt_var(g, 'U', ('f8', 'f8'), ('norb', 'norb'))
        v3 = self._crt_var(g, 'kT', 'f8')
        v1.info = 'Spin densities'
        v2.info = 'Coulomb repulsion parameter in ' + units
        v3.info = 'Temperature of the system in '+ units
        v1[:] = n
        v2[:] = U
        v3[:] = kT

sisl.io.add_sile("HU.nc", ncSileHubbard, gzip=False)
