import numpy as np
import sisl

__all__ = ['ncSileHubbard']


class ncSileHubbard(sisl.SileCDF):
    """ Read and write `hubbard.HubbardHamiltonian` object in binary files (netCDF4 support)

    See Also
    --------
    sisl.io.SileCDF : sisl class
    """
    def read_U(self, group=None, index=0):
        """ Read Coulomb repulsion U parameter from netcdf file

        Parameters
        ----------
        group: str, optional
           netcdf group
        index: int or list of ints, optional
            If there are groups in the file and no group is specified then it reads the U parameter from the index item in the groups dictionary

        Returns
        -------
        Float, numpy.ndarray or list depending on the saved U and if `index` is a list
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
                if self.groups:

                    k = list(self.groups.keys())[index]

                    if isinstance(k,list):
                        U  = []
                        for i in k:
                            g = self.groups[i]
                            # Read U from selcted groups and append them in a list
                            _U = np.array(g.variables['U'][:])
                            U.append(_U)
                    else:
                        g = self.groups[k]
                        U = np.array(g.variables['U'][:])
        return U

    def read_kT(self, group=None, index=0):
        """ Read temperature from netcdf file

        Parameters
        ----------
        group: str, optional
           netcdf group
        index: int or list of ints, optional
            If there are groups in the file and no group is specified then it reads the temperature (kT) from the index item in the groups dictionary

        Returns
        -------
        Float or list of floats containing the temperature times the Boltzmann constant (k)
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
                if self.groups:

                    k = list(self.groups.keys())[index]

                    if isinstance(k,list):
                        kT  = []
                        for i in k:
                            g = self.groups[i]
                            # Read U from selected groups and append them in a list
                            _kT = np.array(g.variables['kT'][:])
                            kT.append(_kT)
                    else:
                        g = self.groups[k]
                        kT = np.array(g.variables['kT'][:])
        return kT

    def read_density(self, group=None, index=0):
        """ Read density from netcdf file

        Parameters
        ----------
        group: str, optional
           netcdf group
        index: int or list of ints, optional
            If there are groups in the file and no group is specified then it reads the n variable (density) from the index item in the groups dictionary


        Returns
        -------
        numpy.ndarray or list of numpy.ndarrays, depending if `index` is a list

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
                if self.groups:
                    k = list(self.groups.keys())[index]

                    if isinstance(k,list):
                        n  = []
                        for i in k:
                            g = self.groups[i]
                            # Read U from selected groups and append them in a list
                            _n = np.array(g.variables['n'][:])
                            n.append(_n)
                    else:
                        g = self.groups[k]
                        n = np.array(g.variables['n'][:])
        return n

    def write_density(self, n, U, kT, units, Uij=None, group=None):
        # Create group
        if group is not None:
            g = self._crt_grp(self, group)
        else:
            g = self

        # Create dimensions
        self._crt_dim(self, 'nspin', n.shape[0])
        self._crt_dim(self, 'norb', n.shape[1])

        # Write variable
        v1 = self._crt_var(g, 'n', ('f8', 'f8'), ('nspin', 'norb'))

        if isinstance(U, np.ndarray):
            v2 = self._crt_var(g, 'U', ('f8'), ('norb'))
        elif isinstance(U, (float, int)):
            v2 = self._crt_var(g, 'U', 'f8')

        v3 = self._crt_var(g, 'kT', 'f8')

        v1.info = 'Spin densities'
        v2.info = 'Coulomb repulsion parameter in ' + units
        v3.info = 'Temperature of the system in '+ units
        v1[:] = n
        v2[:] = U
        v3[:] = kT

        if Uij is not None:

            v4 = self._crt_var(g, 'Uij', ('f8', 'f8'), ('norb', 'norb'))
            v4.info = 'Off diagonal Coulomb repulsion elements in' + units
            v4[:] = Uij

sisl.io.add_sile("HU.nc", ncSileHubbard, gzip=False)
