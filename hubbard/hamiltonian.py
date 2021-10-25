import numpy as np
import sisl
import hubbard.ncsile as nc
import hashlib
import os
import math
import warnings
_pi = math.pi

__all__ = ['HubbardHamiltonian']


class HubbardHamiltonian(object):
    """ A class to create a Self Consistent field (SCF) object related to the mean-field Hubbard (MFH) model

    The `hubbard.HubbardHamiltonian` class opens the possibility to include electron correlations in the tight-binding Hamiltonian
    by solving self-consistently the mean-field Hubbard Hamiltonian

    It enables the convergence of several tight-binding described systems towards a user-defined tolerance criterion

    It takes an input tight-binding Hamiltonian and updates the corresponding matrix elements according to the MFH model

    Parameters
    ----------
    TBHam: sisl.physics.Hamiltonian
        An unpolarized or spin-polarized tight-binding Hamiltonian
    n: numpy.ndarray, optional
        initial spin-densities vectors. The shape of `n` must be (spin deg. of freedom, no. of sites)
    U: float, optional
        on-site Coulomb repulsion
    q: array_like, optional
        One or two values specifying the total charge associated to each spin component.
        The array should contain as many values as the dimension of the problem. I.e., if the
        Hamiltonian is unpolarized q should be a single value
    nkpt: array_like or sisl.physics.BrillouinZone, optional
        Number of k-points along (a1, a2, a3) for Monkhorst-Pack BZ sampling
    kT: float, optional
        Temperature of the system in units of the Boltzmann constant
    """

    def __init__(self, TBHam, n=0, U=0.0, q=(0., 0.), nkpt=[1, 1, 1], kT=1e-5):
        """ Initialize HubbardHamiltonian """

        self.U = U # hubbard onsite Coulomb parameter

        # Copy TB Hamiltonian to store the converged one in a different variable
        self.TBHam = TBHam
        self.H = TBHam.copy()
        self.H.finalize()
        self.geometry = TBHam.geometry
        # So far we only consider either unpolarized or spin-polarized Hamiltonians
        self.spin_size = self.H.spin.spinor

        # Use sum of all matrix elements as a basis for hash function calls
        H0 = self.TBHam.copy()
        H0.shift(np.pi) # Apply a shift to incorporate effect of S
        if self.spin_size > 1:
            s = H0.H.tocsr(0).data.tostring() + H0.H.tocsr(1).data.tostring()
        else:
            s = H0.H.tocsr(0).data.tostring()
        self._hash_base = s
        del H0

        # Total initial charge
        ntot = self.geometry.q0
        if ntot == 0:
            ntot = len(self.geometry)

        self.q = np.array(q, dtype=np.float64).copy()
        # Ensure the length of q is equal to the dimension of the problem
        self.q = self.q[:self.spin_size]

        if self.q[0] <= 0:
            if self.spin_size == 2:
                if self.q[1] <= 0:
                    self.q[1] = ntot // 2
                self.q[0] = ntot - self.q[1]
            else:
                self.q[0] = ntot / 2

        self.sites = len(self.geometry)
        self._update_e0()

        # Set k-mesh
        self.set_kmesh(nkpt)

        self.kT = kT

        # Initialize spin-densities vector
        if not isinstance(n, (np.ndarray, list)):
            self.n = 0.5*np.ones((self.spin_size, self.sites))
        else:
            # Ensure n is an array
            n = np.array(n)
            if n.shape != (self.spin_size, self.sites):
                warnings.warn("Incorrect shape of n, initializing spin densities")
                self.n = (1./self.spin_size)*np.ones((self.spin_size, self.sites))
            else:
                self.n = n
        # Ensure normalized charge
        self.normalize_charge()

    def set_kmesh(self, nkpt=[1, 1, 1]):
        """ Set the k-mesh for the HubbardHamiltonian

        Parameters
        ----------
        nkpt : array_like or sisl.physics.BrillouinZone, optional
            k-mesh to be associated with the `hubbard.HubbardHamiltonian` instance
        """
        if isinstance(nkpt, sisl.BrillouinZone):
            self.mp = nkpt
        elif isinstance(nkpt, (np.ndarray, list)):
            self.mp = sisl.MonkhorstPack(self.H, nkpt)
        else:
            raise ValueError(self.__class__.__name__ + '.set_kmesh(...) requires an array_like input')

    def __str__(self):
        """ Representation of the model """
        s = self.__class__.__name__ + f'{{q: {self.q}, U: {self.U}, kT: {self.kT}\n'
        s += str(self.H).replace('\n', '\n ')
        return s + '\n}'

    def eigh(self, k=[0, 0, 0], eigvals_only=True, spin=0):
        """ Diagonalize Hamiltonian using the ``eigh`` routine

        Parameters
        ----------
        k: array_like, optional
            k-point at which the eigenvalue want to be obtained
        eigvals_only: bool, optional
            if True only eigenvalues are returned, otherwise
            it also returns the eigenvectors
        spin: int, optional
            for ``spin`` = 0 (1) it solves the eigenvalue problem for the spin up (down) Hamiltonian

        See Also
        ------------
        sisl.physics.SparseOrbitalBZ.eigh : sisl class

        Returns
        -------
        eigenvalues: numpy.ndarray
        """
        return self.H.eigh(k=k, eigvals_only=eigvals_only, spin=spin)

    def eigenstate(self, k, spin=0):
        """ Solve the eigenvalue problem at `k` and return it as a `sisl.physics.electron.EigenstateElectron` object containing all eigenstates

        Parameters
        ----------
        k: array_like
            k-point at which the eigenstate is going to be obtained
        spin: int, optional
            for spin=0(1) it solves the eigenvalue problem for the spin up (down) Hamiltonian

        Returns
        -------
        object: `sisl.physics.electron.EigenstateElectron` object
        """
        return self.H.eigenstate(k, spin=spin)

    def tile(self, reps, axis):
        """ Tile the HubbardHamiltonian object along a specified axis to obtain a larger one

        Parameters
        ----------
        reps: int
           number of tiles (repetitions)
        axis: int
            direction of tiling, 0, 1, 2 according to the cell-direction

        See Also
        ------------
        sisl.Geometry.tile: sisl class method

        Returns
        -------
        A new larger `hubbard.HubbardHamiltonian` object
        """
        Htile = self.H.tile(reps, axis)
        ntile = np.tile(self.n, reps)
        q = ntile.sum(1)
        return self.__class__(Htile, n=ntile, U=self.U, q=q, nkpt=self.mp, kT=self.kT)

    def repeat(self, reps, axis):
        """ Repeat the HubbardHamiltonian object along a specified axis to obtain a larger one

        Parameters
        ----------
        reps: int
           number of repetitions
        axis: int
            direction of tiling, 0, 1, 2 according to the cell-direction

        See Also
        ------------
        sisl.Geometry.repeat: sisl class method

        Returns
        -------
        A new larger `hubbard.HubbardHamiltonian` object
        """
        Hrep = self.H.repeat(reps, axis)
        nrep = np.repeat(self.n, reps, axis=1)
        q = nrep.sum(1)
        return self.__class__(Hrep, n=nrep, U=self.U, q=q, nkpt=self.mp, kT=self.kT)

    def remove(self, atoms, q=(0,0)):
        """ Remove a subset of this sparse matrix by only retaining the atoms corresponding to `atom`

        Parameters
        ----------
        atoms: array_like of int
            atomic ndices of removed atoms
        q: array_like, optional
            Two values specifying up, down electron occupations for the remaining subset of atoms after removal

        See Also
        ------------
        sisl.physics.Hamiltonian.remove
        sisl.Geometry.remove
        """
        atoms = self.geometry.sc2uc(atoms)
        import sisl._array as _a
        atoms = np.delete(_a.arangei(self.geometry.na), atoms)
        Hsub = self.H.sub(atoms)
        nsub = self.n[:,atoms]
        return self.__class__(Hsub, n=nsub, U=self.U,
                    q=q, nkpt=self.mp, kT=self.kT)

    def sub(self, atoms, q=(0,0)):
        """ Return a new `HubbardHamiltonian` object of a subset of selected atoms

        Parameters
        ----------
        atoms : int or array_like
            indices/boolean of all atoms to be kept
        q: array_like, optional
            Two values specifying up, down electron occupations for the subset of atoms
        """
        nsub = self.n[:, atoms]
        return self.__class__(self.H.sub(atoms), n=nsub, U=self.U,
                    q=q, nkpt=self.mp, kT=self.kT)

    def copy(self):
        """ Return a copy of the `HubbardHamiltonian` object """
        return self.__class__(self.H, n=self.n, U=self.U,
                    q=(self.q[0], self.q[1]), nkpt=self.mp, kT=self.kT)

    def _update_e0(self):
        """ Internal routine to update e0 """
        self.e0 = np.empty((self.spin_size, self.sites))
        for spin in range(self.spin_size):
            e = self.H.tocsr(spin).diagonal()
            self.e0[spin] = e

    def update_hamiltonian(self):
        """ Update spin Hamiltonian according to the mean-field Hubbard model
        It updtates the diagonal elements for each spin Hamiltonian with the opposite spin densities

        Notes
        -----
        This method has to be generalized for inter-atomic Coulomb repulsion also
        """

        # TODO Generalize this method for inter-atomic Coulomb repulsion also

        q0 = self.geometry.atoms.q0
        E = self.e0.copy()
        ispin = np.arange(self.spin_size)[::-1]
        E += self.U * (self.n[ispin, :] - q0)
        a = np.arange(len(self.H))
        self.H[a, a, range(self.spin_size)] = E.T

    def random_density(self):
        """ Initialize spin polarization  with random density """
        self.n = np.random.rand(self.spin_size, self.sites)
        self.normalize_charge()

    def normalize_charge(self):
        """ Ensure the total up/down charge in pi-network equals Nup/Ndn """
        self.n *= (self.q / self.n.sum(1)).reshape(-1, 1)

    def set_polarization(self, up, dn=tuple()):
        """ Maximize spin polarization on specific atomic sites
        Optionally, sites with down-polarization can be specified

        Parameters
        ----------
        up: array_like
            atomic sites where the spin-up density is going to be maximized
        dn: array_like, optional
            atomic sites where the spin-down density is going to be maximized
        """
        self.n[:, up] = np.array([1., 0.]).reshape(2, 1)
        if len(dn) > 0:
            self.n[:, dn] = np.array([0., 1.]).reshape(2, 1)
        self.normalize_charge()

    def polarize_sublattices(self):
        """ Quick way to polarize the lattices
        without checking that consequtive atoms actually belong to
        different sublattices """
        a = np.arange(len(self.H))
        self.n[0, :] = a % 2
        self.n[1, :] = 1 - self.n[0, :]
        self.normalize_charge()

    def find_midgap(self):
        """ Find the midgap for the system
        taking into account the up and dn different spectrums

        This method makes sense for insulators (where there is a bandgap)

        Returns
        -------
        midgap: float
        """
        HOMO, LUMO = -1e10, 1e10
        ev = np.zeros((self.spin_size, self.sites))
        for k in self.mp.k:
            for s in range(self.spin_size):
                ev[s] = self.eigh(k=k, spin=s)
            # If the Hamiltonian is not spin-polarized then ev is just repeated
            HOMO = max(HOMO, ev[0, int(round(self.q[0] - 1))], ev[-1, int(round(self.q[-1] - 1))])
            LUMO = min(LUMO, ev[0, int(round(self.q[0]))], ev[-1, int(round(self.q[-1]))])
        midgap = (HOMO + LUMO) * 0.5
        return midgap

    def fermi_level(self, q=[None, None], dist='fermi_dirac'):
        """ Find the fermi level for a certain charge `q` at a certain `kT`

        Parameters
        ----------
        q: array_like, optional
            charge per spin channel. First index for spin up, second index for dn
            If the Hamiltonian is unpolarized q should have only one component
            otherwise it will take the first one
        dist: str or sisl.distribution, optional
            distribution function

        See Also
        ------------
        sisl.physics.Hamiltonian.fermi_level : sisl class function

        Returns
        -------
        Ef: numpy.array
            Fermi-level for each spin channel
        """
        Q = 1 * q
        for i in range(self.spin_size):
            if Q[i] is None:
                Q[i] = self.q[i]
        if isinstance(dist, str):
            dist = sisl.get_distribution(dist, smearing=self.kT)

        Ef = self.H.fermi_level(self.mp, q=Q, distribution=dist)
        return Ef

    def shift(self, E):
        """ Shift the electronic structure by a constant energy (in-place operation)

        Parameters
        ----------
        E: float or (2,)
            the energy (in eV) to shift the electronic structure, if two values are passed the two spin-components get shifted individually

        See Also
        ------------
        sisl.physics.Hamiltonian.shift
        """
        self.H.shift(E)

    def __hash__(self):
        return hashlib.md5((self.q.tostring()+np.array([self.U]).tostring()+np.array([self.kT]).tostring()+self._hash_base)).hexdigest()[:7]

    def read_density(self, fn, mode='r', group=None):
        """ Read density from binary file

        Parameters
        ----------
        fn: str
            name of the file that is going to read from
        mode: str, optional
            mode in which the file is opened
        group: str, optional
            netCDF4 group
        """
        if os.path.isfile(fn):
            fh = nc.get_sile(fn, mode=mode)
            if group is not None:
                if group in fh.groups:
                    self.n = fh.read_density(group)
            else:
                warnings.warn(f'Groups found in {fn}, using the density from the first one')
                # Read only the first element from the list
                self.n = fh.read_density()[0]
            self.update_hamiltonian()

    def write_density(self, fn, mode='w', group=None):
        """ Write density in a binary file

        Parameters
        ----------
        fn: str
            name of the file in which the densities are going to be stored
        mode: str, optional
            mode in which the file is opened
        group: str, optional
            netCDF4 group
        """
        if not os.path.isfile(fn):
            mode = 'w'
        fh = nc.get_sile(fn, mode=mode)
        fh.write_density(self.n, self.U, self.kT, group)

    def write_initspin(self, fn, ext_geom=None, spinfix=True, mode='a', eps=0.1):
        """ Write spin polarization to SIESTA fdf-block
        This function only makes sense for spin-polarized calculations

        Parameters
        ----------
        fn: str
            name of the fdf-file
        ext_geom: sisl.Geometry, optional
            an "external" geometry that contains the sp2-sites included in the simulation
        spinfix: bool, optional
            specifies if the Spin.Fix and Spin.Total lines are written to the fdf
        mode: str, optional
            mode in which the file is opened
        eps: float, optional
            atoms within this distance will be considered equivalent in case ``ext_geom != geometry`` (see `sisl.Geometry.overlap`)
        """
        if not os.path.isfile(fn):
            mode = 'w'
        if ext_geom is None:
            idx = np.arange(len(self.H))
            geom = self.geometry
        elif isinstance(ext_geom, sisl.Geometry):
            idx, idx_internal = ext_geom.overlap(self.geometry, eps=eps)
            geom = ext_geom
        else:
            raise ValueError(self.__class__.__name__ + '.write_initspin(...) requires a sisl.Geometry instance for keyword ext_geom')
        polarization = self.n[0] - self.n[1]
        dq = np.sum(polarization)
        f = open(fn, mode=mode)
        f.write('# hubbard: U=%.3f eV\n' % self.U)
        if spinfix:
            f.write('Spin.Fix True\n')
            f.write('Spin.Total %.6f\n' % dq)
        f.write('%block DM.InitSpin\n')
        for i, ia in enumerate(idx):
            s = '%6i %10.6f' % (ia + 1, polarization[i]) # SIESTA counts from 1
            # add documentation string
            s += ' # %2s' % geom.atoms[ia].symbol
            for j in range(3):
                s += ' %9.4f' % geom.xyz[ia, j]
            f.write(s + '\n')
        f.write('%endblock DM.InitSpin\n\n')

    def iterate(self, calc_n_method, q=None, mixer=None, **kwargs):
        r""" Common method to iterate in a SCF loop that corresponds to the mean-field Hubbard approximation

        The only thing that may change is the way in which the spin-densities (``n``) and total energy (``Etot``) are obtained
        where one needs to use the correct `calc_n_method` for the particular system.

        Parameters
        ----------
        calc_n_method: callable
            method to obtain the spin-densities
            it *must* return the corresponding spin-densities (``n``) and the total energy (``Etot``)
        q: array_like, optional
            total charge separated in spin-channels, q=[q_up, q_dn]
        mixer: Mixer, optional
            `sisl.mixing.Mixer` instance for the SCF loop, defaults to `sisl.mixing.DIISMixer`

        See Also
        ------------
        update_hamiltonian
        hubbard.calc_n: method to obtain ``n`` and ``Etot`` for tight-binding Hamiltonians with finite or periodic boundary conditions at a certain `kT`
        hubbard.NEGF.calc_n_open: method to obtain  ``n`` and ``Etot`` for tight-binding Hamiltonians with open boundary conditions
        sisl.mixing.AdaptiveDIISMixer: for adaptative DIIS (Pulay) mixing scheme
        sisl.mixing.LinearMixer: for linear mixing scheme

        Returns
        -------
        dn : array_like
            difference between the ith and the (i-1)th iteration densities
        """
        if q is None:
            q = self.q
        else:
            for s in self.spin_size:
                if q[s] is None:
                    q[s] = int(round(self.q[s]))

        ni, Etot = calc_n_method(self, q, **kwargs)

        # Measure of density change
        ddm = ni - self.n
        dn = np.absolute(ddm).max()

        # Update occupations on sites with mixing algorithm
        if mixer is None:
            mixer = sisl.mixing.DIISMixer(weight=0.7, history=7)
        self.n = mixer(self.n.ravel(), ddm.ravel()).reshape(self.n.shape)

        # Update spin hamiltonian
        self.update_hamiltonian()

        # Store total energy
        self.Etot = Etot - self.U * (ni[0]*ni[-1]).sum()

        return dn

    def converge(self, calc_n_method, tol=1e-6, mixer=None, steps=100, fn=None, print_info=False, func_args=dict()):
        """ Iterate Hamiltonian towards a specified tolerance criterion

        This method calls `iterate` as many times as it needs until it reaches the specified tolerance

        Parameters
        ----------
        calc_n_method: callable
            method to obtain the spin-densities
            it *must* return the corresponding spin-densities (``n``) and the total energy (``Etot``)
        tol: float, optional
            tolerance criterion
        mixer: Mixer
            `sisl.mixing.Mixer` instance, defaults to `sisl.mixing.DIISMixer`
        steps: int, optional
            the code will print some relevant information (if `print_info` is set to ``True``) about the convergence
            process when the number of completed iterations reaches a multiple of the specified `steps`.
            It also will store the densities in a binary file if `fn` is passed
        fn: str, optional
            optionally, one can save the spin-densities during the calculation (when the number of completed iterations reaches
            the specified `steps`), by giving the name of the full name of the *binary file*
        func_args: dictionary, optional
            function arguments to pass to calc_n_method

        See Also
        ------------
        iterate
        hubbard.calc_n: method to obtain ``n`` and ``Etot`` for tight-binding Hamiltonians with finite or periodic boundary conditions at a certain `kT`
        hubbard.NEGF: class that contains the routines to obtain  ``n`` and ``Etot`` for tight-binding Hamiltonians with open boundary conditions
        sisl.mixing.AdaptiveDIISMixer: for adaptative DIIS (Pulay) mixing scheme
        sisl.mixing.LinearMixer: for linear mixing scheme

        Returns
        -------
        dn
            difference between the ith and the (i-1)th iteration densities
        """
        if print_info:
            print('   HubbardHamiltonian: converge towards tol={:.2e}'.format(tol))
        if mixer is None:
            mixer = sisl.mixing.DIISMixer(weight=0.7, history=7)
        dn = 1.0
        i = 0
        while dn > tol:
            i += 1
            dn = self.iterate(calc_n_method, mixer=mixer, **func_args)
            if i % steps == 0:
                # Print some info from time to time
                if print_info:
                    print('   %i iterations completed:' % i, dn, self.Etot)
                if fn:
                    self.write_density(fn)
        if print_info:
            print('   found solution in %i iterations' % i)
        return dn

    def calc_orbital_charge_overlaps(self, k=[0, 0, 0], spin=0):
        r""" Obtain orbital (eigenstate) charge overlaps as :math:`\int dr |\psi_{\sigma\alpha}|^{4}`

        Where :math:`\sigma` is the spin index and :math:`\alpha` is the eigenstate index

        Parameters
        ----------
        k: array_like, optional
            k-point at which the eigenstate is going to be obtained
        spin: int, optional
            spin index

        Returns
        -------
        ev: numpy.ndarray
            eigenvalues
        L: numpy.ndarray
            orbital charge overlaps
        """
        ev, evec = self.eigh(k=k, eigvals_only=False, spin=spin)
        # Compute orbital charge overlaps
        L = np.einsum('ia,ia,ib,ib->ab', evec, evec, evec, evec).real
        return ev, L

    def get_Zak_phase(self, func=None, nk=51, sub='filled', eigvals=False):
        """ Computes the Zak phase for 1D (periodic) systems using `sisl.physics.electron.berry_phase`

        Parameters
        ----------
        func: callable, optional
            function that creates a list of parametrized k-points to generate a new `sisl.physics.BrillouinZone` object parametrized in `N` separations
        nk: int, optional
            number of k-points generated using the parameterization
        sub: int, optional
            number of bands that will be summed to obtain the Zak phase

        Notes
        -----
        If no `func` is passed it assumes the periodicity along the x-axis
        If no `sub` is passed it sums up to the last occuppied band (included)

        Returns
        -------
        Zak: float
            Zak phase for the 1D system
        """

        if not func:
            # Discretize kx over [0.0, 1.0[ in Nx-1 segments (1BZ)
            def func(sc, frac):
                return [frac, 0, 0]
        bz = sisl.BrillouinZone.parametrize(self.H, func, nk)
        if sub == 'filled':
            # Sum up over all occupied bands:
            sub = np.arange(int(round(self.q[0])))
        return sisl.electron.berry_phase(bz, sub=sub, eigvals=eigvals, method='zak')

    def get_bond_order(self, format='csr', midgap=0.):
        """ Compute Huckel bond order

        Parameters
        ----------
        format: {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the `scipy.sparse.csr_matrix`,
           however if one always requires operations on dense matrices, one can always
           return in `numpy.ndarray` (`'array'`) or `numpy.matrix` (`'dense'`).
        midgap: float, optional
           energy value that separates filled states (lower energy) from empty states (higher energy) 

        Returns
        -------
        the Huckel bond-order matrix object
        """
        g = self.geometry
        BO = sisl.Hamiltonian(g)
        R = [0.1, 1.6]
        for w, k in zip(self.mp.weight, self.mp.k):
            # spin-up first
            ev, evec = self.eigh(k=k, eigvals_only=False, spin=0)
            ev -= midgap
            idx = np.where(ev < 0.)[0]
            bo = np.dot(np.conj(evec[:, idx]), evec[:, idx].T)
            # add spin-down
            ev, evec = self.eigh(k=k, eigvals_only=False, spin=1)
            ev -= midgap
            idx = np.where(ev < 0.)[0]
            bo += np.dot(np.conj(evec[:, idx]), evec[:, idx].T)
            for ix in (-1, 0, 1):
                for iy in (-1, 0, 1):
                    for iz in (-1, 0, 1):
                        r = (ix, iy, iz)
                        phase = np.exp(-2.j * np.pi * np.dot(k, r))
                        for ia in g:
                            for ja in g.close_sc(ia, R=R, isc=r)[1]:
                                bor = bo[ia, ja] * phase
                                BO[ia, ja] += w * bor.real
        # Add sigma bond at the end
        for ia in g:
            idx = g.close(ia, R=R)
            BO[ia, idx[1]] += 1.
        return BO.Hk(format=format) # Fold to Gamma

    def spin_contamination(self, ret_exact=False):
        r""" Obtains the spin contamination for the MFH Hamiltonian following
        `Ref. Chemical Physics Letters. 183 (5): 423–431 <https://www.sciencedirect.com/science/article/abs/pii/000926149190405X?via%3Dihub>`_.

        .. math::

            \langle S^{2} \rangle_{MFH} = \langle S^{2} \rangle_{exact} + N_{\beta} - \sum_{ij}^{occ} |\langle \psi^{\alpha}_{i}|\psi^{\beta}_{j}\rangle|^{2}

        Where the exact spin squared expectation value is obtained as

        .. math::

            \langle S^{2} \rangle_{exact}=(\frac{ N_{\alpha}-N_{\beta}}{2})(\frac{N_{\alpha}-N_{\beta} }{2} + 1)

        Notes
        -----
        The current implementation works for non-periodic systems only.

        Parameters
        ----------
        ret_exact: bool, optional
            If true this method will return also the exact spin squared expectation value

        See Also
        ------------
        sisl.physics.electron.spin_squared: sisl class function

        Returns
        -------
        S_MFH: float
            expectation value for the MFH Hamiltonian
        S: float
            exact expectation value
        """

        # Define Nalpha and Nbeta, where Nalpha >= Nbeta
        Nalpha = np.amax(self.q)
        Nbeta = np.amin(self.q)

        # Exact Total Spin expected value (< S² >)
        S = (Nalpha - Nbeta) * ((Nalpha - Nbeta) * .25 + 0.5)

        # Extract eigenvalues and eigenvectors of spin-up and spin-dn electrons
        ev, evec = np.zeros((self.spin_size, self.sites)), np.zeros((self.spin_size, self.sites, self.sites))
        for s in range(self.spin_size):
            ev[s], evec[s] = self.eigh(eigvals_only=False, spin=s)

        # No need to tell which matrix of eigenstates correspond to alpha or beta,
        # the sisl function spin_squared already takes this into account
        s2alpha, s2beta = sisl.electron.spin_squared(evec[0, :, :int(round(self.q[0]))].T, evec[-1, :, :int(round(self.q[-1]))].T)

        # Spin contamination
        S_MFH = S + Nbeta - s2beta.sum()

        if ret_exact:
            return S_MFH, S
        else:
            return S_MFH

    def DOS(self, egrid, eta=1e-3, spin=[0, 1], dist='Lorentzian', eref=0.):
        """ Obtains the density of states (DOS) of the system with a distribution function

        Parameters
        ----------
        egrid: array_like
            Energy grid at which the DOS will be calculated.
        eta: float, optional
            Smearing parameter
        spin: int, optional
            If spin=0(1) it calculates the DOS for up (down) electrons in the system.
            If spin is not specified it returns DOS_up + DOS_dn.
        dist: str or sisl.physics.distribution, optional
            distribution for the convolution, defaults to Lorentzian
        eref: float, optional
            energy reference, defaults to zero

        See Also
        ------------
        sisl.physics.get_distribution: sisl method to create distribution function
        sisl.physics.electron.DOS: sisl method to obtain DOS

        Returns
        -------
        DOS: numpy.ndarray
            density of states at the given energies for the selected spin
        """
        # Ensure spin is iterable
        if not isinstance(spin, (list, np.ndarray)):
            spin = [spin]

        # Check if egrid is numpy.ndarray
        if not isinstance(egrid, np.ndarray):
            egrid = np.array(egrid)

        if isinstance(dist, str):
            dist = sisl.get_distribution(dist, smearing=eta)
        else:
            warnings.warn("Using distribution created outside this function. The energy reference may be shifted if the distribution is calculated with respect to a non-zero energy value")

        # Obtain eigenvalues
        dos = 0
        for ispin in spin:
            eig = self.eigh(spin=ispin) - eref
            dos += sisl.electron.DOS(egrid, eig, distribution=dist)
        return dos

    def PDOS(self, egrid, eta=1e-3, spin=[0, 1], dist='Lorentzian', eref=0.):
        """ Obtains the projected density of states (PDOS) of the system with a distribution function

        Parameters
        ----------
        egrid: array_like
            Energy grid at which the DOS will be calculated.
        eta: float, optional
            Smearing parameter
        spin: int, optional
            If spin=0(1) it calculates the DOS for up (down) electrons in the system.
            If spin is not specified it returns DOS_up + DOS_dn.
        dist: str or sisl.distribution, optional
            distribution for the convolution, defaults to Lorentzian
        eref: float, optional
            energy reference, defaults to zero

        See Also
        ------------
        sisl.get_distribution: sisl method to create distribution function
        sisl.physics.electron.PDOS: sisl method to obtain PDOS

        Returns
        -------
        PDOS: numpy.ndarray
            projected density of states at the given energies for the selected spin
        """
        # Ensure spin is iterable
        if not isinstance(spin, (list, np.ndarray)):
            spin = [spin]

        # Check if egrid is numpy.ndarray
        if not isinstance(egrid, np.ndarray):
            egrid = np.array(egrid)

        if isinstance(dist, str):
            dist = sisl.get_distribution(dist, smearing=eta)
        else:
            warnings.warn("Using distribution created outside this function. The energy reference may be shifted if the distribution is calculated with respect to a non-zero energy value")

        # Obtain PDOS
        pdos = 0
        for ispin in spin:
            ev, evec = self.eigh(eigvals_only=False, spin=ispin)
            ev -= eref
            pdos += sisl.physics.electron.PDOS(egrid, ev, evec.T, distribution=dist)

        return pdos
