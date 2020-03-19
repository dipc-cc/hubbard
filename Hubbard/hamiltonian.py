from __future__ import print_function
import numpy as np
import sisl
import Hubbard.ncsile as nc
import hashlib
import os
import math
from scipy.linalg import inv

_pi = math.pi

__all__ = ['HubbardHamiltonian']


class HubbardHamiltonian(object):
    """ A class to create a Self Consistent Field (SCF) object related to the Mean Field Hubbard (MFH) model

    The `Hubbard.HubbardHamiltonian` class opens the possibility to include electron correlations in the tight-binding Hamiltonian
    by solving self-consistently the Mean Field Hubbard Hamiltonian

    It enables the convergence of several tight-binding described systems towards a user-defined tolerance criterion

    It takes an input tight-binding Hamiltonian and updates the corresponding matrix elements according to the MFH model

    Parameters
    ----------
    TBHam: sisl.Hamiltonian instance
        A spin-polarized tight-binding Hamiltonian
    DM: sisl.DensityMatrix instance, optional
        A spin-polarized density datrix generated with sisl
        to use as a initial spin-densities
    U: float, optional
        on-site Coulomb repulsion
    q: array_like, optional
        Two values specifying up, down electron occupations
    nkpt: array_like, optional
        Number of k-points along (a1, a2, a3) for Monkhorst-Pack BZ sampling
    kT: float, optional
        Temperature of the system in units of the Boltzmann constant

    See Also
    --------
    `sisl.physics.Hamiltonian <https://sisl.readthedocs.io/en/latest/api-generated/sisl.physics.Hamiltonian.html>`_
        sisl module to build sparse Hamiltonian matrices
    """

    def __init__(self, TBHam, DM=0, U=0.0, q=(0., 0.), nkpt=[1, 1, 1], kT=1e-5):
        """ Initialize HubbardHamiltonian """

        if not TBHam.spin.is_polarized:
            raise ValueError(self.__class__.__name__ + ' requires a spin-polarized system')

        # Use sum of all matrix elements as a basis for hash function calls
        H0 = TBHam.copy()
        H0.shift(np.pi) # Apply a shift to incorporate effect of S
        self.hash_base = H0.H.tocsr().sum()

        self.U = U # Hubbard onsite Coulomb parameter

        # Copy TB Hamiltonian to store the converged one in a different variable
        self.H = TBHam.copy()
        self.H.finalize()
        self.geom = TBHam.geometry

        # Total initial charge
        ntot = self.geom.q0
        if ntot == 0:
            ntot = len(self.geom)

        self.q = np.array(q, dtype=np.float64).copy()
        assert len(self.q) == 2 # Users *must* specify two values

        # Use default (low-spin) filling?
        if self.q[1] <= 0:
            self.q[1] = int(ntot / 2)
        if self.q[0] <= 0:
            self.q[0] = int(ntot - self.q[1])

        self.sites = len(self.geom)
        self._update_e0()
        # Generate Monkhorst-Pack
        self.mp = sisl.MonkhorstPack(self.H, nkpt)

        self.kT = kT

        # Initialize density matrix
        if not DM:
            self.DM = sisl.DensityMatrix(self.geom, dim=2, orthogonal=TBHam.orthogonal)
        else:
            self.DM = DM
        self.dm = self.DM._csr.diagonal().T

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
            for ``spin`` =0(1) it solves the eigenvalue problem for the spin up (down) Hamiltonian

        See Also
        --------
        `sisl.physics.SparseOrbitalBZ.eigh <https://sisl.readthedocs.io/en/latest/api-generated/sisl.physics.SparseOrbitalBZ.html#sisl.physics.SparseOrbitalBZ.eigh>`_

        Returns
        -------
        eigenvalues: numpy.ndarray
        """
        return self.H.eigh(k=k, eigvals_only=eigvals_only, spin=spin)

    def eigenstate(self, k, spin=0):
        """ Solve the eigenvalue problem at `k` and return it as a `sisl.EigenstateElectron` object containing all eigenstates

        Parameters
        ----------
        k: array_like
            k-point at which the eigenstate is going to be obtained
        spin: int, optional
            for spin=0(1) it solves the eigenvalue problem for the spin up (down) Hamiltonian

        See Also
        --------
        `sisl.physics.EigenstateElectron <https://sisl.readthedocs.io/en/latest/api-generated/sisl.physics.EigenstateElectron.html>`_

        Returns
        -------
        object: sisl.physics.EigenstateElectron object
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
        --------
        `sisl.Geometry.tile <https://sisl.readthedocs.io/en/latest/api-generated/sisl.Geometry.html#sisl.Geometry.tile>`_

        Returns
        -------
        A new larger HubbardHamiltonian object
        """
        Htile = self.H.tile(reps, axis)
        DMtile = self.DM.tile(reps, axis)
        Nup = (DMtile.tocsr(0).diagonal()).sum()
        Ndn = (DMtile.tocsr(1).diagonal()).sum()
        return self.__class__(Htile, DM=DMtile, U=self.U, q=(int(round(Nup)), int(round(Ndn))))

    def repeat(self, reps, axis):
        """ Repeat the HubbardHamiltonian object along a specified axis to obtain a larger one

        Parameters
        ----------
        reps: int
           number of repetitions
        axis: int
            direction of tiling, 0, 1, 2 according to the cell-direction

        See Also
        --------
        `sisl.Geometry.repeat <https://sisl.readthedocs.io/en/latest/api-generated/sisl.Geometry.html#sisl.Geometry.repeat>`_

        Returns
        -------
        A new larger HubbardHamiltonian object
        """
        Hrep = self.H.repeat(reps, axis)
        DMrep = self.DM.repeat(reps, axis)
        Nup = (DMrep.tocsr(0).diagonal()).sum()
        Ndn = (DMrep.tocsr(1).diagonal()).sum()
        return self.__class__(Hrep, DM=DMrep, U=self.U, q=(int(round(Nup)), int(round(Ndn))))

    def _update_e0(self):
        """ Internal routine to update e0 """
        e0 = self.H.tocsr(0).diagonal()
        e1 = self.H.tocsr(1).diagonal()
        self.e0 = np.array([e0, e1])

    def update_hamiltonian(self):
        """ Update spin Hamiltonian according to the mean field Hubbard model
        It updtates the diagonal elements for each spin Hamiltonian with the opposite spin densities

        Notes
        -----
        This method has to be generalized for inter-atomic Coulomb repulsion also
        """

        # TODO Generalize this method for inter-atomic Coulomb repulsion also

        q0 = self.geom.atoms.q0
        E = self.e0.copy()
        E += self.U * (self.dm[[1, 0], :] - q0)
        a = np.arange(len(self.H))
        self.H[a, a, [0, 1]] = E.T

    def update_density_matrix(self):
        """ Update the full density matrix

        Notes
        -----
        This method can be generalized to return the density matrix with off-diagonal elements
        i.e. for non-orthogonal basis, instead of the summed Mulliken populations (as in `Hubbard.dm`)
        """

        # TODO Generalize this method to return the density matrix with off-diagonal elements
        # for non-orthogonal LCAO basis
        a = np.arange(len(self.H))
        self.DM[a, a, [0, 1]] = self.dm.T

    def random_density(self):
        """ Initialize spin polarization  with random density """
        print('HubbardHamiltonian: Setting random density')
        self.dm = np.random.rand(2, self.sites)
        self.normalize_charge()
        self.update_density_matrix()

    def normalize_charge(self):
        """ Ensure the total up/down charge in pi-network equals Nup/Ndn """
        self.dm *= (self.q / self.dm.sum(1)).reshape(-1, 1)

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
        self.dm[:, up] = np.array([1., 0.]).reshape(2, 1)
        if len(dn) > 0:
            self.dm[:, dn] = np.array([0., 1.]).reshape(2, 1)
        self.normalize_charge()
        self.update_density_matrix()

    def polarize_sublattices(self):
        """ Quick way to polarize the lattices
        without checking that consequtive atoms actually belong to
        different sublattices """
        a = np.arange(len(self.H))
        self.dm[0, :] = a % 2
        self.dm[1, :] = 1 - self.dm[0, :]
        self.normalize_charge()
        self.update_density_matrix()

    def find_midgap(self):
        """ Find the midgap for the system
        taking into account the up and dn different spectrums

        This method makes sense for insulators (where there is a bandgap)
        """
        HOMO, LUMO = -1e10, 1e10
        for k in self.mp.k:
            ev_up = self.eigh(k=k, spin=0)
            ev_dn = self.eigh(k=k, spin=1)
            HOMO = max(HOMO, ev_up[int(round(self.q[0]-1))], ev_dn[int(round(self.q[1]-1))])
            LUMO = min(LUMO, ev_up[int(round(self.q[0]))], ev_dn[int(round(self.q[1]))])
        self.midgap = (HOMO + LUMO) * 0.5

    def fermi_level(self, q=None, dist='fermi_dirac'):
        """ Find the fermi level for  a certain charge `q` at a certain `kT`

        Parameters
        ----------
        q: array_like, optional
            charge per spin channel. First index for spin up, second index for dn
        dist: str, optional
            distribution

        See Also
        --------
        `sisl.physics.Hamiltonian.fermi_level <https://sisl.readthedocs.io/en/latest/api-generated/sisl.physics.Hamiltonian.html#sisl.physics.Hamiltonian.fermi_level>`_

        Returns
        -------
        Ef: numpy.array
            Fermi-level for each spin channel
        """
        Q = q
        for i in (0, 1):
            if Q[i] is None:
                Q[i] = self.q[i]
        dist = sisl.get_distribution(dist, smearing=self.kT)
        Ef = self.H.fermi_level(self.mp, q=Q, distribution=dist)
        return Ef

    def _get_hash(self):
        s = 'U=%.4f' % self.U
        s += ' N=(%.4f,%.4f)' % (self.q[0], self.q[1])
        s += ' base=%.3f' % self.hash_base
        return s, hashlib.md5(s.encode('utf-8')).hexdigest()[:7]

    def read_density(self, fn, mode='a'):
        """ Read density from binary file

        Parameters
        ----------
        fn: str
            name of the file that is going to read from
        mode: str, optional
            mode in which the file is opened
        """
        if os.path.isfile(fn):
            s, group = self._get_hash()
            fh = nc.ncSileHubbard(fn, mode=mode)
            if group in fh.groups:
                dm = fh.read_density(group)
                self.dm = dm
                self.update_density_matrix()
                self.update_hamiltonian()
                print('HubbardHamiltonian: Read charge from %s' % fn)
                return True
            else:
                print('HubbardHamiltonian: Density not found in %s[%s]' % (fn, group))
        return False

    def write_density(self, fn, mode='a'):
        """ Write density in a binary file

        Parameters
        ----------
        fn: str
            name of the file in which the densities are going to be stored
        mode: str, optional
            mode in which the file is opened
        """
        if not os.path.isfile(fn):
            mode = 'w'
        s, group = self._get_hash()
        fh = nc.ncSileHubbard(fn, mode=mode)
        fh.write_density(s, group, self.dm)
        print('HubbardHamiltonian: Wrote charge to %s' % fn)

    def iterate(self, dm_method, q=None, mixer=None, **kwargs):
        r""" Common method to iterate in a SCF loop that corresponds to the Mean Field Hubbard approximation

        The only thing that may change is the way in which the spin-densities (``dm``) and total energy (``Etot``) are obtained
        where one needs to use the correct ``dm_method`` for the particular system.

        The output densities are obtained using a mixing scheme with the previous iteration to help the convergence:

        .. math::

            \langle n_{\sigma}\rangle_{i} = mix \langle n_{\sigma}\rangle_{i} + (1-mix) \langle n_{\sigma}\rangle_{i-1}

        Parameters
        ----------
        dm_method: callable
            method to obtain the spin-densities
            it *must* return the corresponding spin-densities (``dm``) and the total energy (``Etot``)
        q: array_like, optional
            total charge separated in spin-channels, q=[q_up, q_dn]
        mixer: Mixer
            mixing object for the SCF loop

        See Also
        --------
        update_hamiltonian
        update_density_matrix
        Hubbard.dm
            method to obtain ``dm`` and ``Etot`` for tight-binding Hamiltonians with finite or periodic boundary conditions at a certain ``kT``
        Hubbard.NEGF.dm_open
            method to obtain  ``dm`` and ``Etot`` for tight-binding Hamiltonians with open boundary conditions

        Returns
        -------
        dn
            difference between the ith and the (i-1)th iteration densities
        """
        if q is None:
            q = self.q
        else:
            if q[0] is None:
                q[0] = int(round(self.q[0]))
            if q[1] is None:
                q[1] = int(round(self.q[1]))

        ni, Etot = dm_method(self, q, **kwargs)

        # Measure of density change
        ddm = ni - self.dm
        dn = np.absolute(ddm).max()

        # Update occupations on sites with mixing algorithm
        self.dm = mixer(self.dm, ddm)

        # Update density matrix
        self.update_density_matrix()

        # Update spin hamiltonian
        self.update_hamiltonian()

        # Store total energy
        self.Etot = Etot - self.U * np.multiply.reduce(self.dm, axis=0).sum()

        return dn

    def converge(self, dm_method, tol=1e-10, mixer=None, steps=100, fn=None, func_args=dict()):
        """ Iterate Hamiltonian towards a specified tolerance criterion

        This method calls `iterate` as many times as it needs until it reaches the specified tolerance

        Parameters
        ----------
        dm_method: callable
            method to obtain the spin-densities
            it *must* return the corresponding spin-densities (``dm``) and the total energy (``Etot``)
        tol: float, optional
            tolerance criterion
        mixer: Mixer
            mixing object (from `sisl.mixing.Mixer`), defaults to `sisl.mixing.DIISMixer`
        steps: int, optional
            the code will print some relevant information about the convergence process whent the number of completed iterations reaches
            a multiple of the specified `steps`
        fn: str, optional
            optionally, one can save the spin-densities during the calculation (when the number of completed iterations reaches
            the specified `steps`), by giving the name of the full name of the *binary file*
        func_args: dictionary, optional
            function arguments to pass to dm_method

        See Also
        --------
        iterate
        Hubbard.dm
            method to obtain ``dm`` and ``Etot`` for tight-binding Hamiltonians with finite or periodic boundary conditions at a certain ``kT``
        Hubbard.NEGF
            class that contains the routines to obtain  ``dm`` and ``Etot`` for tight-binding Hamiltonians with open boundary conditions

        Returns
        -------
        dn
            difference between the ith and the (i-1)th iteration densities
        """
        if mixer is None:
            mixer = sisl.mixing.LinearMixer(0.5)

        dn = 1.0
        i = 0
        while dn > tol:
            i += 1
            dn = self.iterate(dm_method, mixer=mixer, **func_args)
            # Print some info from time to time
            if i % steps == 0:
                print('   %i iterations completed:' % i, dn, self.Etot)
                if fn:
                    self.write_density(fn)

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
        ev: numpy.array
            eigenvalues
        L: numpy.ndarray
            orbital charge overlaps
        """
        ev, evec = self.eigh(k=k, eigvals_only=False, spin=spin)
        # Compute orbital charge overlaps
        L = np.einsum('ia,ia,ib,ib->ab', evec, evec, evec, evec).real
        return ev, L

    def get_Zak_phase(self, func=None, N=51, sub='filled', eigvals=False):
        """ Computes the Zak phase for 1D (periodic) systems using the sisl function sisl.electron.berry_phase

        Parameters
        ----------
        func: callable, optional
            function that creates a list of parametrized k-points to generate a new ``sisl.BrillouinZone`` object parametrized in ``N`` separations
        N: int, optional
            number of k-points generated using the parameterization
        sub: int, optional
            number of bands that will be summed to obtain the Zak phase

        Notes
        -----
        If no ``func`` is passed it assumes the periodicity along the x-axis
        If no ``sub`` is passed it sums up to the last occuppied band (included)

        See Also
        --------
        `sisl.electron.berry_phase <https://sisl.readthedocs.io/en/latest/api-generated/sisl.physics.electron.berry_phase.html>`_
            sisl routine to obtain the Berry phase

        Returns
        -------
        Zak: float
            Zak phase for the 1D system
        """

        if not func:
            # Discretize kx over [0.0, 1.0[ in Nx-1 segments (1BZ)
            def func(sc, frac):
                return [frac, 0, 0]
        bz = sisl.BrillouinZone.parametrize(self.H, func, N)
        if sub == 'filled':
            # Sum up over all occupied bands:
            sub = np.arange(int(round(self.q[0])))
        return sisl.electron.berry_phase(bz, sub=sub, eigvals=eigvals, method='zak')

    def get_bond_order(self, format='csr'):
        """ Compute Huckel bond order

        Parameters
        ----------
        format: {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the ``scipy.sparse.csr_matrix``,
           however if one always requires operations on dense matrices, one can always
           return in ``numpy.ndarray`` (`'array'`) or ``numpy.matrix`` (`'dense'`).

        Returns
        -------
        the Huckel bond-order matrix object
        """
        g = self.geom
        BO = sisl.Hamiltonian(g)
        R = [0.1, 1.6]
        for w, k in zip(self.mp.weight, self.mp.k):
            # spin-up first
            ev, evec = self.eigh(k=k, eigvals_only=False, spin=0)
            ev -= self.midgap
            idx = np.where(ev < 0.)[0]
            bo = np.dot(np.conj(evec[:, idx]), evec[:, idx].T)
            # add spin-down
            ev, evec = self.eigh(k=k, eigvals_only=False, spin=1)
            ev -= self.midgap
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
        --------
        `sisl.electron.spin_squared <https://sisl.readthedocs.io/en/latest/api-generated/sisl.physics.electron.spin_squared.html>`_
            sisl routine to obtain the spin squared expectation value between two spin states

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
        S = (Nalpha - Nbeta) * ((Nalpha - Nbeta)*.25 + 0.5)

        # Extract eigenvalues and eigenvectors of spin-up and spin-dn electrons
        ev_up, evec_up = self.eigh(eigvals_only=False, spin=0)
        ev_dn, evec_dn = self.eigh(eigvals_only=False, spin=1)

        # No need to tell which matrix of eigenstates correspond to alpha or beta,
        # the sisl function spin_squared already takes this into account
        s2alpha, s2beta = sisl.electron.spin_squared(evec_up[:, :int(round(self.q[0]))].T, evec_dn[:, :int(round(self.q[1]))].T)

        # Spin contamination
        S_MFH = S + Nbeta - s2beta.sum()

        if ret_exact:
            return S_MFH, S
        else:
            return S_MFH

    def band_sym(self, eigenstate, diag=True):
        '''
        Obtains the parity of vector(s) with respect to the rotation of its parent geometry by 180 degrees
        '''
        geom0 = self.geom
        geom180 = geom0.rotate(180, [0, 0, 1], geom0.center())
        sites180 = []
        for ia in geom180:
            for ib in geom0:
                if np.allclose(geom0.xyz[ib], geom180.xyz[ia]):
                    sites180.append(ib)
        if isinstance(eigenstate, sisl.physics.electron.EigenstateElectron):
            # In eigenstate instance dimensions are: (En, sites)
            v1 = np.conjugate(eigenstate.state)
            v2 = eigenstate.state[:, sites180]
        else:
            # Transpose to have dimensions (En, sites)
            if len(eigenstate.shape) == 1:
                eigenstate = eigenstate.reshape(1, eigenstate.shape[0])
            else:
                eigenstate = eigenstate.T
            v1 = np.conjugate(eigenstate)
            v2 = eigenstate[:, sites180]

        if diag:
            sym = (v1 * v2).sum(axis=1)
        else:
            sym = np.dot(v1, v2.T)
        return sym

    def DOS(self, egrid, eta=1e-3, spin=[0, 1]):
        """ Obtains the Density Of States of the system convoluted with a Lorentzian function

        Parameters
        ----------
        egrid: array_like
            Energy grid at which the DOS will be calculated.
        eta: float, optional
            Smearing parameter
        spin: int, optional
            If spin=0(1) it calculates the DOS for up (down) electrons in the system.
            If spin is not specified it returns DOS_up + DOS_dn.

        Returns
        -------
        DOS: array_like
            density of states at the given energies for the selected spin
        """

        # Check if egrid is numpy.ndarray
        if not isinstance(egrid, (np.ndarray)):
            egrid = np.array(egrid)

        # Obtain DOS
        DOS = 0
        for ispin in spin:
            ev, evec = self.eigh(eigvals_only=False, spin=ispin)
            ev -= self.midgap

            id1 = np.ones(ev.shape, np.float)
            id2 = np.ones(egrid.shape, np.float)
            dE = np.outer(id1, egrid) - np.outer(ev, id2)
            LOR = 2 * eta / (dE ** 2 + eta ** 2)
            DOS += np.einsum('ai,ai,ij->aj', evec, evec, LOR) / (2 * np.pi)

        return DOS
