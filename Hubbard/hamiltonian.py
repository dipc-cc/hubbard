"""

:mod:`Hubbard.hamiltonian`
==========================

Function for the meanfield Hubbard Hamiltonian

.. currentmodule:: Hubbard.hamiltonian

"""

from __future__ import print_function
import numpy as np
import sisl
import Hubbard.ncsile as nc
import hashlib
import os
import math
from scipy.linalg import inv

_pi = math.pi

class HubbardHamiltonian(object):
    """ sisl-type object

    Parameters:
    -----------
    TBHam : sisl.Hamiltonian instance
        A spin-polarized tight-Binding Hamiltonian
    U : float, optional
        on-site Coulomb repulsion
    Nup : int, optional
        Number of up-electrons
    Ndn : int, optional
        Number of down electrons
    nkpt : array_like, optional
        Number of k-points along (a1, a2, a3) for Monkhorst-Pack BZ sampling
    """

    def __init__(self, TBHam, DM=0, U=0.0, Nup=0, Ndn=0, nkpt=[1, 1, 1], kT=0, elecs=0, elec_indx=0, elec_dir=['-A', '+A'], CC=None, V=0):
        """ Initialize HubbardHamiltonian """
        
        if not TBHam.spin.is_polarized:
            raise ValueError(self.__class__.__name__ + ' requires as spin-polarized system')

        # Use sum of all matrix elements as a basis for hash function calls
        H0 = TBHam.copy()
        H0.shift(np.pi) # Apply a shift to incorporate effect of S
        self.hash_base = H0.H.tocsr().sum()

        self.U = U # Hubbard onsite Coulomb parameter

        # Copy TB Hamiltonian to store the converged one in a different variable
        self.H = TBHam.copy()
        self.geom = TBHam.geometry

        # Total initial charge
        ntot = self.geom.q0

        print('Neutral system corresponds to a total of %i electrons' %ntot)

        self.Nup = Nup # Total number of up-electrons
        self.Ndn = Ndn # Total number of down-electrons

        # Use default (low-spin) filling?
        if Ndn <= 0:
            self.Ndn = int(ntot/2)
        if Nup <= 0:
            self.Nup = int(ntot-self.Ndn)

        self.sites = len(self.geom)
        self._update_e0()
        # Generate Monkhorst-Pack
        self.mp = sisl.MonkhorstPack(self.H, nkpt)

        self.kT = kT

        # Initialize density matrix
        if not DM:
            self.DM = sisl.DensityMatrix(self.geom, dim=2)
            self.random_density()
        else:
            self.DM = DM
            self.nup = self.DM.tocsr(0).diagonal()
            self.ndn = self.DM.tocsr(1).diagonal()

        if elecs:
            self.Ef = np.zeros([2], np.float64)
            dist = sisl.get_distribution('fermi_dirac', smearing=kT)

            self.eta = 0.1

            if not CC:
                CC = os.path.split(__file__)[0]+'/EQCONTOUR'
            contour_weight = sisl.io.tableSile(CC).read_data()
            self.CC_eq = np.array([contour_weight[0] + contour_weight[1]*1j])
            self.w_eq = (contour_weight[2] + contour_weight[3]*1j) / np.pi
            self.NEQ = V != 0

            mu = [V*0.5, -V*0.5]
            if self.NEQ:
                self.CC_eq = np.array([self.CC_eq[0] + mu[0], self.CC_eq[0] + mu[1]])

                # Integration path for the non-Eq window
                dE = 0.01
                self.CC_neq = np.arange(min(mu)-5*kT, max(mu)+5*kT + dE, dE) + 1j * 0.001
                # Weights for the non-Eq integrals
                w_neq = dE * ( dist(self.CC_neq.real - mu[0]) - dist(self.CC_neq.real - mu[1]) )
                # Store weights for correction to RIGHT [0] and LEFT [1]
                self.w_neq = np.array([w_neq, -w_neq]) / np.pi

                # Initialize the neq-self-energies matrix
                # electrode, spin, energy
                self._cc_neq_SE = [[[None] * len(self.CC_neq)] * 2] * 2

            self.elec_indx = [np.array(idx).reshape(-1, 1) for idx in elec_indx]

            # electrode, spin
            self._ef_SE = [[None] * 2] * 2
            # electrode, spin, EQ-contour, energy
            self._cc_eq_SE = [[[[None] * self.CC_eq.shape[1]] * self.CC_eq.shape[0]] * 2] * 2

            for i, elec in enumerate(elecs):
                Ef_elec = elec.H.fermi_level(elec.mp, q=[elec.Nup, elec.Ndn], distribution=dist)
                # Shift each electrode with its Fermi-level
                # And also shift the chemical potential
                # Since the electrodes are *bulk* i.e. the entire electronic structure
                # is simply shifted we need to subtract since the contour shifts the chemical
                # potential back.
                elec.H.shift(-Ef_elec - mu[i])
                se = sisl.RecursiveSI(elec.H, elec_dir[i])
                for spin in [0,1]:
                    # Map self-energy at the Fermi-level of each electrode into the device region
                    self._ef_SE[i][spin] = se.self_energy(2 * mu[i] + 1j * self.eta, spin=spin)

                    for cc_eq_i, CC_eq in enumerate(self.CC_eq):
                        for ic, cc in enumerate(CC_eq):
                            # Do it also for each point in the CC, for all EQ CC
                            self._cc_eq_SE[i][spin][cc_eq_i][ic] = se.self_energy(cc, spin=spin)
                    if self.NEQ:
                        for ic, cc in enumerate(self.CC_neq):
                            # And for each point in the Neq CC
                            self._cc_neq_SE[i][spin][ic] = se.self_energy(cc, spin=spin)

    def eigh(self, k=[0, 0, 0], eigvals_only=True, spin=0):
        return self.H.eigh(k=k, eigvals_only=eigvals_only, spin=spin)
    
    def eigenstate(self, k, spin=0):
        return self.H.eigenstate(k, spin=spin)

    def tile(self, reps, axis):
        Htile = self.H.tile(reps, axis)
        DMtile = self.DM.tile(reps, axis)
        Nup = (DMtile.tocsr(0).diagonal()).sum()
        Ndn = (DMtile.tocsr(1).diagonal()).sum()
        return self.__class__(Htile, DM=DMtile, U=self.U, Nup=int(round(Nup)), Ndn=int(round(Ndn)))

    def repeat(self, reps, axis):
        Hrep = self.H.repeat(reps, axis)
        DMrep = self.DM.repeat(reps, axis)
        Nup = (DMrep.tocsr(0).diagonal()).sum()
        Ndn = (DMrep.tocsr(1).diagonal()).sum()
        return self.__class__(Hrep, DM=DMrep, U=self.U, Nup=int(round(Nup)), Ndn=int(round(Ndn)))

    def _update_e0(self):
        """Internal routine to update e0 """
        e0 = self.H.tocsr(0).diagonal()
        e1 = self.H.tocsr(1).diagonal()
        self.e0 = np.array([e0, e1])

    def update_hamiltonian(self):
        # Update spin Hamiltonian
        q0 = self.geom.atoms.q0
        E = self.e0.copy()
        E[0, :] += self.U * (self.ndn - q0)
        E[1, :] += self.U * (self.nup - q0)
        a = np.arange(len(self.H))
        self.H[a, a, [0, 1]] = E.T

    def update_density_matrix(self):
        a = np.arange(len(self.H))
        self.DM[a, a, [0, 1]] = np.array([self.nup, self.ndn]).T

    def random_density(self):
        """ Initialize spin polarization  with random density """
        print('Setting random density')
        self.nup = np.random.rand(self.sites)
        self.ndn = np.random.rand(self.sites)
        self.normalize_charge()
        self.update_density_matrix()

    def normalize_charge(self):
        """ Ensure the total up/down charge in pi-network equals Nup/Ndn """
        self.nup = self.nup / self.nup.sum() * self.Nup
        self.ndn = self.ndn / self.ndn.sum() * self.Ndn
        print('Normalized charge distributions to Nup=%i, Ndn=%i' % (self.Nup, self.Ndn))

    def set_polarization(self, up, dn=[]):
        """ Maximize spin polarization on specific atomic sites.
        Optionally, sites with down-polarization can be specified

        Parameters
        ----------
        up : array_like
            atomic sites where the spin-up density is going to be maximized
        dn : array_like, optional
            atomic sites where the spin-down density is going to be maximized
        """
        print('Setting up-polarization for sites', up)
        self.nup[up] = 1.
        self.ndn[up] = 0.
        if len(dn) > 0:
            print('Setting down-polarization for sites', dn)
            self.nup[dn] = 0.
            self.ndn[dn] = 1.
        self.normalize_charge()
        self.update_density_matrix()

    def polarize_sublattices(self):
        # This is just a quick way to polarize the lattice
        # without checking that consequtive atoms actually belong to
        # different sublattices
        for i in range(len(self.nup)):
            self.nup[i] = i%2
            self.ndn[i] = 1-i%2
        self.normalize_charge()
        self.update_density_matrix()

    def find_midgap(self):
        HOMO, LUMO = -1e10, 1e10
        for k in self.mp.k:
            ev_up = self.eigh(k=k, spin=0)
            ev_dn = self.eigh(k=k, spin=1)
            HOMO = max(HOMO, ev_up[self.Nup-1], ev_dn[self.Ndn-1])
            LUMO = min(LUMO, ev_up[self.Nup], ev_dn[self.Ndn])
        self.midgap = (HOMO + LUMO) * 0.5

    def _get_hash(self):
        s = 'U=%.4f' % self.U
        s += ' N=(%i,%i)' % (self.Nup, self.Ndn)
        s += ' base=%.3f' % self.hash_base
        return s, hashlib.md5(s.encode('utf-8')).hexdigest()[:7]

    def read_density(self, fn, mode='a'):
        if os.path.isfile(fn):
            s, group = self._get_hash()
            fh = nc.ncSileHubbard(fn, mode=mode)
            if group in fh.groups:
                nup, ndn = fh.read_density(group)
                self.nup = nup
                self.ndn = ndn
                self.update_density_matrix()
                self.update_hamiltonian()
                print('Read charge from %s' % fn)
                return True
            else:
                print('Density not found in %s[%s]' % (fn, group))
        return False

    def write_density(self, fn, mode='a'):
        if not os.path.isfile(fn):
            mode='w'
        s, group = self._get_hash()
        fh = nc.ncSileHubbard(fn, mode=mode)
        fh.write_density(s, group, self.nup, self.ndn)
        print('Wrote charge to %s' % fn)

    def iterate(self, mix=1.0):
        # Create short-hands
        nup = self.nup
        ndn = self.ndn
        Nup = int(round(self.Nup))
        Ndn = int(round(self.Ndn))

        # Initialize HOMO/LUMO variables
        HOMO = -1e10
        LUMO = 1e10

        # Initialize new occupations and total energy with Hubbard U
        ni_up = np.zeros(nup.shape)
        ni_dn = np.zeros(ndn.shape)
        Etot = 0.

        # Solve eigenvalue problems
        def calc_occ(k, weight, HOMO, LUMO):
            """ My wrap function for calculating occupations """
            es_up = self.eigenstate(k, spin=0)
            es_dn = self.eigenstate(k, spin=1)

            # Update HOMO, LUMO
            HOMO = max(HOMO, es_up.eig[Nup-1], es_dn.eig[Ndn-1])
            LUMO = min(LUMO, es_up.eig[Nup], es_dn.eig[Ndn])
            
            es_up = es_up.sub(range(Nup))
            es_dn = es_dn.sub(range(Ndn))

            ni_up = (es_up.norm2(False).real).sum(0) * weight
            ni_dn = (es_dn.norm2(False).real).sum(0) * weight

            # Calculate total energy
            Etot = (es_up.eig.sum() + es_dn.eig.sum()) * weight
            # Return values
            return HOMO, LUMO, ni_up, ni_dn, Etot

        # Loop k-points and weights
        for w, k in zip(self.mp.weight, self.mp.k):
            HOMO, LUMO, up, dn, etot = calc_occ(k, w, HOMO, LUMO)
            ni_up += up
            ni_dn += dn
            Etot += etot

        # Determine midgap energy reference
        self.midgap = (LUMO + HOMO) / 2
        
        # Measure of density change
        dn = (np.absolute(nup - ni_up) + np.absolute(ndn - ni_dn)).sum()

        # Update occupations on sites with mixing
        self.nup = mix * ni_up + (1. - mix) * nup
        self.ndn = mix * ni_dn + (1. - mix) * ndn
        
        # Update density matrix
        self.update_density_matrix()

        # Update spin hamiltonian
        self.update_hamiltonian()

        # Store total energy
        self.Etot = Etot - self.U * (self.nup * self.ndn).sum()

        return dn

    def iterate2(self, mix=1.0, q_up=None, q_dn=None):
        # Create short-hands
        nup = self.nup
        ndn = self.ndn
        if q_up is None:
            q_up = self.Nup
        if q_dn is None:
            q_dn = self.Ndn

        # Create fermi-level determination distribution
        dist = sisl.get_distribution('fermi_dirac', smearing=self.kT)
        Ef = self.H.fermi_level(self.mp, q=[q_up, q_dn], distribution=dist)
        dist_up = sisl.get_distribution('fermi_dirac', smearing=self.kT, x0=Ef[0])
        dist_dn = sisl.get_distribution('fermi_dirac', smearing=self.kT, x0=Ef[1])

        # Initialize new occupations and total energy with Hubbard U
        ni_up = np.zeros(nup.shape)
        ni_dn = np.zeros(ndn.shape)
        Etot = 0.

        # Solve eigenvalue problems
        def calc_occ(k, weight):
            """ My wrap function for calculating occupations """
            es_up = self.eigenstate(k, spin=0)
            es_dn = self.eigenstate(k, spin=1)

            # Reduce to occupied stuff
            occ_up = es_up.occupation(dist_up).reshape(-1, 1) * weight
            ni_up = (es_up.norm2(False).real * occ_up).sum(0)
            occ_dn = es_dn.occupation(dist_dn).reshape(-1, 1) * weight
            ni_dn = (es_dn.norm2(False).real * occ_dn).sum(0)
            Etot = (es_up.eig * occ_up.ravel()).sum() + (es_dn.eig * occ_dn.ravel()).sum()

            # Return values
            return ni_up, ni_dn, Etot

        # Loop k-points and weights
        for w, k in zip(self.mp.weight, self.mp.k):
            up, dn, etot = calc_occ(k, w)
            ni_up += up
            ni_dn += dn
            Etot += etot

        # Determine midgap energy reference (or simply the fermi-level)
        self.midgap = Ef.sum() / 2
        
        # Measure of density change
        dn = (np.absolute(nup - ni_up) + np.absolute(ndn - ni_dn)).sum()

        # Update occupations on sites with mixing
        self.nup = mix * ni_up + (1. - mix) * nup
        self.ndn = mix * ni_dn + (1. - mix) * ndn
        
        # Update density matrix
        self.update_density_matrix()

        # Update spin hamiltonian
        self.update_hamiltonian()

        # Store total energy
        self.Etot = Etot - self.U * (self.nup * self.ndn).sum()

        return dn

    def iterate3(self, mix=1.0, q_up=None, q_dn=None, qtol=1e-5):
        """
        Iterative method for solving open systems self-consistently
        It computes the spin densities from the Neq Green's function
        So far it is implemented for an equilibrium calculation (it reads the equilibrium contour)

        Parameters
        ----------
        elecs : list of HubbardHamiltonian instances
            list of (already converged) electrode Hamiltonians present in the system
        elec_indx : list, numpy array
            list of the atomic indices of the electrodes in the device region

        """

        nup = self.nup
        ndn = self.ndn

        if q_up is None:
            q_up = self.Nup
        if q_dn is None:
            q_dn = self.Ndn

        no = len(self.H)
        inv_GF = np.empty([no, no], dtype=np.complex128)
        ni = np.empty([2, no], dtype=np.float64)
        ntot = -1.
        Ef = -self.Ef.copy()
        while abs(ntot - q_up - q_dn) > qtol:

            if ntot > 0.:
                # correct fermi-level
                dq = np.empty([2])
                dq[0] = ni[0].sum() - q_up
                dq[1] = ni[1].sum() - q_dn

                # Fermi-level is at 0.
                # The Lorentzian has ~70% of its integral within
                #   2 * \eta (on both sides)
                # To cover 200 meV ~ 2400 K in the integration window
                # and expect the Lorentzian peak to be positioned at
                # the current Fermi-level we will use eta = 100 meV
                # Calculate charge at the Fermi-level
                for spin in [0, 1]:
                    HC = self.H.Hk(spin=spin).todense()
                    cc = - Ef[spin] + 1j * self.eta

                    inv_GF[:, :] = 0.
                    np.fill_diagonal(inv_GF, cc)
                    inv_GF[:, :] -= HC[:, :]
                    for i, SE in enumerate(self._ef_SE):
                        inv_GF[self.elec_indx[i], self.elec_indx[i].T] -= SE[spin]

                    # Now we need to calculate the new Fermi level based on the
                    # difference in charge and by estimating the current Fermi level
                    # as the peak position of a Lorentzian.
                    # I.e. F(x) = \int_-\infty^x L(x) dx = arctan(x) / pi + 0.5
                    #   F(x) - F(0) = arctan(x) / pi = dq
                    # In our case we *know* that 0.5 = - Im[Tr(Gf)] / \pi
                    # and consider this a pre-factor
                    f = dq[spin] / (- np.trace(inv(inv_GF)).imag / _pi)
                    # Since x above is in units of eta, we have to multiply with eta
                    if abs(f) < 0.45:
                        Ef[spin] += self.eta * math.tan(f * _pi)
                    else:
                        Ef[spin] += self.eta * math.tan((_pi / 2 - math.atan(1 / (f * _pi))))

            Etot = 0.
            for spin in [0, 1]:
                HC = self.H.Hk(spin=spin, format='array')
                D = np.zeros((len(self.CC_eq),)+HC.shape)
                ni[spin, :] = 0.
                if self.NEQ:
                    # Correct Density matrix with Non-equilibrium integrals
                    Delta, w = self.Delta(HC, Ef[spin], spin=spin)
                    # Transfer Delta to D
                    D[0, :, :] = Delta[1]
                    D[1, :, :] = Delta[0]
                    # TODO We need to also calculate the total energy for NEQ
                    #      this should probably be done in the Delta method
                    del Delta
                else:
                    # This ensures we can calculate for EQ only calculations
                    w = 1.

                # Loop over all eq. Contours
                for cc_eq_i, CC in enumerate(self.CC_eq):
                    for ic, [cc, wi] in enumerate(zip(CC - Ef[spin], self.w_eq)):
                        inv_GF[:, :] = 0.
                        np.fill_diagonal(inv_GF, cc)
                        inv_GF[:, :] -= HC[:, :]
                        for i, SE in enumerate(self._cc_eq_SE):
                            inv_GF[self.elec_indx[i], self.elec_indx[i].T] -= SE[spin][cc_eq_i][ic]

                        # Greens function evaluated at each point of the CC multiplied by the weight
                        Gf_wi = - inv(inv_GF) * wi
                        D[cc_eq_i] += Gf_wi.imag

                        # Integrate density of states to obtain the total energy
                        # For the non equilibrium energy maybe we could obtain it as in PRL 70, 14 (1993)
                        if cc_eq_i == 0:
                            Etot += (np.diag(w*Gf_wi).sum() * cc).imag
                        else:
                            Etot += (np.diag((1-w)*Gf_wi).sum() * cc).imag

                if self.NEQ:
                    D = w * D[0] + (1-w) * D[1]
                else:
                    D = D[0]

                ni[spin, :] = np.diag(D)

            # Calculate new charge
            ntot = ni.sum()

        # Save Fermi-level of the device
        self.Ef = -Ef

        # Measure of density change
        dn = (np.absolute(nup - ni[0]) + np.absolute(ndn - ni[1])).sum()

        # Update occupations on sites with mixing
        self.nup = mix * ni[0] + (1. - mix) * nup
        self.ndn = mix * ni[1] + (1. - mix) * ndn

        # Update spin Hamiltonians
        self.update_hamiltonian()

        self.Etot = Etot - self.U * (self.nup*self.ndn).sum()

        return dn

    def Delta(self, HC, Ef, spin=0):

        def spectral(G, self_energy, elec):
            # Use self-energy of elec, now the matrix will have dimension (Nelec, Nelec)
            Gamma = 1j*(self_energy - np.conjugate(self_energy.T))
            G = G[:, self.elec_indx[elec].T[0]]
            # Product of (Ndev, Nelec) x (Nelec, Nelec) x (Nelec, Ndev)
            return np.dot(G, np.dot(Gamma, np.conjugate(G.T)))

        no = len(HC)
        Delta = np.zeros([2, no, no], dtype=np.complex128)
        inv_GF = np.empty([no, no], dtype=np.complex128)

        for ic, cc in enumerate(self.CC_neq - Ef):
            inv_GF[:, :] = 0.
            np.fill_diagonal(inv_GF, cc)
            inv_GF[:, :] -= HC[:, :]
            for i, SE in enumerate(self._cc_neq_SE):
                inv_GF[self.elec_indx[i], self.elec_indx[i].T] -= SE[spin][ic]
            # Calculate the Green function once
            inv_GF[:, :] = inv(inv_GF)
            # Elec (0, 1) are (left, right)
            for i, SE in enumerate(self._cc_neq_SE):
                Delta[i] += spectral(inv_GF, SE[spin][ic], i) * self.w_neq[i, ic]

        # Firstly implement it for two terminals following PRB 65 165401 (2002)
        # then we can think of implementing it for N terminals as in Com. Phys. Comm. 212 8-24 (2017)
        weight = Delta[0]**2 / ((Delta**2).sum(axis=0))

        # Get rid of the numerical imaginary part (which is ~0)
        return Delta.real, weight.real

    def converge(self, tol=1e-10, steps=100, mix=1.0, premix=0.1, method=0, fn=None, func_args=dict()):
        """ Iterate Hamiltonian towards a specified tolerance criterion """
        print('Iterating towards self-consistency...')
        if method == 2:
            iterate_ = self.iterate2
            # Use finite T close to zero
            if self.kT == 0:
                self.kT = 0.00001
        elif method == 3:
            iterate_ = self.iterate3
        else:
            if self.kT == 0:
                iterate_ = self.iterate
            else:
                iterate_ = self.iterate2
        dn = 1.0
        i = 0
        while dn > tol:
            i += 1
            if dn > 0.1:
                # precondition when density change is relatively large
                dn = iterate_(mix=premix, **func_args)
            else:
                dn = iterate_(mix=mix, **func_args)
            # Print some info from time to time
            if i%steps == 0:
                print('   %i iterations completed:'%i, dn, self.Etot)
                if fn:
                    self.write_density(fn)

        print('   found solution in %i iterations'%i)
        return dn

    def calc_orbital_charge_overlaps(self, k=[0, 0, 0], spin=0):
        ev, evec = self.eigh(k=k, eigvals_only=False, spin=spin)
        # Compute orbital charge overlaps
        L = np.einsum('ia,ia,ib,ib->ab', evec, evec, evec, evec).real
        return ev, L


    def get_Zak_phase(self, Nx=51, sub='filled', eigvals=False):
        """ Compute Zak phase for 1D systems oriented along the x-axis.
        Keep in mind that the current implementation does not handle correctly band intersections.
        Meaningful Zak phases can thus only be computed for the non-crossing bands.
        """
        # Discretize kx over [0.0, 1.0[ in Nx-1 segments (1BZ)
        def func(sc, frac):
            return [frac, 0, 0]
        bz = sisl.BrillouinZone.parametrize(self.H, func, Nx)
        if sub == 'filled':
            # Sum up over all occupied bands:
            sub = range(self.Nup)
        return sisl.electron.berry_phase(bz, sub=sub, eigvals=eigvals, method='zak')


    def get_bond_order(self, format='csr'):
        """ Compute Huckel bond order

        Parameters
        ----------
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the ``scipy.sparse.csr_matrix``,
           however if one always requires operations on dense matrices, one can always
           return in `numpy.ndarray` (`'array'`) or `numpy.matrix` (`'dense'`).

        Returns
        -------
        object : the Huckel bond-order matrix
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
                        phase = np.exp(-2.j*np.pi*np.dot(k, r))
                        for ia in g:
                            for ja in g.close_sc(ia, R=R, isc=r)[1]:
                                bor = bo[ia, ja]*phase
                                BO[ia, ja] += w*bor.real
        # Add sigma bond at the end
        for ia in g:
            idx = g.close(ia, R=R)
            BO[ia, idx[1]] += 1.
        return BO.Hk(format=format) # Fold to Gamma

    def spin_contamination(self):
        """
        Obtains the spin contamination after the MFH calculation
        Ref. Chemical Physics Letters. 183 (5): 423–431.
        
        This function works for non-periodic systems only.
        """
        # Define Nalpha and Nbeta, where Nalpha >= Nbeta 
        Nalpha = max(self.Nup, self.Ndn)
        Nbeta = min(self.Nup, self.Ndn)
        
        # Exact Total Spin expected value (< S² >)
        S = .5*(Nalpha - Nbeta) * ( (Nalpha - Nbeta)*.5 + 1)

        # Extract eigenvalues and eigenvectors of spin-up and spin-dn electrons
        ev_up, evec_up = self.eigh(eigvals_only=False, spin=0)
        ev_dn, evec_dn = self.eigh(eigvals_only=False, spin=1)

        # No need to tell which matrix of eigenstates correspond to alpha or beta, 
        # the sisl function spin_squared already takes this into account
        s2alpha, s2beta = sisl.electron.spin_squared(evec_up[:, :self.Nup].T, evec_dn[:, :self.Ndn].T)
        
        # Spin contamination 
        S_MFH = S + Nbeta - s2beta.sum()

        return S, S_MFH

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

    def DOS(self, egrid, eta=1e-3, spin=[0,1]):
        """
        Obtains the Density Of States of the system convoluted with a Lorentzian function

        Parameters
        ----------
        egrid: float or array_like
            Energy grid at which the DOS will be calculated.
        eta: float
            Smearing parameter
        spin: integer or list of integers
            If spin=0(1) it calculates the DOS for up (down) electrons in the system.
            If spin is not specified it returns DOS_up + DOS_dn.
        """

        # Check if egrid is numpy.ndarray
        if not isinstance(egrid, (np.ndarray)):
            egrid = np.array(egrid)
        
        # Obtain DOS
        DOS = 0
        for ispin in spin:
            ev, evec = self.eigh(eigvals_only=False, spin=ispin)
            ev -= self.midgap

            id1 = np.ones(ev.shape,np.float)
            id2 = np.ones(egrid.shape,np.float)
            dE = np.outer(id1,egrid)-np.outer(ev,id2)
            LOR = 2*eta/(dE**2+eta**2)
            DOS += np.einsum('ai,ai,ij->aj',evec,evec,LOR)/(2*np.pi)
        
        return DOS
