import numpy as np
from numpy import einsum, conj
import sisl
import os
import math
from scipy.interpolate import interp1d
from scipy.linalg import inv

_pi = math.pi

__all__ = ['NEGF']


def _G(e, HC, elec_idx, SE):
    """ Calculate Green function

    Parameters
    ----------
    e : complex
       energy at which the Green function is calculated
    HC : numpy.ndarray
       the Hamiltonian
    elec_idx : list of indices
       indices for the electrode orbitals
    SE : list of numpy.ndarray
       self-energies for the electrodes
    """
    no = len(HC)
    inv_GF = np.zeros([no, no], dtype=np.complex128)
    np.fill_diagonal(inv_GF, e)
    inv_GF[:, :] -= HC[:, :]
    for idx, se in zip(elec_idx, SE):
        inv_GF[idx, idx.T] -= se
    return inv(inv_GF)


def _nested_list(*args):
    if len(args) == 0:
        return None
    l = []
    for _ in range(args[0]):
        l.append(_nested_list(*args[1:]))
    return l


class NEGF:
    r""" This class creates the open quantum system object for a N-terminal device

    For the Non-equilibrium case :math:`V\neq 0` the current implementation
    only can deal with a 2-terminal device

    Parameters
    ----------
    Hdev: HubbardHamiltonian
        `hubbard.HubbardHamiltonian` object of the device
    elec_SE: list of sisl.SelfEnergy or tuple of (HubbardHamiltonian, str)
        list of (already converged) `hubbard.HubbardHamiltonian` objects for the electrodes
        plus the semi-infinite direction for the respective electrode.
        Alternatively one may directly pass `sisl.SelfEnergy` instances
    elec_idx: array_like
        list of atomic positions that *each* electrode occupies in the device geometry
    CC: str, optional
        name of the file containing the energy contour in the complex plane to integrate the density matrix
    V: float, optional
        applied bias between the two electrodes


    Examples
    --------
    >>> NEGF(Hdev, [sisl.SelfEnergy(), (H, '+A')])


    See Also
    ------------
    sisl.physics.RecursiveSI: sisl routines to create semi-infinite object (obtain self-energy, etc.)

    Notes
    -----
    This class has to be generalized to non-orthogonal basis
    """

    def __init__(self, Hdev, elec_SE, elec_idx, CC=None, V=0, **kwargs):
        """ Initialize NEGF class """

        # Global charge neutral reference energy (conveniently named fermi)
        self.Ef = 0.
        self.kT = Hdev.kT
        self.eta = 0.1

        # Immediately retrieve the distribution
        dist = sisl.get_distribution('fermi_dirac', smearing=self.kT)

        if not CC:
            CC = os.path.split(__file__)[0] + "/EQCONTOUR"

        # Extract weights and energy points from CC
        contour_weight = sisl.io.tableSile(CC).read_data()
        self.CC_eq = np.array([contour_weight[0] + 1j * contour_weight[1]])
        self.w_eq = (contour_weight[2] + 1j * contour_weight[3]) / np.pi
        self.NEQ = V != 0

        self.mu = np.zeros(len(elec_SE))
        if self.NEQ:
            # in case the user has WBL electrodes
            self.mu[0] = V * 0.5
            self.mu[1] = -V * 0.5
            # Define equilibrium contours
            self.CC_eq = np.array([self.CC_eq[0] + self.mu[0], self.CC_eq[0] + self.mu[1]])

            # Integration path for the non-Eq window
            dE = 0.01
            self.CC_neq = np.arange(min(self.mu) - 5 * self.kT, max(self.mu) + 5 * self.kT + dE, dE) + 1j * 0.001
            # Weights for the non-Eq integrals
            w_neq = dE * (dist(self.CC_neq.real - self.mu[0]) - dist(self.CC_neq.real - self.mu[1]))
            # Store weights for calculation of DELTA_L [0] and DELTA_R [1]
            self.w_neq = np.array([w_neq, -w_neq]) / _pi
        else:
            self.CC_neq = np.array([])

        def convert2SelfEnergy(elec_SE, mu):
            if isinstance(elec_SE, (tuple, list)):
                # here HH *must* be a HubbardHamiltonian (otherwise it will probably crash)
                HH, semi_inf = elec_SE
                Ef_elec = HH.fermi_level(q=HH.q, distribution=dist)
                # Shift each electrode with its Fermi-level and chemical potential
                # Since the electrodes are *bulk* i.e. the entire electronic structure
                # is simply shifted
                HH.shift(-Ef_elec + mu)
                return sisl.RecursiveSI(HH.H, semi_inf)

            # If elec_SE is already a SelfEnergy instance
            # this will work since SelfEnergy instances overloads unknown attributes
            # to the parent.
            # This will only shift the electronic structure of the RecursiveSI.spgeom0
            # and not RecursiveSI.spgeom1. However, for orthogonal basis, this is equivalent.
            # TODO this should be changed when non-orthogonal basis' are used
            try:
                elec_SE.shift(mu)
            except:
                # the parent does not have the shift method
                pass
            return elec_SE

        # convert all matrices to a sisl.SelfEnergy instance
        self.elec_SE = list(map(convert2SelfEnergy, elec_SE, self.mu))
        self.elec_idx = [np.array(idx).reshape(-1, 1) for idx in elec_idx]

        # Ensure commensurate shapes
        for SE, idx in zip(self.elec_SE, self.elec_idx):
            assert len(SE) == len(idx)

        # For a bias calcualtion, ensure that only the first
        # two electrodes are RecursiveSI (all others should be WideBandSE)
        if self.NEQ:
            for i in range(2):
                assert isinstance(self.elec_SE[i], sisl.RecursiveSI)
            for i in range(2, len(self.elec_SE)):
                assert isinstance(self.elec_SE[i], sisl.WideBandSE)

        # In case all self-energies are WB, then we can change the eta value
        if all(map(lambda obj: isinstance(obj, sisl.WideBandSE), self.elec_SE)):
            # The wide-band limit ensures that all electrons comes at a constant rate per
            # energy.
            # Although this *could* potentially be too high, then I think it should be ok
            # since the electrodes more govern the DOS.
            self.eta = 1.

        # spin, k-sampling, electrode
        self._ef_SE = _nested_list(Hdev.spin_size, len(Hdev.mp.k), len(self.elec_SE))
        # spin, k-sampling, EQ-contour, energy, electrode
        self._cc_eq_SE = _nested_list(Hdev.spin_size, len(Hdev.mp.k), *self.CC_eq.shape, len(self.elec_SE))
        # spin, k-sampling, energy, electrode
        self._cc_neq_SE = _nested_list(Hdev.spin_size, len(Hdev.mp.k), self.CC_neq.shape[0], len(self.elec_SE))

        kw = {}
        for i, se in enumerate(self.elec_SE):
            for ik, k in enumerate(Hdev.mp.k):
                for spin in range(Hdev.spin_size):
                    # Map self-energy at the Fermi-level of each electrode into the device region
                    if Hdev.spin_size > 1:
                        kw = {'spin':spin}

                    self._ef_SE[spin][ik][i] = se.self_energy(1j * self.eta, k=k, **kw)

                    for cc_eq_i, CC_eq in enumerate(self.CC_eq):
                        for ic, cc in enumerate(CC_eq):
                            # Do it also for each point in the CC, for all EQ CC
                            self._cc_eq_SE[spin][ik][cc_eq_i][ic][i] = se.self_energy(cc, k=k, **kw)

                    if self.NEQ:
                        for ic, cc in enumerate(self.CC_neq):
                            # And for each point in the Neq CC
                            self._cc_neq_SE[spin][ik][ic][i] = se.self_energy(cc, k=k, **kw)

    def calc_n_open(self, H, q, qtol=1e-5):
        """
        Method to compute the spin densities from the non-equilibrium Green's function

        Parameters
        ----------
        H: HubbardHamiltonian
            `hubbard.HubbardHamiltonian` of the object that is being iterated
        q: float
            charge associated to the up and down spin-components
        qtol: float, optional
            tolerance to which the charge is going to be converged in the internal loop
            that finds the potential of the device (i.e. that makes the device neutrally charged)

        Returns
        -------
        ni: numpy.ndarray
            spin densities
        Etot: float
            total energy
        """
        # ensure scalar, for open systems one cannot impose a spin-charge
        # This spin-charge would be dependent on the system size
        q = np.asarray(q).sum()

        # Create short-hands
        ef_SE = self._ef_SE
        cc_eq_SE = self._cc_eq_SE

        no = len(H.H)
        ni = np.empty([H.spin_size, no], dtype=np.float64)
        ntot = -1.
        Ef = self.Ef

        conv_q = []
        while abs(ntot - q) > qtol:
            # for debug purposes
            #print(ntot, q, qtol, Ef)

            if ntot > 0.:
                # correct fermi-level
                dq = ni.sum() - q

                conv_q.append([dq, Ef])
                if len(conv_q) > 1:
                    # do a bisection of the fermi level
                    f = np.array(conv_q)
                    try:
                        Ef = interp1d(f[:, 0], f[:, 1],
                                          fill_value="extrapolate")(0.)
                    except:
                        Ef = conv_q[-1][1]
                    if np.isnan(Ef):
                        Ef = conv_q[-1][1]

                    # check that the fermi-level is not one we already have
                    if np.any(np.fabs(Ef - f[:, 1]) < 1e-15):
                        conv_q = [conv_q[-1]]

                if len(conv_q) == 1:

                    # Fermi-level is at 0.
                    # The Lorentzian has ~70% of its integral within
                    #   2 * \eta (on both sides)
                    # To cover 200 meV ~ 2400 K in the integration window
                    # and expect the Lorentzian peak to be positioned at
                    # the current Fermi-level we will use eta = 100 meV
                    # Calculate charge at the Fermi-level
                    f = 0.
                    for spin in range(H.spin_size):
                        for ik, k in enumerate(H.mp.k):
                            if H.spin_size == 2:
                                HC = H.H.Hk(spin=spin, k=k, format='array')
                            else:
                                HC = H.H.Hk(k=k, format='array')
                            cc = Ef + 1j * self.eta

                            GF = _G(cc, HC, self.elec_idx, ef_SE[spin][ik])

                            # Now we need to calculate the new Fermi level based on the
                            # difference in charge and by estimating the current Fermi level
                            # as the peak position of a Lorentzian.
                            # I.e. F(x) = \int_-\infty^x L(x) dx = arctan(x) / pi + 0.5
                            #   F(x) - F(0) = arctan(x) / pi = dq
                            # In our case we *know* that 0.5 = - Im[Tr(Gf)] / \pi
                            # and consider this a pre-factor
                            f -= np.trace(GF).imag * wk / _pi # Integrate over k-space with weight wk

                        # calculate fractional change
                        f = dq / f
                        # Since x above is in units of eta, we have to multiply with eta
                        if abs(f) < 0.45:
                            Ef -= self.eta * math.tan(f * _pi) * 0.5
                        else:
                            Ef -= self.eta * math.tan((_pi / 2 - math.atan(1 / (f * _pi)))) * 0.5

            Etot = 0.
            for spin in range(H.spin_size):
                # Loop k-points and weights
                D = np.zeros([len(self.CC_eq), no], dtype=np.complex128)
                for ik, [wk, k] in enumerate(zip(H.mp.weight, H.mp.k)):
                    Dk = np.zeros_like(D) # Density matrix per k point
                    if H.spin_size==2:
                        HC = H.H.Hk(spin=spin, k=k, format='array')
                    else:
                        HC = H.H.Hk(k=k, format='array')
                    if self.NEQ:
                        # Correct Density matrix with Non-equilibrium integrals
                        Delta, w = self.Delta(HC, Ef, ik, spin=spin)
                        # Store only diagonal
                        w = np.diag(w)
                        # Transfer Delta to D
                        Dk[0, :] = np.diag(Delta[1]) # Correction to left: Delta_R
                        Dk[1, :] = np.diag(Delta[0]) # Correction to right: Delta_L
                        # TODO We need to also calculate the total energy for NEQ
                        #      this should probably be done in the Delta method
                        del Delta
                    else:
                        # This ensures we can calculate energy for EQ only calculations
                        w = 1.

                    # Loop over all eq. Contours
                    for cc_eq_i, CC in enumerate(self.CC_eq):
                        for ic, [cc, wi] in enumerate(zip(CC + Ef, self.w_eq)):

                            GF = _G(cc, HC, self.elec_idx, cc_eq_SE[spin][ik][cc_eq_i][ic])

                            # Greens function evaluated at each point of the CC multiplied by the weight
                            Gf_wi = - np.diag(GF) * wi
                            Dk[cc_eq_i] += Gf_wi.imag

                            # Integrate density of states to obtain the total energy
                            # For the non equilibrium energy maybe we could obtain it as in PRL 70, 14 (1993)
                            if cc_eq_i == 0:
                                Etot += ((w * Gf_wi).sum() * cc).imag * wk # Integrate over k-space with weight wk
                            else:
                                Etot += (((1 - w) * Gf_wi).sum() * cc).imag * wk # Integrate over k-space with weight wk

                    if self.NEQ:
                        Dk = w * Dk[0] + (1 - w) * Dk[1]
                    else:
                        Dk = Dk[0]

                    # Integrate over k-space with weight wk
                    D += Dk * wk
                    ni[spin, :] = D.real

            # Calculate new charge
            ntot = ni.sum()

        # Save Fermi-level of the device
        self.Ef = Ef

        # Return spin densities and total energy, if the Hamiltonian is not spin-polarized
        # multiply Etot by 2 for spin degeneracy
        return ni, (2./H.spin_size)*Etot

    def Delta(self, HC, Ef, ik, spin=0):
        """
        Finds the non-equilibrium integrals to correct the left and right equilibrium integrals

        Parameters
        ----------
        HC: numpy.ndarray
            Hamiltonian of the central region in its matrix form
        Ef: float
            Potential of the device
        ik: int
            k-point index
        spin: int
            spin index (0=up, 1=dn)

        Returns
        -------
        Delta
        weight
        """
        def spectral(G, self_energy):
            # Use self-energy of elec, now the matrix will have dimension (Nelec, Nelec)
            Gamma = 1j * (self_energy - np.conjugate(self_energy.T))
            # Product of (Ndev, Nelec) x (Nelec, Nelec) x (Nelec, Ndev)
            return np.dot(G, np.dot(Gamma, np.conjugate(G.T)))

        no = len(HC)
        Delta = np.zeros([2, no, no], dtype=np.complex128)
        cc_neq_SE = self._cc_neq_SE[spin][ik]

        for ic, cc in enumerate(self.CC_neq + Ef):

            GF = _G(cc, HC, self.elec_idx, cc_neq_SE[ik][ic])

            # Elec (0, 1) are (left, right)
            # only do for the first two!
            for i, SE in enumerate(cc_neq_SE[ik][ic][:2]):
                Delta[i] += spectral(GF[:, self.elec_idx[i].ravel()], SE) * self.w_neq[i, ic]

        # Firstly implement it for two terminals following PRB 65 165401 (2002)
        # then we can think of implementing it for N terminals as in Com. Phys. Comm. 212 8-24 (2017)
        weight = Delta[0] ** 2 / (Delta ** 2).sum(axis=0)

        return Delta, weight

    def DOS(self, H, E, spin=[0, 1], eta=0.01):
        r"""
        Obtains the density of states (DOS) from the Green's function of the device

        .. math::

            \mathrm{DOS}_\sigma = -\frac{1}{\pi}\Im\lbrace\mathrm{Tr}[G^{\sigma}]\rbrace

        Parameters
        ----------
        H: HubbardHamiltonian
            `hubbard.HubbardHamiltonian` object of the system
        E: array_like
            energy grid to obtan the DOS
        spin: int, array_like, optional
            spin index. If ``spin=[0,1]`` (default) it sums the DOS corresponding to both spin indices
        eta: float, optional
            smearing parameter (complex term in the Green's function)

        Returns
        -------
        DOS : numpy.ndarray
        """
        # Ensure spin instance is iterable
        if not isinstance(spin, (list, tuple, np.ndarray)):
            spin = [spin]

        dos = np.zeros([len(E)])
        for ispin in spin:
            HC = H.H.Hk(spin=ispin, format='array')
            for i, e in enumerate(E):

                # Append all the self-energies for the electrodes at each energy point
                SE = [se.self_energy(e, spin=ispin) for se in self.elec_SE]
                GF = _G(e + 1j * eta, HC, self.elec_idx, SE)

                dos[i] -= np.trace(GF).imag

        return dos / np.pi

    def PDOS(self, H, E, spin=(0, 1), eta=0.01):
        r"""
        Obtains the projected density of states (PDOS) onto the atomic sites from the Green's function of the device

        .. math::

            \mathrm{PDOS}_{i\sigma} = -\frac{1}{\pi} \Im\lbrace G^{\sigma}_{ii}\rbrace

        Where :math:`i` represents the atomic site position

        Parameters
        ----------
        H: HubbardHamiltonian
            `hubbard.HubbardHamiltonian` object of the system
        E: array_like
            energy grid to obtan the PDOS
        spin: int, array_like, optional
            spin index. If ``spin=[0,1]`` (default) it sums the PDOS corresponding to both spin indices
        eta: float, optional
            smearing parameter (complex term in the Green's function)

        Returns
        -------
        PDOS : numpy.ndarray
        """
        # Ensure spin instance is iterable
        if not isinstance(spin, (list, tuple, np.ndarray)):
            spin = [spin]

        ldos = np.zeros([len(E), len(H.H)])
        for ispin in spin:
            HC = H.H.Hk(spin=ispin, format='array')
            for i, e in enumerate(E):
                SE = [se.self_energy(e, spin=ispin) for se in self.elec_SE]
                GF = _G(e + 1j * eta, HC, self.elec_idx, SE)
                ldos[i] -= GF.diagonal().imag

        return ldos / np.pi
