import numpy as np
from numpy import einsum, conj
import sisl
import os
import math
from scipy.linalg import inv

_pi = math.pi

__all__ = ['NEGF']

def _inv_G(e, HC, elec_indx, SE):
    no = len(HC)
    inv_GF = np.zeros([no, no], dtype=np.complex128)
    np.fill_diagonal(inv_GF, e)
    inv_GF[:, :] -= HC[:, :]
    for i, se in enumerate(SE):
        inv_GF[elec_indx[i], elec_indx[i].T] -= se
    return inv_GF

class NEGF(object):
    r""" This class creates the open quantum system object for a N-terminal device

    For the Non-equilibrium case :math:`V\neq 0` the current implementation
    only can deal with a 2-terminal device

    Parameters
    ----------
    Hdev: HubbardHamiltonian instance
        `Hubbard.HubbardHamiltonian` object of the device
    Helecs: list of HubbardHamiltonian instances
        list of (already converged) `Hubbard.HubbardHamiltonian` objects for the electrodes
    elec_indx: array_like
        list of atomic positions that *each* electrode occupies in the device geometry
    elec_dir: array_like of strings
        list of axis specification for the semi-infinite direction (`+A`/`-A`/`+B`/`-B`/`+C`/`-C`)
    CC: str, optional
        name of the file containing the energy contour in the complex plane to integrate the density matrix
    V: float, optional
        applied bias between the two electrodes
    WBL: bool, optional
        if True this method uses the wide band limit approximation, where the real part of the self-energy
        is neglected, and the imaginary part is approximated by a constant: ``-1j*gamma``
        on specified sites (`gamma_indx`)

    See Also
    --------
    `sisl.physics.RecursiveSI` : sisl routines to create semi-infinite object (obtain self-energy, etc.)

    Notes
    -----
    This class has to be generalized to non-orthogonal basis
    """

    def __init__(self, Hdev, Helecs, elec_indx, elec_dir=['-A', '+A'], CC=None, V=0, WBL=False, gamma_indx=None, gamma=[0.]):
        """ Initialize NEGF class """
        # Save some relevant quantities in the object
        self.Ef = 0.
        self.kT = Hdev.kT
        self.elec_indx = elec_indx
        self.Helecs = Helecs
        self.elec_dir = elec_dir
        self.Hdev = Hdev
        self.WBL = WBL

        dist = sisl.get_distribution('fermi_dirac', smearing=self.kT)

        self.eta = 0.1

        if not CC:
            CC = os.path.split(__file__)[0]+'/EQCONTOUR'
        contour_weight = sisl.io.tableSile(CC).read_data()
        self.CC_eq = np.array([contour_weight[0] + contour_weight[1]*1j])
        self.w_eq = (contour_weight[2] + contour_weight[3]*1j) / np.pi
        self.NEQ = V != 0

        mu = np.zeros(len(Helecs))
        if self.NEQ:
            mu = [V*0.5, -V*0.5]
            self.CC_eq = np.array([self.CC_eq[0] + mu[0], self.CC_eq[0] + mu[1]])

            # Integration path for the non-Eq window
            dE = 0.01
            self.CC_neq = np.arange(min(mu)-5*self.kT, max(mu)+5*self.kT + dE, dE) + 1j * 0.001
            # Weights for the non-Eq integrals
            w_neq = dE * (dist(self.CC_neq.real - mu[0]) - dist(self.CC_neq.real - mu[1]))
            # Store weights for correction to RIGHT [0] and LEFT [1]
            self.w_neq = np.array([w_neq, -w_neq]) / np.pi
        else:
            self.CC_neq = []

        self.elec_indx = [np.array(idx).reshape(-1, 1) for idx in elec_indx]

        # electrode, spin
        self._ef_SE = []
        # electrode, spin, EQ-contour, energy
        self._cc_eq_SE = []
        # electrode, spin, energy
        self._cc_neq_SE = []

        for i, elec in enumerate(Helecs):
            Ef_elec = elec.H.fermi_level(elec.mp, q=elec.q, distribution=dist)
            # Shift each electrode with its Fermi-level
            # And also shift the chemical potential
            # Since the electrodes are *bulk* i.e. the entire electronic structure
            # is simply shifted we need to subtract since the contour shifts the chemical
            # potential back.
            elec.H.shift(-Ef_elec - mu[i])
            se = sisl.RecursiveSI(elec.H, elec_dir[i])
            _cc_eq_SE = np.array([[[None] * self.CC_eq.shape[1]] * self.CC_eq.shape[0]] * 2)
            _ef_SE = np.array([None] * 2)
            _cc_neq_SE = np.array([[None] * len(self.CC_neq)] * 2)
            for spin in [0, 1]:
                # Map self-energy at the Fermi-level of each electrode into the device region
                _ef_SE[spin] = se.self_energy(2 * mu[i] + 1j * self.eta, spin=spin)

                for cc_eq_i, CC_eq in enumerate(self.CC_eq):
                    for ic, cc in enumerate(CC_eq):
                        # Do it also for each point in the CC, for all EQ CC
                        _cc_eq_SE[spin][cc_eq_i][ic] = se.self_energy(cc, spin=spin)
                if self.NEQ:
                    for ic, cc in enumerate(self.CC_neq):
                        # And for each point in the Neq CC
                        _cc_neq_SE[spin][ic] = se.self_energy(cc, spin=spin)
            self._ef_SE.append(_ef_SE)
            self._cc_eq_SE.append(_cc_eq_SE)
            self._cc_neq_SE.append(_cc_neq_SE)

        if WBL:
            # Ensure gamma is iterable, as the implementation is generalized
            # for several possible gammas
            if not isinstance(gamma, list) or isinstance(gamma, np.ndarray):
                    gamma = [gamma]

            # Atomic indices at which the WBL is going to be applied, default to all sites if the list is empty
            if not gamma_indx:
                gamma_indx = [np.arange(len(Hdev.H)).reshape(-1, 1)] * len(gamma)
            else:
                gamma_indx = [np.array(idx).reshape(-1, 1) for idx in gamma_indx]

            # Save WBL related quantities in the object
            self.gamma = gamma
            self.gamma_indx = gamma_indx

            # All the WBL related quantities (both the atomic indices and the self-energies) are stored
            # at the end of the lists that contain the electrode information
            self.elec_indx += gamma_indx
            for i, g in enumerate(gamma):
                _cc_eq_SE = np.array([[[None] * self.CC_eq.shape[1]] * self.CC_eq.shape[0]] * 2)
                _ef_SE = np.array([None] * 2)
                _cc_neq_SE = np.array([[None] * len(self.CC_neq)] * 2)
                for spin in [0,1]:
                    # For all energies the self-energy term is the same
                    _ef_SE[spin] = -1j*g*np.identity(len(gamma_indx[i]))
                    for cc_eq_i, CC_eq in enumerate(self.CC_eq):
                        for ic, cc in enumerate(CC_eq):
                            # For all energies the self-energy term is the same
                            _cc_eq_SE[spin][cc_eq_i][ic] = -1j*g*np.identity(len(gamma_indx[i]))
                    if self.NEQ:
                        for ic, cc in enumerate(self.CC_neq):
                            # And for each point in the Neq CC
                            _cc_neq_SE[spin][ic] = -1j*g*np.identity(len(gamma_indx[i]))
                self._ef_SE.append(_ef_SE)
                self._cc_eq_SE.append(_cc_eq_SE)
                self._cc_neq_SE.append(_cc_neq_SE)

    def dm_open(self, H, q, qtol=1e-5):
        """
        Method to compute the spin densities from the Neq Green's function

        Parameters
        ----------
        H: HubbardHamiltonian instances
            `Hubbard.HubbardHamiltonian` of the object that is being iterated
        q: float
            charge associated to the up and down spin-components
        qtol: float, optional
            tolerance to which the charge is going to be converged in the internal loop
            that finds the potential of the device (i.e. that makes the device neutrally charged)

        Returns
        -------
        ni
        Etot
        """
        # ensure scalar, for open systems one cannot impose a spin-charge
        # This spin-charge would be dependent on the system size
        q = np.asarray(q).sum()
        no = len(H.H)
        ni = np.empty([2, no], dtype=np.float64)
        ntot = -1.
        Ef = self.Ef

        ef_SE = np.array(self._ef_SE)
        CC_SE = np.array(self._cc_eq_SE)

        while abs(ntot - q) > qtol:

            if ntot > 0.:
                # correct fermi-level
                dq = ni.sum() - q

                # Fermi-level is at 0.
                # The Lorentzian has ~70% of its integral within
                #   2 * \eta (on both sides)
                # To cover 200 meV ~ 2400 K in the integration window
                # and expect the Lorentzian peak to be positioned at
                # the current Fermi-level we will use eta = 100 meV
                # Calculate charge at the Fermi-level
                f = 0.
                for spin in [0, 1]:
                    HC = H.H.Hk(spin=spin).todense()
                    cc = - Ef + 1j * self.eta
                    inv_GF = _inv_G(cc, HC, self.elec_indx, ef_SE[:, spin])

                    # Now we need to calculate the new Fermi level based on the
                    # difference in charge and by estimating the current Fermi level
                    # as the peak position of a Lorentzian.
                    # I.e. F(x) = \int_-\infty^x L(x) dx = arctan(x) / pi + 0.5
                    #   F(x) - F(0) = arctan(x) / pi = dq
                    # In our case we *know* that 0.5 = - Im[Tr(Gf)] / \pi
                    # and consider this a pre-factor
                    f -= np.trace(inv(inv_GF)).imag / _pi

                # calculate fractional change
                f = dq / f 
                # Since x above is in units of eta, we have to multiply with eta
                if abs(f) < 0.45:
                    Ef += self.eta * math.tan(f * _pi) * 0.5
                else:
                    Ef += self.eta * math.tan((_pi / 2 - math.atan(1 / (f * _pi)))) * 0.5

            Etot = 0.
            for spin in [0, 1]:
                HC = H.H.Hk(spin=spin, format='array')
                D = np.zeros([len(self.CC_eq), len(HC)])
                if self.NEQ:
                    # Correct Density matrix with Non-equilibrium integrals
                    Delta, w = self.Delta(HC, Ef, spin=spin)
                    # Store only diagonal
                    w = np.diag(w)
                    # Transfer Delta to D
                    D[0, :] = np.diag(Delta[1])
                    D[1, :] = np.diag(Delta[0])
                    # TODO We need to also calculate the total energy for NEQ
                    #      this should probably be done in the Delta method
                    del Delta
                else:
                    # This ensures we can calculate for EQ only calculations
                    w = 1.

                # Loop over all eq. Contours
                for cc_eq_i, CC in enumerate(self.CC_eq):
                    for ic, [cc, wi] in enumerate(zip(CC - Ef, self.w_eq)):
                        inv_GF = _inv_G(cc, HC, self.elec_indx, CC_SE[:, spin, cc_eq_i, ic])

                        # Greens function evaluated at each point of the CC multiplied by the weight
                        Gf_wi = - np.diag(inv(inv_GF)) * wi
                        D[cc_eq_i] += Gf_wi.imag

                        # Integrate density of states to obtain the total energy
                        # For the non equilibrium energy maybe we could obtain it as in PRL 70, 14 (1993)
                        if cc_eq_i == 0:
                            Etot += ((w*Gf_wi).sum() * cc).imag
                        else:
                            Etot += (((1-w)*Gf_wi).sum() * cc).imag

                if self.NEQ:
                    D = w * D[0] + (1-w) * D[1]
                else:
                    D = D[0]

                ni[spin, :] = D

            # Calculate new charge
            ntot = ni.sum()

        # Save Fermi-level of the device
        self.Ef = Ef

        return ni, Etot

    def Delta(self, HC, Ef, spin=0):
        """
        Finds the non-equilibrium integrals to correct the left and right equilibrium integrals

        Parameters
        ----------
        HC: numpy.ndarray
            Hamiltonian of the central region in its matrix form
        Ef: float
            Potential of the device
        spin: int
            spin index (0=up, 1=dn)

        Returns
        -------
        Delta
        weight
        """
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
            # Calculate the Green function
            inv_GF[:, :] = inv(inv_GF)
            # Elec (0, 1) are (left, right)
            for i, SE in enumerate(self._cc_neq_SE):
                Delta[i] += spectral(inv_GF, SE[spin][ic], i) * self.w_neq[i, ic]

        # Firstly implement it for two terminals following PRB 65 165401 (2002)
        # then we can think of implementing it for N terminals as in Com. Phys. Comm. 212 8-24 (2017)
        weight = Delta[0]**2 / ((Delta**2).sum(axis=0))

        # Get rid of the numerical imaginary part (which is ~0)
        return Delta.real, weight.real

    def DOS(self, H, E, spin=[0,1]):
        '''
        Returns
        -------
        DOS: numpy.array
        '''

        # Ensure spin instance is iterable
        if not isinstance(spin, list) or isinstance(spin, np.ndarray):
            spin = [spin]

        dos = np.zeros((len(spin),len(E)))
        for ispin in range(len(spin)):
            HC = H.H.Hk(spin=ispin, format='array')
            for i, e in enumerate(E):
                # Append all the self-energies for the electrodes at each energy point
                SE = []
                for j, elec in enumerate(self.Helecs):
                    se = sisl.RecursiveSI(elec.H, self.elec_dir[j])
                    SE.append(se.self_energy(e, spin=spin[ispin]))
                if self.WBL:
                    # Append also the WBL self energies
                    for k, g in enumerate(self.gamma):
                        SE.append(-1j*g*np.identity(len(self.gamma_indx[k])))
                inv_GF = _inv_G(e + self.eta*1j, HC, self.elec_indx, SE)

                dos[ispin, i] = - np.trace(inv(inv_GF)).imag
        dos = dos.sum(axis=0)
        return dos/np.pi
