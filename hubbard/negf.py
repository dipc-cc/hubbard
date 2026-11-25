import numpy as np
from numpy import einsum, conj
import sisl
import os
import math
from scipy.linalg import inv

_pi = math.pi
# Largest step of the device Fermi level before the charge root is bracketed (eV)
# and the number of such steps calc_n_open takes before giving up.
# The step is below the band width of the models hubbard builds (sp2: t = 2.7 eV),
# so it cannot jump across a whole spectrum; 50 such steps reach any band
# from a start up to 50 eV away.
_EF_STEP_MAX = 1.
_EF_BRACKET_PASSES = 50

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


def _self_energy(se, E, **kwargs):
    r""" Self-energy of an electrode at energy `E`

    sisl's own `sisl.WideBandSE.self_energy` is replaced by :math:`-i\eta`
    times the identity, with :math:`\eta` the electrode's own ``eta``, at
    every `E`. sisl < 0.16 returns this itself; sisl >= 0.16 returns
    :math:`-i\pi\Im(E)` times the identity (or times the overlap matrix,
    for an electrode built from a non-orthogonal matrix) instead. Whether
    it is replaced is decided by the method bound to `se`: an override, in
    a subclass or on the instance, is called as is, and so is any other
    self-energy. An override that itself calls sisl's
    `sisl.WideBandSE.self_energy` gets sisl's version-dependent value.

    Parameters
    ----------
    se : sisl.SelfEnergy
       the electrode
    E : complex
       energy at which the self-energy is calculated
    **kwargs :
       passed on to ``se.self_energy`` (``k``, ``spin``)
    """
    if (isinstance(se, sisl.WideBandSE)
            and getattr(se.self_energy, "__func__", None) is sisl.WideBandSE.self_energy):
        return np.diag(np.full(len(se), -1j * se.eta, dtype=np.complex128))
    return se.self_energy(E, **kwargs)


def _nested_list(*args):
    if len(args) == 0:
        return None
    l = []
    for _ in range(args[0]):
        l.append(_nested_list(*args[1:]))
    return l


class NEGF:
    r""" This class creates the open quantum system object for a N-terminal device

    Only equilibrium (:math:`V=0`) is supported; finite bias raises
    `NotImplementedError`.

    Parameters
    ----------
    Hdev: HubbardHamiltonian
        `hubbard.HubbardHamiltonian` object of the device
    elec_SE: list of sisl.SelfEnergy or tuple of (HubbardHamiltonian, str)
        list of (already converged) `hubbard.HubbardHamiltonian` objects for the electrodes
        plus the semi-infinite direction for the respective electrode.
        Alternatively one may directly pass `sisl.SelfEnergy` instances.
        A `sisl.WideBandSE` that keeps sisl's own ``self_energy`` method couples
        its orbitals with the energy-independent self-energy :math:`-i\eta` times
        the identity, with :math:`\eta` (in eV) its own ``eta``, whatever the
        sisl version: that method, which on sisl >= 0.16 returns
        :math:`-i\pi\Im(E)` instead, is not called. An override of
        ``self_energy``, in a subclass or on the instance, is called as is;
        if it calls sisl's ``WideBandSE.self_energy`` itself, it gets sisl's
        version-dependent value
    elec_idx: array_like
        list of atomic positions that *each* electrode occupies in the device geometry
    CC: str, optional
        name of the file containing the energy contour in the complex plane to integrate the density matrix
    V: float, optional
        applied bias between the two electrodes. Only ``V=0`` (the default)
        is supported.

    Raises
    ------
    NotImplementedError
        if `V` is non-zero: finite bias is not supported.


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
        if V != 0:
            raise NotImplementedError(f"{self.__class__.__name__}: finite bias (V={V!r}) is not supported; use V=0")

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

                    self._ef_SE[spin][ik][i] = _self_energy(se, 1j * self.eta, k=k, **kw)

                    for cc_eq_i, CC_eq in enumerate(self.CC_eq):
                        for ic, cc in enumerate(CC_eq):
                            # Do it also for each point in the CC, for all EQ CC
                            self._cc_eq_SE[spin][ik][cc_eq_i][ic][i] = _self_energy(se, cc, k=k, **kw)

                    if self.NEQ:
                        for ic, cc in enumerate(self.CC_neq):
                            # And for each point in the Neq CC
                            self._cc_neq_SE[spin][ik][ic][i] = _self_energy(se, cc, k=k, **kw)

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

        Raises
        ------
        RuntimeError
            if no device Fermi level brackets the charge `q` within 50 steps of at
            most 1 eV each from the starting `self.Ef`, for instance when it starts
            far outside the bands; or if the bracket around the Fermi level shrinks
            to floating-point resolution before the charge is within `qtol`, for
            instance when `qtol` is below the precision of the contour integration
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

        # The device charge grows with Ef, so the root of dq(Ef) = ni.sum() - q
        # is searched in two stages:
        # 1. until a bracket exists, a Lorentzian step from the DOS at Ef. A step
        #    that did not at least halve |dq| did worse than a bisection would,
        #    so the next one is expanded to the zero of the line through the last
        #    two points (twice that if |dq| hardly changed), and at least doubled,
        #    until the root gets bracketed.
        #    Steps are capped at _EF_STEP_MAX, so that Ef cannot run away where
        #    N(Ef) no longer changes (outside the bands), and after
        #    _EF_BRACKET_PASSES steps without a bracket the search fails;
        # 2. once Ef with dq < 0 (lo) and dq > 0 (hi) are known, the Illinois
        #    variant of regula falsi inside [lo, hi], keeping only the latest
        #    point on each side. Plain regula falsi stalls when one end of the
        #    bracket never moves (as it does when Ef must cross a gap); Illinois
        #    halves the dq of an end retained twice in a row, which restores
        #    superlinear convergence. A step outside the bracket, a non-finite
        #    one, or a fourth update in a row on the same side bisects instead,
        #    so the bracket always shrinks, until no float lies inside it and the
        #    search fails.
        lo = hi = None # [Ef, dq] of the bracket ends
        side = 0 # side of the last update: -1 (lo) or 1 (hi)
        nside = 0 # number of consecutive updates on that side
        step = 0. # the last step taken without a bracket
        dq_step = 0. # dq before that step
        nopen = 0 # number of steps taken without a bracket
        first = True # the first pass only computes the charge at the starting Ef
        dq_min = math.inf # smallest |dq| reached
        while abs(ntot - q) > qtol:
            # for debug purposes
            #print(ntot, q, qtol, Ef)

            if not first:
                # correct fermi-level
                dq = ni.sum() - q
                dq_min = min(dq_min, abs(dq))

                s = -1 if dq < 0 else 1
                nside = nside + 1 if s == side else 1
                side = s
                if s < 0:
                    lo = [Ef, dq]
                else:
                    hi = [Ef, dq]

                if lo is not None and hi is not None:
                    if nside > 1:
                        # Illinois: halve dq of the end that was retained again
                        if s < 0:
                            hi[1] *= 0.5
                        else:
                            lo[1] *= 0.5
                    Ef = lo[0] - lo[1] * (hi[0] - lo[0]) / (hi[1] - lo[1])
                    a, b = min(lo[0], hi[0]), max(lo[0], hi[0])
                    if nside > 3 or not np.isfinite(Ef) or not a < Ef < b:
                        Ef = 0.5 * (a + b)
                    if not a < Ef < b:
                        # No float lies strictly inside the bracket: it cannot shrink
                        raise RuntimeError(f"{self.__class__.__name__}.calc_n_open cannot shrink the bracket "
                                           f"[{a}, {b}] of the Fermi level any further: |dq| = {dq_min} "
                                           f"reached, qtol = {qtol}")

                else:

                    # Fermi-level is at 0.
                    # The Lorentzian has ~70% of its integral within
                    #   2 * \eta (on both sides)
                    # To cover 200 meV ~ 2400 K in the integration window
                    # and expect the Lorentzian peak to be positioned at
                    # the current Fermi-level we will use eta = 100 meV
                    # Calculate charge at the Fermi-level
                    # The density of states at Ef, broadened by eta, summed over the
                    # spin channels in ni and over k with the weights wk.
                    # For an unpolarized H, ni and q both hold one spin channel, so
                    # dq and this sum are both per spin and their ratio needs no
                    # degeneracy factor.
                    f = 0.
                    cc = Ef + 1j * self.eta
                    for spin in range(H.spin_size):
                        for ik, [wk, k] in enumerate(zip(H.mp.weight, H.mp.k)):
                            if H.spin_size == 2:
                                HC = H.H.Hk(spin=spin, k=k, format='array')
                            else:
                                HC = H.H.Hk(k=k, format='array')

                            GF = _G(cc, HC, self.elec_idx, ef_SE[spin][ik])

                            # Now we need to calculate the new Fermi level based on the
                            # difference in charge and by estimating the current Fermi level
                            # as the peak position of a Lorentzian.
                            # I.e. F(x) = \int_-\infty^x L(x) dx = arctan(x) / pi + 0.5
                            #   F(x) - F(0) = arctan(x) / pi = dq
                            # In our case we *know* that 0.5 = - Im[Tr(Gf)] / \pi
                            # and consider this a pre-factor
                            f -= np.trace(GF).imag * wk / _pi # Integrate over k-space with weight wk

                    # calculate fractional change, and move Ef once for all spins
                    f = dq / f
                    # Since x above is in units of eta, we have to multiply with eta
                    if abs(f) < 0.45:
                        dEf = - self.eta * math.tan(f * _pi) * 0.5
                    else:
                        # eta * tan(pi/2 - atan(1 / (f * pi))) / 2, written so that
                        # the sign is kept and an infinite f stays infinite
                        dEf = - self.eta * _pi * f * 0.5
                    if not np.isfinite(dEf):
                        dEf = - s * self.eta
                    # Without a bracket every step goes the same way, -sign(dq)
                    r = abs(dq / dq_step) if step != 0. else 0.
                    if r > 0.5:
                        # The last step did not halve |dq|. Expand it to where the line
                        # through the last two points meets dq = 0, and at least double it.
                        # If |dq| changed by less than 10%, that line is the local slope,
                        # which undershoots on the exponential tail of a band edge (about
                        # kT per step), so aim twice as far to bracket the root.
                        m = r / (1 - r) if r < 1 else math.inf
                        if r > 0.9:
                            m *= 2
                        dEf = math.copysign(max(abs(dEf), max(2., m) * abs(step)), step)
                    dEf = math.copysign(min(abs(dEf), _EF_STEP_MAX), dEf)
                    nopen += 1
                    if nopen > _EF_BRACKET_PASSES:
                        raise RuntimeError(f"{self.__class__.__name__}.calc_n_open found no Fermi level "
                                           f"that brackets the charge q={q} within {_EF_BRACKET_PASSES} "
                                           f"steps of at most {_EF_STEP_MAX} eV (Ef={Ef}, dq={dq})")
                    step = dEf
                    dq_step = dq
                    Ef += dEf

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
            first = False

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

    def _check_dos_input(self, H, method):
        """ Refuse an input for which `DOS` and `PDOS` would return a wrong spectrum """
        name = f"{self.__class__.__name__}.{method}"
        spin_size = len(self._ef_SE)
        if H.spin_size != spin_size:
            raise ValueError(f"{name}: H has spin_size={H.spin_size}, "
                             f"but the electrode self-energies were set up for spin_size={spin_size}")
        for i, se in enumerate(self.elec_SE):
            # The Hamiltonian a semi-infinite (spgeom0) or real-space (parent) electrode
            # is built from; read from the instance, not through SelfEnergy.__getattr__.
            # A WideBandSE couples every spin channel alike and is not checked
            spin = None
            for attr in ('spgeom0', 'parent'):
                spin = getattr(getattr(se, '__dict__', {}).get(attr), 'spin', None)
                if spin is not None:
                    break
            if spin is not None and spin.spinor != spin_size:
                raise ValueError(f"{name}: electrode {i} has spin_size={spin.spinor}, "
                                 f"but the device has spin_size={spin_size}")
        if len(H.mp.k) > 1:
            raise NotImplementedError(f"{name}: H has {len(H.mp.k)} k-points, but only the Gamma point "
                                      "is evaluated; k-sampling is #141")
        if not H.H.orthogonal:
            raise NotImplementedError(f"{name}: a non-orthogonal H is not supported")

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
            spin index. If ``spin=[0,1]`` (default) it sums the DOS corresponding to both spin indices.
            For an unpolarized Hamiltonian ``spin=0`` and ``spin=1`` both give the DOS of its
            single spin channel, so the default is twice that of ``spin=0``, as in
            `hubbard.HubbardHamiltonian.DOS`
        eta: float, optional
            smearing parameter (complex term in the Green's function)

        Returns
        -------
        DOS : numpy.ndarray

        Raises
        ------
        ValueError
            if ``H.spin_size`` differs from that of the device this object was built for, or
            if a recursive or real-space electrode was built from a Hamiltonian of another spin
            size than the device's
        NotImplementedError
            if `H` has more than one k-point (only the Gamma point is evaluated), or if `H` is
            non-orthogonal
        """
        self._check_dos_input(H, 'DOS')
        # Ensure spin instance is iterable
        if not isinstance(spin, (list, tuple, np.ndarray)):
            spin = [spin]

        dos = np.zeros([len(E)])
        for ispin in spin:
            # An unpolarized Hamiltonian has a single spin channel, which
            # Hk and the self-energies take without a spin keyword
            kw = {'spin': ispin} if H.spin_size > 1 else {}
            HC = H.H.Hk(format='array', **kw)
            for i, e in enumerate(E):

                # Append all the self-energies for the electrodes at each energy point
                SE = [_self_energy(se, e, **kw) for se in self.elec_SE]
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
            spin index. If ``spin=[0,1]`` (default) it sums the PDOS corresponding to both spin indices.
            For an unpolarized Hamiltonian ``spin=0`` and ``spin=1`` both give the PDOS of its
            single spin channel, so the default is twice that of ``spin=0``, as in
            `hubbard.HubbardHamiltonian.PDOS`
        eta: float, optional
            smearing parameter (complex term in the Green's function)

        Returns
        -------
        PDOS : numpy.ndarray

        Raises
        ------
        ValueError
            if ``H.spin_size`` differs from that of the device this object was built for, or
            if a recursive or real-space electrode was built from a Hamiltonian of another spin
            size than the device's
        NotImplementedError
            if `H` has more than one k-point (only the Gamma point is evaluated), or if `H` is
            non-orthogonal
        """
        self._check_dos_input(H, 'PDOS')
        # Ensure spin instance is iterable
        if not isinstance(spin, (list, tuple, np.ndarray)):
            spin = [spin]

        ldos = np.zeros([len(E), len(H.H)])
        for ispin in spin:
            # An unpolarized Hamiltonian has a single spin channel, which
            # Hk and the self-energies take without a spin keyword
            kw = {'spin': ispin} if H.spin_size > 1 else {}
            HC = H.H.Hk(format='array', **kw)
            for i, e in enumerate(E):
                SE = [_self_energy(se, e, **kw) for se in self.elec_SE]
                GF = _G(e + 1j * eta, HC, self.elec_idx, SE)
                ldos[i] -= GF.diagonal().imag

        return ldos / np.pi
