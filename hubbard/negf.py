import numpy as np
from numpy import einsum, conj
import sisl
import os
import math
from scipy.interpolate import interp1d
import scipy.sparse as sp
from hubbard.block_linalg import block_td, Blocksparse2Numpy, sparse_find_faster, Build_BTD_vectorised
from hubbard.block_linalg import slices_to_npslices, test_partition_2d_sparse_matrix

_pi = math.pi

__all__ = ['NEGF']


def _G_dens(e, HC, elec_idx, SE, mode='DOS'):
    """ Calculate Green's function and return the diagonal

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
    mode: str, optional
        return the full Green's function (``mode='Full'``) or only the diagonal (``mode='DOS'``)
    """
    no = len(HC)
    if mode == 'DOS':
        GF = np.zeros([len(e), no], dtype=np.complex128)
    elif mode == 'Full':
        GF = np.zeros([len(e), no, no], dtype=np.complex128)
    # This if statement overcomes cases where there are no electrodes
    for ie, e_i in enumerate(e):
        inv_GF = e_i * np.identity(no) - HC
        for idx, se in zip(elec_idx, SE):
            inv_GF[idx, idx.T] -= se[ie]
        if mode == 'DOS':
            GF[ie] = np.linalg.inv(inv_GF)[np.arange(no), np.arange(no)]
        elif mode == 'Full':
            GF[ie] = np.linalg.inv(inv_GF)
    return GF

# shorthand
def CZ(s, dt=np.complex128):
    return np.zeros(s, dtype = dt)

# Proposed new _G
def _G(e, HC, elec_idx, SE, tbt=None, Ov=None,
       dtype=np.complex128, mode='DOS', alloced_G=None):
    """ Calculate Green function using BTD procedure

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
    tbt: sisl.Sile, optional
        sisl.Sile with a tbtrans calculation to get the pivotting scheme. Defaults to None
        If `tbt=None` should give _G:
    Ov: scipy.sparse.csr (check), optional
        Overlap matrix, May just be an allocated identity matrix in sparse format
    dtype: np.dtype, optional
        datatype of BTD matrix
    mode: str, optional
        "DOS", "SpectralColumn", or "Full". Defaults to "DOS"
    alloced_G:  block_td instance, optional
        the arrays in the BTD class can be preallocated, may or may not be notable. Defaults to None
    """

    no = HC.shape[0]
    if isinstance(HC, np.ndarray):
        return _G_dens(e, HC, elec_idx, SE, mode=mode)

    else:
        if tbt is None:
            return _G_dens(e, HC.toarray(), elec_idx, SE, mode=mode)
        print(mode)
        piv  = tbt.pivot(); ipiv = tbt.ipivot()
        btd  = tbt.btd()
        hk  =  HC
        if Ov is None:
            sk = sp.identity(no)

        else:
            sk = Ov

        Part = [0]
        for b in btd:
            Part+= [Part[-1] + b]

        npiv = len(piv)
        # We could put in the Hamiltonian/overlap here to really check if we
        # are throwing away matrix elements
        f, S = test_partition_2d_sparse_matrix(sp.csr_matrix((npiv, npiv)),Part)

        nS   = slices_to_npslices(S)
        n_diags = len(Part)-1
        ne = len(e)
        if alloced_G is None:
            Al    =  [CZ((ne,Part[i+1]-Part[i  ],Part[i+1]-Part[i  ]), dt = dtype) for i in range(n_diags  )]
            Bl    =  [CZ((ne,Part[i+2]-Part[i+1],Part[i+1]-Part[i  ]), dt = dtype) for i in range(n_diags-1)]
            Cl    =  [CZ((ne,Part[i+1]-Part[i  ],Part[i+2]-Part[i+1]), dt = dtype) for i in range(n_diags-1)]
            Ia    =  [i for i in range(n_diags  )]
            Ib    =  [i for i in range(n_diags-1)]
            Ic    =  [i for i in range(n_diags-1)]
            iGreens = block_td(Al,Bl,Cl,Ia,Ib,Ic,diagonal_zeros=False, E_grid = e)
        else:
            iGreens = alloced_G

        ELEC_IDX = []
        for e_idx in elec_idx:
            iidx = np.zeros(len(e_idx[:,0])**2, dtype = np.int32)
            jidx = np.zeros(len(e_idx[:,0])**2, dtype = np.int32)
            it = 0
            _help_idx = np.arange(no)
            for i in e_idx[:,0]:
                for j in e_idx[:,0]:
                    i, j = _help_idx[i], _help_idx[j]
                    iidx[it] = i
                    jidx[it] = j
                    it += 1

            ELEC_IDX+=[(iidx,jidx)]

        i1, j1, d1 = [],[],[]
        if len(SE[0].shape)==2:
            SE = [se[np.newaxis, :,:] for se in SE]

        for j, z in enumerate(e):
            se_list = []
            for ielec in range(len(SE)):
                IDX = ELEC_IDX[ielec]
                se  = SE[ielec][j]
                se_sparse = sp.csr_matrix((se.ravel(), IDX), shape = (no,no),dtype = complex)
                se_list  += [se_sparse]

            iG = sk * z - hk - sum(se_list)
            iG = iG[piv, :][:, piv]
            di, dj, dv = sparse_find_faster(iG) # sp.find(iG) but faster
            i1.append(di); j1.append(dj); d1.append(dv)

        Av, Bv, Cv = Build_BTD_vectorised(np.vstack(i1), np.vstack(j1), np.vstack(d1), nS)

        for b in range(n_diags):
            iGreens.Al[b][ :, :, :] = Av[b]
            if b<n_diags-1:
                iGreens.Bl[b][ :, :, :] = Bv[b]
                iGreens.Cl[b][ :, :, :] = Cv[b]

        if mode == 'DOS':
            nb  = iGreens.Block_shape[0]
            msk = np.diag(np.ones(nb).astype(int))
            G   = iGreens.Invert(msk)
            return np.hstack([np.diagonal(G.Block(i,i), axis1=1, axis2=2)
                              for i in range(nb)])[:,ipiv]

        elif mode == 'SpectralColumn':
            cols = []
            for eidx in elec_idx:
                eidx = np.array([np.where(piv == eidx[i])[0][0] for i in range(len(eidx))])
                emin, emax = eidx.min(), eidx.max()
                begin = False
                for i in range(n_diags):
                    s = iGreens.all_slices[i][i][0]
                    start,stop = s.start,s.stop
                    if start <= emin < stop: begin = True
                    if start>emax:           begin = False
                    if begin:                cols+=[i]

            cols = np.array(cols)
            cols = np.unique(cols)
            mask = np.diag(np.ones(n_diags)).astype(np.int32)
            mask[:,cols] = 1
            return Blocksparse2Numpy(iGreens.Invert(mask), iGreens.all_slices)[:,ipiv, :][:,:,ipiv]


        elif mode == 'Full':
            G = iGreens.Invert(BW='Upper')
            G.Symmetric = 'Set this to whatever, the code checks if the G object has this attribute, not its value'
            return Blocksparse2Numpy(G, iGreens.all_slices)[:,ipiv, :][:,:,ipiv]
        elif mode == 'nonsymmetric':
            G = iGreens.Invert(BW='all')
            return Blocksparse2Numpy(G, iGreens.all_slices)[:,ipiv, :][:,:,ipiv]

def _nested_list(*args):
    if len(args) == 0:
        return None
    l = []
    for _ in range(args[0]):
        l.append(_nested_list(*args[1:]))
    return l


"""def _choose_G(*args, **kwargs):
    if "tbt" not in kwargs:
        _G = _G_dens(*args)
    else:
        tbt = kwargs["tbt"]
        Ov  = kwargs["Ov"] if "Ov" in kwargs else None
        Alloced_G  = kwargs["Alloced_G"] if "Alloced_G" in kwargs else None
        def _G(*args):
            return _G_btd(*args, tbt=tbt, Ov=Ov, Alloced_G=Alloced_G)"""

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

        if "tbt" in kwargs:
            self.tbt = kwargs["tbt"]
        else:
            self.tbt = None
        if "Ov" in kwargs:
            self.Ov  = kwargs["Ov"]
        else:
            self.Ov = None
        if "Alloced_G" in kwargs:
            self.Alloced_G  = kwargs["Alloced_G"]
        else:
            self.Alloced_G = None

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

        # spin, electrode
        self._ef_SE = _nested_list(Hdev.spin_size, len(self.elec_SE))
        # spin, EQ-contour, electrode, energy
        self._cc_eq_SE = _nested_list(Hdev.spin_size, self.CC_eq.shape[0], len(self.elec_SE), self.CC_eq.shape[1])
        # spin, electrode, energy
        self._cc_neq_SE = _nested_list(Hdev.spin_size, len(self.elec_SE), self.CC_neq.shape[0])

        kw = {}
        for i, se in enumerate(self.elec_SE):
            for spin in range(Hdev.spin_size):
                # Map self-energy at the Fermi-level of each electrode into the device region
                if Hdev.spin_size > 1:
                    kw = {'spin':spin}

                self._ef_SE[spin][i] = se.self_energy(1j * self.eta, **kw)

                for cc_eq_i, CC_eq in enumerate(self.CC_eq):
                    for ic, cc in enumerate(CC_eq):
                        # Do it also for each point in the CC, for all EQ CC
                        self._cc_eq_SE[spin][cc_eq_i][i][ic] = se.self_energy(cc, **kw)
                if self.NEQ:
                    for ic, cc in enumerate(self.CC_neq):
                        # And for each point in the Neq CC
                        self._cc_neq_SE[spin][i][ic] = se.self_energy(cc, **kw)


    def calc_n_open(self, H, q, qtol=1e-5, Nblocks=5):
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
        Nblocks: int, optional
            number of blocks in which the energy contour will be split to obtain the Green's matrix  (to obtain the NEQ integrals)

        Returns
        -------
        ni: numpy.ndarray
            spin densities
        Etot: float
            total energy
        """
        form   = 'csr' if self.tbt is not None else 'array'

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
                        if H.spin_size == 2:
                            HC = H.H.Hk(spin=spin, format=form)
                        else:
                            HC = H.H.Hk(format=form)

                        GF = _G([Ef + 1j * self.eta], HC, self.elec_idx, ef_SE[spin],
                                tbt=self.tbt, Ov=self.Ov, alloced_G=self.Alloced_G)

                        # Now we need to calculate the new Fermi level based on the
                        # difference in charge and by estimating the current Fermi level
                        # as the peak position of a Lorentzian.
                        # I.e. F(x) = \int_-\infty^x L(x) dx = arctan(x) / pi + 0.5
                        #   F(x) - F(0) = arctan(x) / pi = dq
                        # In our case we *know* that 0.5 = - Im[Tr(Gf)] / \pi
                        # and consider this a pre-factor
                        f -= GF.sum(axis=1).imag / _pi

                        # calculate fractional change
                        f = dq / f
                        # Since x above is in units of eta, we have to multiply with eta
                        if abs(f) < 0.45:
                            Ef -= self.eta * math.tan(f * _pi) * 0.5
                        else:
                            Ef -= self.eta * math.tan((_pi / 2 - math.atan(1 / (f * _pi)))) * 0.5

            Etot = 0.
            for spin in range(H.spin_size):
                if H.spin_size==2:
                    HC = H.H.Hk(spin=spin, format=form)
                else:
                    HC = H.H.Hk(format=form)
                D = np.zeros([len(self.CC_eq), no], dtype=np.complex128)
                if self.NEQ:
                    # Correct Density matrix with Non-equilibrium integrals
                    Delta, w = self.Delta(HC, Ef, spin=spin, Nblocks=Nblocks)
                    # Transfer Delta to D
                    D[0, :] = Delta[1] # Correction to left: Delta_R
                    D[1, :] = Delta[0] # Correction to right: Delta_L
                    # TODO We need to also calculate the total energy for NEQ
                    #      this should probably be done in the Delta method
                    del Delta
                else:
                    # This ensures we can calculate energy for EQ only calculations
                    w = 1.

                # Loop over all eq. Contours
                for cc_eq_i, CC in enumerate(self.CC_eq):
                    cc = CC + Ef
                    self_energy = cc_eq_SE[spin][cc_eq_i]
                    GF = _G(cc, HC, self.elec_idx, self_energy, mode='DOS')

                    # Greens function evaluated at each point of the CC multiplied by the weight
                    # Each row is the diagonal of Gf(e) multiplied by the weight
                    Gf_wi = - GF * self.w_eq.reshape(-1,1)
                    D[cc_eq_i] = Gf_wi.imag.sum(axis=0) # sum elements of each row to evaluate integral over energy

                    # Integrate density of states to obtain the total energy
                    # For the non equilibrium energy maybe we could obtain it as in PRL 70, 14 (1993)
                    if cc_eq_i == 0:
                        # Sum over spin components
                        Etot += ((w * Gf_wi).sum(axis=1) * cc).sum().imag
                    else:
                        Etot += (((1 - w) * Gf_wi).sum(axis=1) * cc).sum().imag

                if self.NEQ:
                    D = w * D[0] + (1 - w) * D[1]
                else:
                    D = D[0]

                ni[spin, :] = D.real

            # Calculate new charge
            ntot = ni.sum()

        # Save Fermi-level of the device
        self.Ef = Ef

        # Return spin densities and total energy, if the Hamiltonian is not spin-polarized
        # multiply Etot by 2 for spin degeneracy
        return ni, (2./H.spin_size)*Etot

    def Delta(self, HC, Ef, spin=0, Nblocks=3):
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
        Nblocks: int, optional
            number of blocks in which the energy contour will be split to obtain the Green's matrix, this number should be larger
            for a relative large number of orbitals

        Returns
        -------
        Delta
        weight
        """
        def spectral(G, self_energy):
            # Use self-energy of elec, now the matrix will have dimension (E, Nelec, Nelec)
            Gamma = 1j * (self_energy - np.conjugate(np.transpose(self_energy, axes=[0,2,1])))
            # Product of (E, Ndev, Nelec) x (E, Nelec, Nelec) x (E, Nelec, Ndev) -> (E, Ndev, Ndev)
            return einsum('ijk, ikm, imj ->  ij', G, Gamma, np.conjugate(np.transpose(G, axes=[0,2,1])))

        no = len(HC)
        Delta = np.zeros([2, no], dtype=np.complex128)
        cc_neq_SE = self._cc_neq_SE[spin]

        # Elec (0, 1) are (left, right)
        # only do for the first two!
        for i in range(2):
            for ic, CC in enumerate(np.array_split((self.CC_neq + Ef), Nblocks)):
                GF = _G(CC, HC, self.elec_idx, cc_neq_SE, mode='Full')
                A = spectral(GF[:, :, self.elec_idx[i].ravel()], np.array_split(np.array(cc_neq_SE)[:, i], Nblocks)[ic])
                # Build Delta for each electrode
                Delta[i] += einsum('i, ij -> j', np.array_split(self.w_neq[i], Nblocks)[ic], A)

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
                GF = _G([e + 1j * eta], HC, self.elec_idx, SE)

                dos[i] -= GF.sum().imag

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
                GF = _G([e + 1j * eta], HC, self.elec_idx, SE)
                ldos[i] -= GF.imag

        return ldos / np.pi
