from __future__ import print_function
import numpy as np
from numpy import einsum, conj
import sisl

__all__ = ['dm', 'dm_insulator']


def dm(H, q):
    r""" General method to obtain the spin-densities for periodic or finite systems at a given temperature

    It obtains the densities from the direct diagonalization of the Hamiltonian (``H.H``) taking into account
    a possible overlap matrix (``H.H.S``). 

    It computes the densities as Mulliken populations:

    .. math::
        \langle n_{i\sigma} \rangle = \sum_{\alpha}f_{\alpha\sigma}\sum_{j}c^{\alpha}_{i\sigma}c^{*\alpha}_{j\sigma}S_{ij} = \sum_{j}D_{ij\sigma}S_{ij} =  (\textbf{D}_{\sigma}\textbf{S})_{ii}

    Where :math:`f_{\alpha\sigma}` is the weight of eigenstate :math:`\alpha` for spin :math:`\sigma` at temperature ``kT`` (Fermi-Dirac distribution),
    :math:`c^{\alpha}_{i\sigma}` are the elements of the eigenstate :math:`\alpha` and spin :math:`\sigma` represented in the basis of atomic orbitals,
    and :math:`D_{ij\sigma}` is the full density matrix in the basis of atomic orbitals (indices :math:`i,j`)

    Parameters
    ----------
    H: `Hubbard.HubbardHamiltonian` instance
        `HubbardHamiltonian` object of the system to obtain the spin-densities from
    q: array_like
        charge resolved in spin channels (first index for up-electrons and second index for down-electrons)

    See Also
    --------
    `sisl.physics.EigenstateElectron.norm2` sisl routine to obtain the dot product of the eigenstates with the overlap matrix
    """
    # Create fermi-level determination distribution
    dist = sisl.get_distribution('fermi_dirac', smearing=H.kT)
    Ef = H.H.fermi_level(H.mp, q=q, distribution=dist)
    dist_up = sisl.get_distribution('fermi_dirac', smearing=H.kT, x0=Ef[0])
    dist_dn = sisl.get_distribution('fermi_dirac', smearing=H.kT, x0=Ef[1])

    ni = np.zeros((2, H.sites))
    Etot = 0

    # Solve eigenvalue problems
    def calc_occ(k, weight):
        n = np.empty_like(ni)
        es_up = H.eigenstate(k, spin=0)
        es_dn = H.eigenstate(k, spin=1)

        # Reduce to occupied stuff
        occ_up = es_up.occupation(dist_up) * weight
        n[0] = einsum('i,ij->j', occ_up, es_up.norm2(False).real)
        occ_dn = es_dn.occupation(dist_dn) * weight
        n[1] = einsum('i,ij->j', occ_dn, es_dn.norm2(False).real)
        Etot = es_up.eig.dot(occ_up) + es_dn.eig.dot(occ_dn)

        # Return values
        return n, Etot

    # Loop k-points and weights
    for w, k in zip(H.mp.weight, H.mp.k):
        n, etot = calc_occ(k, w)
        ni += n
        Etot += etot

    return ni, Etot


def dm_insulator(H, q):
    """ Method to obtain the spin-densities only for the corner case for *insulators* at *T=0* """
    ni = np.zeros((2, H.sites))
    Etot = 0

    iup = np.arange(int(round(q[0])))
    idn = np.arange(int(round(q[1])))

    # Solve eigenvalue problems
    def calc_occ(k, weight):
        n = np.empty_like(ni)
        es_up = H.eigenstate(k, spin=0)
        es_dn = H.eigenstate(k, spin=1)

        n[0] = einsum('ij,ij->j', conj(es_up.state[iup]), es_up.state[iup]).real * weight
        n[1] = einsum('ij,ij->j', conj(es_dn.state[idn]), es_dn.state[idn]).real * weight

        # Calculate total energy
        Etot = (es_up.eig[iup].sum() + es_dn.eig[idn].sum()) * weight
        # Return values
        return n, Etot

    # Loop k-points and weights
    for w, k in zip(H.mp.weight, H.mp.k):
        n, etot = calc_occ(k, w)
        ni += n
        Etot += etot

    return ni, Etot
