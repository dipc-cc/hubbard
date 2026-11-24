import numpy as np
from numpy import einsum, conj
import sisl

__all__ = ['calc_n', 'calc_n_insulator']


def calc_n(H, q):
    r""" General method to obtain the spin densities for periodic or finite systems at a given temperature

    It obtains the spin densities from the direct diagonalization of the Hamiltonian (``H.H``) taking into account
    a possible overlap matrix (``H.H.S``):

    .. math::
        \langle n_{i\sigma} \rangle = \sum_{\alpha}f_{\alpha\sigma}\sum_{j}c^{\alpha}_{i\sigma}c^{*\alpha}_{j\sigma}S_{ij}

    Where :math:`f_{\alpha\sigma}` is the weight of eigenstate :math:`\alpha` for spin :math:`\sigma` at temperature ``kT`` (Fermi-Dirac distribution),
    :math:`c^{\alpha}_{i\sigma}` are coefficients for eigenstate :math:`\alpha` with spin :math:`\sigma` represented in the basis of atomic orbitals

    Parameters
    ----------
    H: HubbardHamiltonian
        `hubbard.HubbardHamiltonian` object of the system to obtain the spin-densities from
    q: array_like
        charge resolved in spin channels (first index for up-electrons and second index for down-electrons)

    See Also
    ------------
    sisl.physics.electron.EigenstateElectron.norm2: sisl routine to obtain the dot product of the eigenstates with the overlap matrix
    """
    # Create fermi-level determination distribution
    dist = sisl.get_distribution('fermi_dirac', smearing=H.kT)
    Ef = H.H.fermi_level(H.mp, q=q, distribution=dist)
    if not isinstance(Ef, (tuple, list, np.ndarray)):
        Ef = np.array([Ef])
    dist = [sisl.get_distribution('fermi_dirac', smearing=H.kT, x0=Ef[s]) for s in range(H.spin_size)]

    ni = np.zeros((H.spin_size, H.sites))
    Etot = 0

    # Solve eigenvalue problems
    def calc_occ(k, weight, spin):
        es = H.eigenstate(k, spin=spin)

        # Reduce to occupied stuff
        occ = es.occupation(dist[spin]) * weight
        n = einsum('i,ij->j', occ, es.norm2(projection='orbitals').real)

        Etot = es.eig.dot(occ)

        # Return values
        return n, Etot

    # Loop k-points and weights
    for w, k in zip(H.mp.weight, H.mp.k):
        for s in range(H.spin_size):
            n, etot = calc_occ(k, w, s)
            ni[s] += n
            Etot += etot

    # Return spin densities and total energy
    # if the Hamiltonian is not spin-polarized multiply Etot by 2 for spin degeneracy
    return ni, (2./H.spin_size)*Etot


def calc_n_insulator(H, q):
    r""" Method to obtain the spin densities only for the corner case for *insulators* at *T=0*

    For each spin channel and k-point it occupies the lowest ``round(q[s])`` eigenstates of
    ``H.H`` and leaves the rest empty, which is correct only when each spin channel has a gap
    at its Fermi level. For a non-orthogonal basis the occupations are Mulliken populations,
    as in `calc_n`:

    .. math::
        \langle n_{i\sigma} \rangle = \sum_{\mathbf k}w_{\mathbf k}\sum_{\alpha\ \mathrm{occ}}\mathrm{Re}\Big[c^{*\alpha}_{i\sigma\mathbf k}\sum_{j}S_{ij}(\mathbf k)c^{\alpha}_{j\sigma\mathbf k}\Big]

    with :math:`w_{\mathbf k}` the k-point weights of ``H.mp``, so they sum to ``round(q[s])`` per spin (per unit cell for a periodic system).

    Parameters
    ----------
    H: HubbardHamiltonian
        `hubbard.HubbardHamiltonian` object of the system to obtain the spin-densities from
    q: array_like
        charge resolved in spin channels (first index for up-electrons and second index for down-electrons),
        rounded to the nearest integer

    Returns
    -------
    n: numpy.ndarray
        occupations per orbital, of shape ``(H.spin_size, H.sites)``
    Etot: float
        sum of the occupied eigenvalues in eV, weighted by the k-point weights, and doubled
        for spin degeneracy if ``H`` is not spin-polarized
    """
    ni = np.zeros((H.spin_size, H.sites))
    Etot = 0

    idx = [np.arange(int(round(q[s]))) for s in range(H.spin_size)]

    # Solve eigenvalue problems
    def calc_occ(k, weight, spin):
        es = H.eigenstate(k, spin=spin)

        c = es.state[idx[spin]]
        if H.H.orthogonal:
            Sc = c
        else:
            # Mulliken populations, with S(k) in the gauge of the eigenstates
            Sc = (es.Sk() @ c.T).T
        n = einsum('ij,ij->j', conj(c), Sc).real * weight

        # Calculate total energy
        Etot = es.eig[idx[spin]].sum() * weight
        # Return values
        return n, Etot

    # Loop k-points and weights
    for w, k in zip(H.mp.weight, H.mp.k):
        for s in range(H.spin_size):
            n, etot = calc_occ(k, w, s)
            ni[s] += n
            Etot += etot

    return ni, (2./H.spin_size)*Etot
