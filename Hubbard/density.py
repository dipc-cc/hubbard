from __future__ import print_function
import numpy as np
import sisl

def occ(H, q_up, q_dn):
    """ General method to obtain the occupations for periodic or finite systems at a certain temperature"""
    # Create fermi-level determination distribution
    dist = sisl.get_distribution('fermi_dirac', smearing=H.kT)
    Ef = H.H.fermi_level(H.mp, q=[q_up, q_dn], distribution=dist)
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
        occ_up = es_up.occupation(dist_up).reshape(-1, 1) * weight
        n[0] = (es_up.norm2(False).real * occ_up).sum(0)
        occ_dn = es_dn.occupation(dist_dn).reshape(-1, 1) * weight
        n[1] = (es_dn.norm2(False).real * occ_dn).sum(0)
        Etot = (es_up.eig * occ_up.ravel()).sum() + (es_dn.eig * occ_dn.ravel()).sum()

        # Return values
        return n, Etot

    # Loop k-points and weights
    for w, k in zip(H.mp.weight, H.mp.k):
        n, etot = calc_occ(k, w)
        ni += n
        Etot += etot

    return ni, Etot

def occ_insulator(H, q_up, q_dn):
    """ Method to obtain the occupations only for the corner case for insulators at T=0 """
    ni = np.zeros((2, H.sites))
    Etot = 0

    # Solve eigenvalue problems
    def calc_occ(k, weight):
        n = np.empty_like(ni)
        es_up = H.eigenstate(k, spin=0)
        es_dn = H.eigenstate(k, spin=1)

        es_up = es_up.sub(range(q_up))
        es_dn = es_dn.sub(range(q_dn))

        n[0] = (es_up.norm2(False).real).sum(0) * weight
        n[1] = (es_dn.norm2(False).real).sum(0) * weight

        # Calculate total energy
        Etot = (es_up.eig.sum() + es_dn.eig.sum()) * weight
        # Return values
        return n, Etot

    # Loop k-points and weights
    for w, k in zip(H.mp.weight, H.mp.k):
        n, etot = calc_occ(k, w)
        ni += n
        Etot += etot

    return ni, Etot
