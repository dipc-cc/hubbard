"""

:mod:`Hubbard.sp2`
==================

Function for the meanfield Hubbard Hamiltonian

.. currentmodule:: Hubbard.sp2

"""

from __future__ import print_function
import numpy as np
import sisl

def sp2(ext_geom, t1=2.7, t2=0.2, t3=0.18, eB=3., eN=-3.,
        kmesh=[1, 1, 1], s0=1.0, s1=0, s2=0, s3=0, dim=2):

    # Determine pz sites
    aux = []
    sp3 = []
    for ia in ext_geom:
        if ext_geom.atoms[ia].Z not in [5, 6, 7]:
            aux.append(ia)
        idx = ext_geom.close(ia, R=[0.1, 1.6])
        if len(idx[1])==4: # Search for atoms with 4 neighbors
            if ext_geom.atoms[ia].Z == 6:
                sp3.append(ia)

    # Remove all sites not carbon-type
    pi_geom = ext_geom.remove(aux+sp3)
    sites = len(pi_geom)
    print('Found %i pz sites' %sites)

    # Set pz orbital for each pz site
    r = np.linspace(0, 1.6, 700)
    func = 5 * np.exp(-r * 5)
    pz = sisl.SphericalOrbital(1, (r, func))
    for ia in pi_geom:
        pi_geom.atom[ia].orbital[0] = pz

    # Construct Hamiltonian
    if s1 != 0:
        orthogonal = False
    else:
        orthogonal = True
    H = sisl.Hamiltonian(pi_geom, orthogonal=orthogonal, dim=dim)

    # Radii defining 1st, 2nd, and 3rd neighbors
    R = [0.1, 1.6, 2.6, 3.1]
    # Build hamiltonian for backbone

    for ia in pi_geom:
        idx = pi_geom.close(ia, R=R)
        # NB: I found that ':' is necessary in the following lines, but I don't understand why...
        if pi_geom.atoms[ia].Z == 5:
            H[ia, ia, :] = eB # set onsite for B sites
        elif pi_geom.atoms[ia].Z == 7:
            H[ia, ia, :] = eN # set onsite for N sites
        # set hoppings
        H[ia, idx[1], :] = -t1
        if t2 != 0:
            H[ia, idx[2], :] = -t2
        if t3 != 0:
            H[ia, idx[3], :] = -t3
        if not H.orthogonal:
            H.S[ia, ia] = s0
            H.S[ia, idx[1]] = s1
            H.S[ia, idx[2]] = s2
            H.S[ia, idx[3]] = s3

    return H
