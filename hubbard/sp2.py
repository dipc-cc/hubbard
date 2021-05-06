import numpy as np
import sisl

__all__ = ['sp2']


def sp2(ext_geom, t1=2.7, t2=0.2, t3=0.18, eB=3., eN=-3.,
        s0=1.0, s1=0, s2=0, s3=0, dq=0, dim=2):
    """ Function to create a Tight Binding Hamiltoninan for sp2 Carbon systems

    It takes advantage of the `sisl` class for building sparse Hamiltonian matrices,
    `sisl.physics.Hamiltonian`

    It obtains the Hamiltonian for ``ext_geom`` (which must be a `sisl.Geometry` instance)
    with the parameters for first, second and third nearest neighbors (``t1``, ``t2``, ``t3``).

    One can also use a non-orthogonal basis of atomic orbitals by passing the parameters for the
    overlap matrix between first, second and third nearest neighbors (``s1``, ``s2``, ``s3``).

    The function will also take into account the possible presence of Boron or Nitrogen atoms,
    for which one would need to specify the on-site energy for those atoms (``eB`` and ``eN``)

    Returns
    -------
    H: sisl.physics.Hamiltonian
        tight-binding Hamiltonian for the sp2 structure of ``dim=2`` (for the two spin channels)
    """

    # Determine pz sites
    aux = []
    sp3 = []
    for ia, atom in enumerate(ext_geom.atoms.iter()):
        # Append non C-type atoms in aux list
        if atom.Z not in [5, 6, 7]:
            aux.append(ia)
        idx = ext_geom.close(ia, R=[0.1, 1.6])
        if len(idx[1]) == 4: # Search for atoms with 4 neighbors
            if atom.Z == 6:
                sp3.append(ia)

    # Remove all sites not carbon-type
    pi_geom = ext_geom.remove(aux + sp3)
    pi_geom.reduce()

    # Iterate over atomic species to set initial charge
    maxR = 20
    r = np.linspace(0, maxR, 700)
    # In Slater-type orbitals (Hydrogen-like atom solution), the radial function is ~exp(-Zr/2a)
    # where a=0.529 \AA is the Bohr radius and Z is the atomic number.
    # We use the effective nuclear charge instead, which for Carbon atoms is approximately Zeff~3.
    func =  np.exp(-3*r)
    for atom, _ in pi_geom.atoms.iter(True):
        pz = sisl.AtomicOrbital('pz', (r, func), R=maxR, q0=atom.Z-5+dq)
        atom.orbitals[0] = pz

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
