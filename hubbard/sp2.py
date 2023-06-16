import numpy as np
import sisl

__all__ = ['sp2']


def sp2(ext_geom, t1=2.7, t2=0.2, t3=0.18, eB=3., eN=-3.,
        s1=0, s2=0, s3=0, dq=0, spin=sisl.physics.Spin('polarized'), atoms=None):
    """ Function to create a Tight Binding Hamiltoninan for sp2 Carbon systems

    It takes advantage of the `sisl` class for building sparse Hamiltonian matrices,
    `sisl.physics.Hamiltonian`

    It obtains the Hamiltonian for ``ext_geom`` (which must be a `sisl.Geometry` instance)
    with the parameters for first, second and third nearest neighbors (``t1``, ``t2``, ``t3``).

    One can also use a non-orthogonal basis of atomic orbitals by passing the parameters for the
    overlap matrix between first, second and third nearest neighbors (``s1``, ``s2``, ``s3``).

    The function will also take into account the possible presence of Boron or Nitrogen atoms,
    for which one would need to specify the on-site energy for those atoms (``eB`` and ``eN``)

    If no atoms are passed it will assign to each sp2 atom a `pz` hydrogen-like orbital
    (i.e., `sisl.HydrogenicOrbital` with effective nuclear charge `Zeff=3.2`).
    This can be modified by the user with the argument ``atoms``

    Parameters
    ----------
    ext_geom: sisl.Geometry
        geometry of the sp2 carbon system
    t1: float, optional
        1NN hopping defaults to 2.7 eV
    t2: float, optional
        2NN hopping defaults to 0.2 eV
    t3: float, optional
        3NN hopping defaults to 0.18 eV
    eB: float, optional
        on-site energy for Boron atoms
    eN: float, optional
        on-site energy for Nitrogen atoms
    s1: float, optional
        overlap between 1NN, default to zero (orthogonal basis)
    s2: float, optional
        overlap between 2NN, default to zero
    s3: float, optional
        overlap between 3NN, default to zero
    dq: float, optional
        additional atomic charge, defaults to zero
    spin: str or sisl.physics.Spin, optional
        to define a polarized or unpolarized system pass ``spin=polarized`` or ``spin=unpolarized``
        or the corresponding `sisl.physics.Spin` object
    atoms: sisl.Atom or sisl.Atoms, optional
        Atoms for the geometry containing orbital information. It can be a `sisl.Atoms` instance containing different `sisl.Atom` objects.
        In this case it should contain all atoms of the sp2 geometry (i.e. `Z in [5,6,7]`)

    Returns
    -------
    H: sisl.physics.Hamiltonian
        tight-binding Hamiltonian for the sp2 structure
    """
    if isinstance(spin,str):
        spin = sisl.physics.Spin(spin)

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
    for atom, ia in pi_geom.atoms.iter(True):

        # Check if there are orbitals saved in the geometry
        if not all(isinstance(orb, (sisl.AtomicOrbital, sisl.HydrogenicOrbital)) for orb in atom.orbitals):
            if atoms is None:
                # Default atomic orbitals is pz orbitals
                pz = sisl.HydrogenicOrbital(2, 1, 0, 3.2, q0=atom.Z-5+dq, R=10.)
                pi_geom.atoms.replace_atom(atom, sisl.Atom(atom.Z, orbitals=pz))
            else:
                atoms = sisl.Atoms(atoms)
                try:
                    atom_ = list(filter(lambda x: (x.Z == atom.Z), atoms))[0]
                except:
                    raise ValueError(f'Atom with Z={atom.Z} not found')
                pi_geom.atoms.replace_atom(atom, atom_)

    # Construct Hamiltonian
    if s1 != 0:
        orthogonal = False
    else:
        orthogonal = True
    H = sisl.Hamiltonian(pi_geom, orthogonal=orthogonal, spin=spin)

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
            H.S[ia, ia] = 1.0
            H.S[ia, idx[1]] = s1
            H.S[ia, idx[2]] = s2
            H.S[ia, idx[3]] = s3

    return H
