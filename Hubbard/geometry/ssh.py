"""

:mod:`Hubbard.geometry`
==========================

Function to generate the SSH-chain model

.. currentmodule:: Hubbard.geometry

"""

from __future__ import print_function
import numpy as np
import sisl

def ssh(d1=1.0, d2=1.5, dy=10):
    xyz = [[0.,0.,0.],[d1, 0, 0]]
    # Create supercell and geometry
    sc = sisl.SuperCell([d1+d2,dy,10,90,90,90],nsc=[3,1,1])
    uc = sisl.Geometry(xyz, atom=sisl.Atom(Z=6), sc=sc)
    # Iterate over atomic species to set initial charge
    r = np.linspace(0, 1.6, 700)
    func = 5 * np.exp(-r * 5)
    for atom, _ in uc.atoms.iter(True):
        pz = sisl.AtomicOrbital('pz', (r, func), q0=atom.Z-5)
        atom.orbital[0] = pz
    uc = uc.move(-uc.center(what='xyz'))
    return uc
