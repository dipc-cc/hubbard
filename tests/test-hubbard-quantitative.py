import Hubbard.hamiltonian as hh
import Hubbard.density as dens
import Hubbard.sp2 as sp2
import numpy as np
import sisl

# Test quantitatively that densities and eigenvalue spectrum are unchanged
# using a reference molecule (already converged)

# Build sisl Geometry object
molecule = sisl.get_sile('mol-ref/mol-ref.XV').read_geometry()
molecule.sc.set_nsc([1,1,1])

# Try reading from file
Hsp2 = sp2(molecule)
H = hh.HubbardHamiltonian(Hsp2, U=3.5)
H.read_density('mol-ref/density.nc')
H.iterate(dens.occ_insulator)

# Determine reference values for the tests
ev0, evec0 = H.eigh(eigvals_only=False, spin=0)
Etot0 = H.Etot*1

for m in [dens.occ_insulator, dens.occ_insulator]:
    # Reset density and iterate
    H.random_density()

    dn = H.converge(dens.occ_insulator, tol=1e-10, steps=10)
    ev1, evec1 = H.eigh(eigvals_only=False, spin=0)

    # Total energy check:
    print('Total energy difference: %.4e eV' %(Etot0-H.Etot))

    # Eigenvalues are easy to check
    if np.allclose(ev1, ev0):
        print('Eigenvalue check passed')
    else:
        # Could be that up and down spins are interchanged
        print('Warning: Engenvalues for up-spins different. Checking down-spins instead')
        ev1, evec1 = H.eigh(eigvals_only=False, spin=1)
        if np.allclose(ev1, ev0):
            print('Eigenvalue check passed')

    # Eigenvectors are a little more tricky due to arbitrary sign
    if np.allclose(np.abs(evec1), np.abs(evec0)):
        print('Eigenvector check passed\n')
    else:
        print('Eigenvector check failed!!!\n')
