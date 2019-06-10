import Hubbard.HubbardSCF as HubbardSCF
import Hubbard.sp2 as sp2
import Hubbard.ncdf as ncdf
import numpy as np
import sisl

# Test quantitatively that densities and eigenvalue spectrum are unchanged
# using a reference molecule (already converged)

# Build sisl Geometry object
molecule = sisl.get_sile('mol-ref/mol-ref.XV').read_geometry()
molecule.sc.set_nsc([1,1,1])

# Try reading from file
calc = ncdf.read('mol-ref/mol-ref.nc')
# Build Hamiltonian of sp2 carbon system
Hsp2 = sp2(molecule, dim=2)
H = HubbardSCF(Hsp2.H, U=3.5)
H.U = calc.U
H.Nup, H.Ndn = calc.Nup, calc.Ndn
H.nup, H.ndn = calc.nup, calc.ndn
H.update_hamiltonian()

# Determine reference values for the tests
ev0, evec0 = H.eigh(eigvals_only=False, spin=0)
Etot0 = calc.Etot*1

for m in range(1,4):
    # Reset density and iterate
    H.random_density()

    dn = H.converge(tol=1e-10, steps=10, method=m)
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
