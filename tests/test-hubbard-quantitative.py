import Hubbard.hamiltonian as hh
import numpy as np
import sisl

# Test quantitatively that densities and eigenvalue spectrum are unchanged
# using a reference molecule (already converged)

# Build sisl Geometry object
fn = sisl.get_sile('mol-ref/mol-ref.XV').read_geom()
fn.sc.set_nsc([1,1,1])

H = hh.HubbardHamiltonian(fn, fn_title='mol-ref/mol-ref', U=3.5)

# Determine reference values for the tests
ev0, evec0 = H.eigh(eigvals_only=False, spin=0)

Etot0 = H.Etot*1

# Reset density and iterate
H.random_density()
dn, etot = H.converge(tol=1e-10, steps=10)
ev1, evec1 = H.eigh(eigvals_only=False, spin=0)

# Total energy check:
print 'Total energy difference: %.4e eV' %(Etot0-etot)

# Eigenvalues are easy to check
if np.allclose(ev1, ev0):
    print 'Eigenvalue check passed'
else:
    # Could be that up and down spins are interchanged
    print 'Warning: Engenvalues for up-spins different. Checking down-spins instead'
    ev1, evec1 = H.eigh(eigvals_only=False, spin=1)
    if np.allclose(ev1, ev0):
        print 'Eigenvalue check passed'

# Eigenvectors are a little more tricky due to arbitrary sign
if np.allclose(np.abs(evec1), np.abs(evec0)):
    print 'Eigenvector check passed'
else:
    print 'Eigenvector check failed!!!'
