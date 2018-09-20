import Hubbard.hubbard as HH
import numpy as np

# Test quantitatively that densities and eigenvalue spectrum are unchanged
# using a reference molecule (already converged)
H = HH.Hubbard('mol-ref/mol-ref.XV', U=3.5)

# Determine reference values for the tests
ev0, evec0 = H.Hup.eigh(eigvals_only=False)

Etot0 = H.Etot*1

# Reset density and iterate
H.random_density()
deltaN = 1.0
i = 0
print 'Iterations / deltaN / etot / midgap energy reference'
while deltaN > 1e-10:
    if deltaN > 1e-2:
        # preconditioning
        deltaN, etot = H.iterate(mix=.1)
    else:
        deltaN, etot = H.iterate(mix=1)
    i += 1
    if i%10 == 0:
        print i, deltaN, etot, H.midgap
    
ev1, evec1 = H.Hup.eigh(eigvals_only=False)

# Total energy check:
print 'Total energy difference: %.4e eV' %(Etot0-etot)

# Eigenvalues are easy to check
if np.allclose(ev1, ev0):
    print 'Eigenvalue check passed'
else:
    # Could be that up and down spins are interchanged
    print 'Warning: Engenvalues for up-spins different. Checking down-spins instead'
    ev1, evec1 = H.Hdn.eigh(eigvals_only=False)
    if np.allclose(ev1, ev0):
        print 'Eigenvalue check passed'

# Eigenvectors are a little more tricky due to arbitrary sign
if np.allclose(np.abs(evec1), np.abs(evec0)):
    print 'Eigenvector check passed'
else:
    print 'Eigenvector check failed!!!'
