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

# Eigenvalues are easy to check
if np.allclose(ev1, ev0):
    print 'Eigenvalue check passed'
else:
    raiseError('Eigenvalue check failed!!!')
    
# Eigenvectors are a little more tricky due to arbitrary sign
if np.allclose(np.abs(evec1), np.abs(evec0)):
    print 'Eigenvector check passed'
else:
    raiseError('Eigenvector check failed!!!')
    
# Total energy check:
print Etot0, etot, Etot0-etot
