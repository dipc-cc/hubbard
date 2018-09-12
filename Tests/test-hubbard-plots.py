import Hubbard.hubbard as HH
import numpy as np

# Test all plot functionalities of Hubbard module
# using a reference molecule (already converged)

H = HH.Hubbard('mol-ref/mol-ref.XV', U=3.5)

# Produce plots
H.plot_polarization()
H.plot_rs_polarization()
H.plot_wf(EnWindow=0.25, ispin=0)
H.plot_wf(EnWindow=0.25, ispin=1)
H.plot_wf(EnWindow=0.25, ispin=1, density=False)
H.plot_rs_wf(EnWindow=0.25, ispin=0, z=0.5)
H.plot_rs_wf(EnWindow=0.25, ispin=1, z=0.5)
H.plot_rs_wf(EnWindow=0.25, ispin=1, z=0.5, density=False)

# Test also quantitatively that densities and eigenvalue spectrum are unchanged
# Compute reference values:
ev0, evec0 = H.Hup.eigh(eigvals_only=False)
Etot0 = H.Etot*1

# Reset density and iterate
H.random_density()
deltaN = 1.0

while deltaN > 1e-10:
    if deltaN > 0.1:
        # preconditioning
        deltaN, etot = H.iterate(mix=.1)
    else:
        deltaN, etot = H.iterate(mix=1)

ev1, evec1 = H.Hup.eigh(eigvals_only=False)

# Eigenvalues are easy to check
if np.allclose(ev1, ev0):
    print 'Eigenvalue check passed!'

# Eigenvectors are a little more tricky due to arbitrary sign
if np.allclose(np.abs(evec1), np.abs(evec0)):
    print 'Eigenvector check passed!'

# Total energy check:
print Etot0, etot, Etot0-etot
