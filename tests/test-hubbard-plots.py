import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import numpy as np

# Test all plot functionalities of Hubbard module
# using a reference molecule (already converged)

H = hh.HubbardHamiltonian('mol-ref/mol-ref.XV', U=3.5, what='xyz')

# Old plotting routines
if False:
    H.plot_spectrum()
    H.plot_charge()
    H.plot_polarization()
    H.plot_rs_polarization()
    H.plot_wf(EnWindow=0.25, ispin=0)
    H.plot_wf(EnWindow=0.25, ispin=1)
    H.plot_wf(EnWindow=0.25, ispin=1, density=False)
    H.plot_rs_wf(EnWindow=0.25, ispin=0, z=0.5)
    H.plot_rs_wf(EnWindow=0.25, ispin=1, z=0.5)
    H.plot_rs_wf(EnWindow=0.25, ispin=1, z=0.5, density=False)

# new routines
if True:
    p = plot.Charge(H)
    p.savefig('chg.pdf')

    p = plot.ChargeDifference(H)
    p.savefig('chgdiff.pdf')

    p = plot.SpinPolarization(H)
    p.annotate()
    p.savefig('pol.pdf')

    ev, evec = H.eigh(eigvals_only=False, spin=0)
    p = plot.Wavefunction(H, 500*evec[:, 10])
    p.savefig('wf.pdf')

    p = plot.Spectrum(H)
    p.savefig('spectrum.pdf')
