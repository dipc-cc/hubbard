import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import numpy as np
import sisl

# Test all plot functionalities of Hubbard module
# using a reference molecule (already converged)

# Build sisl Geometry object
fn = sisl.get_sile('mol-ref/mol-ref.XV').read_geometry()
fn.sc.set_nsc([1,1,1])
fn = fn.move(-fn.center(what='xyz')).rotate(220, [0,0,1])

H = hh.HubbardHamiltonian(fn, fn_title='mol-ref/mol-ref', U=3.5)

# new routines
if True:
    p = plot.Charge(H, colorbar=True)
    p.savefig('chg.pdf')

    p = plot.Charge(H, realspace=True, colorbar=True)
    p.savefig('chg_rs.pdf')

    p = plot.ChargeDifference(H)
    p.savefig('chgdiff.pdf')

    p = plot.ChargeDifference(H, realspace=True)
    p.savefig('chgdiff_rs.pdf')

    p = plot.SpinPolarization(H)
    p.annotate()
    p.savefig('pol.pdf')

    p = plot.SpinPolarization(H, realspace=True)
    p.annotate()
    p.savefig('pol_rs.pdf')

    ev, evec = H.eigh(eigvals_only=False, spin=0)
    p = plot.Wavefunction(H, 500*evec[:, 10])
    p.savefig('wf.pdf')

    ev, evec = H.eigh(eigvals_only=False, spin=0)
    p = plot.Wavefunction(H, 10*evec[:, 10], realspace=True)
    p.savefig('wf_rs.pdf')

    p = plot.Spectrum(H)
    p.savefig('spectrum.pdf')

    p = plot.LDOSmap(H)
    p.savefig('ldos.pdf')
