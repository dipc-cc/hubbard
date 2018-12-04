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

    p = plot.LDOSmap(H)
    p.savefig('ldos.pdf')
