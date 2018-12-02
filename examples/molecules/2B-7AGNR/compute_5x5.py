import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import sys
import numpy as np
import sisl

pol = False

# Build sisl Geometry object
fn = sisl.get_sile('7AGNR2B_5x5.XV').read_geometry()
fn.sc.set_nsc([1,1,1])
fn = fn.move(-fn.center(what='xyz'))

grp = 'AFM-AFM-AFM'
H = hh.HubbardHamiltonian(fn, fn_title='7AGNR2B_5x5', t1=2.7, t2=0.2, t3=.18, U=5.0, ncgroup=grp)
if pol:
    H.set_polarization([3, 99], dn=[82, 178])
dn, etot = H.converge()
H.save(grp)
p = plot.SpinPolarization(H)
p.annotate()
p.savefig('pol_5x5_%s.pdf'%grp)

p = plot.LDOSmap(H)
p.savefig('pol_5x5_%s_ldos.pdf'%grp)

grp = 'AFM-FM-AFM'
H = hh.HubbardHamiltonian(fn, fn_title='7AGNR2B_5x5', t1=2.7, t2=0.2, t3=.18, U=5.0, ncgroup=grp)
if pol:
    H.set_polarization([3, 178], dn=[82, 99])
dn, etot = H.converge()
H.save(grp)
p = plot.SpinPolarization(H)
p.annotate()
p.savefig('pol_5x5_%s.pdf'%grp)

grp = 'FM-AFM-FM'
H = hh.HubbardHamiltonian(fn, fn_title='7AGNR2B_5x5', t1=2.7, t2=0.2, t3=.18, U=5.0, ncgroup=grp)
if pol:
    H.set_polarization([3, 82], dn=[99, 178])
dn, etot = H.converge()
H.save(grp)
p = plot.SpinPolarization(H)
p.annotate()
p.savefig('pol_5x5_%s.pdf'%grp)
