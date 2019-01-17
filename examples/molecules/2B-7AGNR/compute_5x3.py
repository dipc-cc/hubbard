import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import sys
import numpy as np
import sisl

pol = False

# Build sisl Geometry object
fn = sisl.get_sile('7AGNR2B_5x3.XV').read_geometry()
fn.sc.set_nsc([1,1,1])
fn = fn.move(-fn.center(what='xyz'))

grp = 'AFM-AFM-AFM'
H = hh.HubbardHamiltonian(fn, fn_title='7AGNR2B_5x3', t1=2.7, t2=0.2, t3=.18, U=5.0, ncgroup=grp)
if pol:
    H.set_polarization([1, 99], dn=[80, 152])
dn = H.converge()
H.save(grp)
p = plot.SpinPolarization(H)
p.annotate()
p.savefig('pol_5x3_%s.pdf'%grp)

p = plot.LDOSmap(H)
p.savefig('pol_5x3_%s_ldos.pdf'%grp)

grp = 'AFM-FM-AFM'
H = hh.HubbardHamiltonian(fn, fn_title='7AGNR2B_5x3', t1=2.7, t2=0.2, t3=.18, U=5.0, ncgroup=grp)
if pol:
    H.set_polarization([1, 152], dn=[80, 99])
dn = H.converge()
H.save(grp)
p = plot.SpinPolarization(H)
p.annotate()
p.savefig('pol_5x3_%s.pdf'%grp)

grp = 'FM-AFM-FM'
H = hh.HubbardHamiltonian(fn, fn_title='7AGNR2B_5x3', t1=2.7, t2=0.2, t3=.18, U=5.0, ncgroup=grp)
if pol:
    H.set_polarization([1, 80], dn=[99, 152])
dn = H.converge()
H.save(grp)
p = plot.SpinPolarization(H)
p.annotate()
p.savefig('pol_5x3_%s.pdf'%grp)
