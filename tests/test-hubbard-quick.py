from __future__ import print_function

import Hubbard.hamiltonian as hh
import sisl

# Build sisl Geometry object
fn = sisl.get_sile('mol-ref/mol-ref.XV').read_geometry()
fn.sc.set_nsc([1,1,1])

# Run one iteration
H = hh.HubbardHamiltonian(fn, fn_title='mol-ref/mol-ref', U=5.0)
dn = H.iterate(mix=.1)
print(dn, H.Etot)

# Run also one iteration with data from ncfile
H = hh.HubbardHamiltonian(fn, fn_title='mol-ref/mol-ref', U=3.5)
dn = H.iterate(mix=1)
etot = 1*H.Etot

for d in [H.iterate2(mix=1), H.iterate3(mix=1)]:
    e = H.Etot
    print(d-dn, e-etot)
