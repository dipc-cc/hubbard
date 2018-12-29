from __future__ import print_function

import Hubbard.hamiltonian as hh
import sisl

# Build sisl Geometry object
fn = sisl.get_sile('mol-ref/mol-ref.XV').read_geometry()
fn.sc.set_nsc([1,1,1])

# Run one iteration
H = hh.HubbardHamiltonian(fn, fn_title='mol-ref/mol-ref', U=5.0)
dn, etot = H.iterate(mix=.1)
print(dn, etot)

# Run also one iteration with data from ncfile
H = hh.HubbardHamiltonian(fn, fn_title='mol-ref/mol-ref', U=3.5)
dn, etot = H.iterate(mix=1)

for d, e in [H.iterate2(mix=1), H.iterate3(mix=1)]:
    print(d-dn, e-etot)
