from __future__ import print_function

import Hubbard.hamiltonian as hh
import Hubbard.ncdf as ncdf
import sisl

# Build sisl Geometry object
molecule = sisl.get_sile('mol-ref/mol-ref.XV').read_geometry()
molecule.sc.set_nsc([1,1,1])

# Run one iteration
H = hh.HubbardHamiltonian(molecule, U=3.5)
H.random_density()
dn = H.iterate(mix=.1)
print(dn, H.Etot)

# Run also one iteration with data from ncfile
H.read_density('mol-ref/density.nc')
dn = H.iterate(mix=1)
etot = 1*H.Etot
print(dn, etot)

for d in [H.iterate2(mix=1), H.iterate3(mix=1)]:
    e = H.Etot
    print(d-dn, e-etot)

# Write new data structure
H.write_density('mol-ref/test.nc', mode='w')
