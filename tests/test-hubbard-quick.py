from __future__ import print_function

import Hubbard.hamiltonian as hh
import Hubbard.ncdf as ncdf
import sisl

# Build sisl Geometry object
molecule = sisl.get_sile('mol-ref/mol-ref.XV').read_geometry()
molecule.sc.set_nsc([1,1,1])

# Run one iteration
H = hh.HubbardHamiltonian(molecule, U=5.0)
H.random_density()
dn = H.iterate(mix=.1)
print(dn, H.Etot)

# Run also one iteration with data from ncfile
calc = ncdf.read('mol-ref/mol-ref.nc')
H = hh.HubbardHamiltonian(molecule)
H.U = calc.U
H.Nup, H.Ndn = calc.Nup, calc.Ndn
H.nup, H.ndn = calc.nup, calc.ndn
H.update_hamiltonian()
dn = H.iterate(mix=1)
etot = 1*H.Etot
print(dn, etot-calc.Etot)

for d in [H.iterate2(mix=1), H.iterate3(mix=1)]:
    e = H.Etot
    print(d-dn, e-etot)

# Write new data structure
H.write_density('mol-ref/density.nc', mode='w')

H.read_density('mol-ref/density.nc')
for d in [H.iterate2(mix=1), H.iterate3(mix=1)]:
    e = H.Etot
    print(d-dn, e-etot)
