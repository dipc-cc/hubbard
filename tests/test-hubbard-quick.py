from __future__ import print_function

import Hubbard.hamiltonian as hh
import Hubbard.sp2 as sp2
import Hubbard.density as dm
import sisl

# Build sisl Geometry object
molecule = sisl.get_sile('mol-ref/mol-ref.XV').read_geometry()
molecule.sc.set_nsc([1, 1, 1])

# Run one iteration
Hsp2 = sp2(molecule)
H = hh.HubbardHamiltonian(Hsp2, U=3.5)
H.random_density()
mixer = sisl.mixing.PulayMixer(0.7, history=7)
dn = H.iterate(dm.dm_insulator, mixer=mixer)
print(dn, H.Etot)

# Run also one iteration with data from ncfile
H.read_density('mol-ref/density.nc')
dn = H.iterate(dm.dm_insulator, mixer=mixer)
etot = 1*H.Etot
print(dn, etot)

# Test iterate2 method
d = H.iterate(dm.dm, mixer=mixer)
e = H.Etot
print(d-dn, e-etot)

# Write new data structure
H.write_density('mol-ref/test.nc', mode='w')
