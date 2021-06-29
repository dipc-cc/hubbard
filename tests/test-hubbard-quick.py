from hubbard import HubbardHamiltonian, sp2, density, NEGF
import numpy as np
import sisl

# Build sisl Geometry object
molecule = sisl.get_sile('mol-ref/mol-ref.XV').read_geometry()
molecule.sc.set_nsc([1, 1, 1])

print('1. Run one iteration with calc_n_insulator')
Hsp2 = sp2(molecule)
H = HubbardHamiltonian(Hsp2, U=3.5)
H.random_density()
dn = H.iterate(density.calc_n_insulator, mixer=sisl.mixing.LinearMixer())
print('   dn, Etot: ', dn, H.Etot, '\n')

print('2. Run one iteration with data from ncfile')
H.read_density('mol-ref/density.nc')
dn = H.iterate(density.calc_n_insulator, mixer=sisl.mixing.LinearMixer())
etot = 1 * H.Etot
print('   dn, Etot: ', dn, etot, '\n')

print('3. Run one iteration with calc_n')
d = H.iterate(density.calc_n, mixer=sisl.mixing.LinearMixer())
e = H.Etot
print('   dn, dEtot: ', d - dn, e - etot, '\n')

# Write fdf-block
print('\n4. Write initspin to fdf-block')
H.write_initspin('test.fdf', mode='w')

import random
print('5. Run one iteration for spin-degenerate calculation')
Hsp2 = sp2(molecule, spin='unpolarized')
H = HubbardHamiltonian(Hsp2, U=3.5, kT=0.025)
n = random.seed(10)
dn = H.iterate(density.calc_n)
print('   dn, Etot: ', dn, H.Etot, '\n')

print('6. Run one iteration for spin-degenerate calculation with NEGF')
Hsp2 = sp2(molecule, spin='unpolarized')
H = HubbardHamiltonian(Hsp2, U=3.5, kT=0.025)
n = random.seed(10)
negf = NEGF(H, [],[])
dn = H.iterate(negf.calc_n_open, qtol=1e-7)
print('   dn, Etot: ', dn, H.Etot, '\n')

# Write new data structure
print('7. Write data in ncfile')
H.write_density('mol-ref/test.nc', mode='w')
