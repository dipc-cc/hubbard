from hubbard import HubbardHamiltonian, sp2, density
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

# Write new data structure
print('4. Write data in ncfile')
H.write_density('mol-ref/test.nc', mode='w')

# Write fdf-block
print('\n5. Write initspin to fdf-block')
H.write_initspin('test.fdf', mode='w')
