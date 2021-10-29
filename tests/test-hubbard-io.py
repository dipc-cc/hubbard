from hubbard import HubbardHamiltonian, sp2, ncsile
import sisl
import numpy as np

# Build sisl Geometry object only for a subset of atoms
molecule = sisl.get_sile('mol-ref/mol-ref.XV').read_geometry().sub([2,3,5])
molecule.sc.set_nsc([1, 1, 1])

# Build HubbardHamiltonian object
Hsp2 = sp2(molecule)
H = HubbardHamiltonian(Hsp2, U=3.5)
# Generate simple density
H.n = np.ones((2, H.sites))*0.5

print(f'1. Write and read densities under group {H.get_hash()} using the HubbardHamiltonian class\n')
# Write density in file
H.write_density('mol-ref/test.nc', group=H.get_hash(), mode='w')
# Read density using the HubbardHamiltonian class
H.read_density('mol-ref/test.nc', group=H.get_hash())

# Write another density in file under another group
print(f'2. Write another densities under another group\n')
H.n *= 2
H.write_density('mol-ref/test.nc', group='group2', mode='a')


print('3. Read density, U and kT using ncsile')
# Since there are two groups in the file at this point and no one is specified it reads everything
fh = ncsile.ncSilehubbard('mol-ref/test.nc', mode='r')
print('n: ', fh.read_density(group=None))
print('U: ', fh.read_U(group=None))
print('kT: ', fh.read_kT(group=None))

print('\n')

print('4. Read using HubbardHamiltoninan class')
# Read density using the HubbardHamiltonian class
H.read_density('mol-ref/test.nc', group=None)
print('H.n:', H.n)

print('\n')

# Write another density in file under no group
print(f'5. Write and read another densities without group using the HubbardHamiltonian class\n')
H.n *= 4
H.write_density('mol-ref/test.nc', group=None, mode='a')
# Read density using the HubbardHamiltonian class
H.read_density('mol-ref/test.nc', group=None)
print('H.n:', H.n)


'''
# These lines don't work for some reason when the file is in another folder,
# but when the folder path is the same as this script then it works....
# Add sile and use sisl to read density, U and kT
sisl.io.add_sile('mol-ref/test.nc', hubbard.ncsile.ncSilehubbard, gzip=False)
print('n: ', sisl.get_sile('mol-ref/test.nc').read_density(group='random'))
print('U: ',sisl.get_sile('mol-ref/test.nc').read_U(group='random'))
print('kT: ',sisl.get_sile('mol-ref/test.nc').read_kT(group='random'))
'''