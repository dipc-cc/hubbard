from hubbard import HubbardHamiltonian, sp2, ncsile
import sisl
import numpy as np

# Build sisl Geometry object only for a subset of atoms
molecule = sisl.get_sile('mol-ref/mol-ref.XV').read_geometry().sub([2,3,5])
molecule.sc.set_nsc([1, 1, 1])

# Build HubbardHamiltonian object
Hsp2 = sp2(molecule)
H = HubbardHamiltonian(Hsp2, U=3.5, Uij=np.ones((Hsp2.no, Hsp2.no)))
# Generate simple density
H.n = np.ones((2, H.sites))*0.5

print(f'1. Write and read densities under group U{H.U:.1f} using the HubbardHamiltonian class\n')
# Write density in file
H.write_density('mol-ref/test.HU.nc', group=f'U{H.U:.1f}', mode='w')
# Read density using the HubbardHamiltonian class
H.read_density('mol-ref/test.HU.nc', group=f'U{H.U:.1f}')

# Write another density in file under another group
H.n *= 2
H.U *= 2
print(f'2. Write another densities under group U{H.U:.1f}\n')
H.write_density('mol-ref/test.HU.nc', group=f'U{H.U:.1f}', mode='a')

print('3. Read density, U and kT using ncsile from all groups')
fh = sisl.get_sile('mol-ref/test.HU.nc', mode='r')
for g in fh.groups:
    print('group: ', g)
    print('n:', fh.read_density(group=g))
    print('U:', fh.read_U(group=g))
    print('kT:', fh.read_kT(group=g))
    print('\n')

print('4. Read using HubbardHamiltoninan class')
# Read density using the HubbardHamiltonian class with no group specified. It reads from the first one saved
H.read_density('mol-ref/test.HU.nc', group=None)
print('H.n:', H.n)

print('\n')

# Write another density in file under no group
print(f'5. Write and read another densities without group using the HubbardHamiltonian class\n')
H.n *= 4
H.write_density('mol-ref/test.HU.nc', group=None, mode='a')
# Read density using the HubbardHamiltonian class
H.read_density('mol-ref/test.HU.nc', group=None)
print('H.n:', H.n)
