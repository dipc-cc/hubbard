import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.sp2 as sp2
import sisl

mol = sisl.get_sile('benzene.xyz').read_geometry()

# 1NN tight-binding model
geom = sp2(mol, t1=2.7, t2=0., t3=0.)

H = hh.HubbardHamiltonian(geom, U=0.)

ev, evec = H.eigh(eigvals_only=False, spin=0)

print('Eigenvalues:')
print(ev)

# Plot wavefunctions
for i in range(6):
    p = plot.Wavefunction(H, 500*evec[:, i], figsize=(10, 3))
    p.set_title('State %i' % i)
    p.annotate()
    p.savefig('state%i.pdf'%i)

bo = H.get_bond_order(format='csr')

print('Bond order:')
print(bo)
