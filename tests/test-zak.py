from hubbard import HubbardHamiltonian, sp2
import numpy as np
import sisl

for w in range(1, 25, 2):
    g = sisl.geom.agnr(w)
    H0 = sp2(g)
    H = HubbardHamiltonian(H0, U=0)
    zak = H.get_Zak_phase()
    print(f'width={w:3}, zak={zak:7.3f}')

# SSH model, topological cell
g = sisl.Geometry([[0, 0, 0], [0, 1.65, 0]], sisl.Atom(6, 1.001), lattice=[10, 3, 10])
g.set_nsc([1, 3, 1])
H0 = sp2(g)
H = HubbardHamiltonian(H0, U=0)
zak = H.get_Zak_phase(axis=1)
print(f'SSH topo : zak={zak:7.3f}')

# SSH model, trivial cell
g = sisl.Geometry([[0, 0, 0], [0, 1.42, 0]], sisl.Atom(6, 1.001), lattice=[10, 3, 10])
g.set_nsc([1, 3, 1])
H0 = sp2(g)
H = HubbardHamiltonian(H0, U=0)
zak = H.get_Zak_phase(axis=1)
print(f'SSH triv : zak={zak:7.3f}')
