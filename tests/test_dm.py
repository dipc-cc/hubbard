import sisl
import numpy as np

# build 1D chain
natoms = 1
c = sisl.Atom('C', R=1.)
g = sisl.geom.sc(1.42, c) * (natoms, 1, 1)
g.set_nsc([3, 1, 1])

# initialize Hamiltonian
H = sisl.Hamiltonian(g, orthogonal=False)
#print(H)

# nearest-neighbor overlap
overlap = 0.1
for ia in g:
    idx = g.close(ia, R=(0.1, 1.44))
    H[ia, idx[0]] = [0, 1]
    H[ia, idx[1]] = [-2.7, overlap]

dist = sisl.get_distribution('fermi_dirac', smearing=0.1)
mp = sisl.MonkhorstPack(H, [1000, 1, 1])

mulliken = 0j
dm = 0j
for w, k in zip(mp.weight, mp.k):
    es = H.eigenstate(k, spin=0)
    occ = es.occupation(dist) * w
    mulliken += np.einsum('i,ij->j', occ, es.norm2(False))
    # build extended state vectors in the whole supercell
    extstate = np.concatenate([np.exp(2j * np.pi * k.dot(isc)) * es.state.T for _, isc in g.sc]).T
    dm += np.einsum('n,ni,nj->ij', occ, es.state, np.conj(extstate))

print('mulliken', mulliken)

# initialize density matrix
DM = sisl.DensityMatrix(g, spin='unpolarized', orthogonal=False, dtype='complex')
#print(DM)

# insert DM elements where relevant
for ia in g:
    idx = g.close(ia, R=1.44)
    for j in idx:
        DM[ia, j] = [dm[ia, j], H[ia, j, -1]]

print('DM', DM.tocsr())
print('DM-derived mulliken:', DM.mulliken())
