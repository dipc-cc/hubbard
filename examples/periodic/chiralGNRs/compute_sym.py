import sisl
import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.geometry as geom
import numpy as np
import os
 
n,m,w = 3,1,8
geom = geom.cgnr(n,m,w).uc

H = hh.HubbardHamiltonian(geom, t1=2.7, t2=0., t3=0., U=0)

# Gamma
ev, evec_G = H.eigh(k=[0., 0, 0], eigvals_only=False)
evec_G = evec_G[:, :H.Nup]
M = H.band_sym(evec_G, diag=False)
e_G, v = np.linalg.eigh(M)
print('     Sym eigenvalues at Gamma:')
print(e_G)

# Now at X
ev, evec_X = H.eigh(k=[0.5, 0, 0], eigvals_only=False ,spin=0)
evec_X = evec_X[:, :H.Nup]
M = H.band_sym(evec_X, diag=False)
e_X, v = np.linalg.eigh(M)
print('     Sym eigenvalues at X:')
print(e_X)

prd=1
for i in range(len(geom)/2):
    prd *= e_G[i]*e_X[i]
print(prd)