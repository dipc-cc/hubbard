from __future__ import print_function
import numpy as np
import sys
import matplotlib.pyplot as plt
from pythtb import *
import sisl
import Hubbard.geometry as geom
import Hubbard.hamiltonian as hh

n,m,w = 3,1,4
geom = geom.cgnr(n,m,w).uc
H = sisl.Hamiltonian(geom)

xyz = geom.xyz[:, :2]

sc = [[geom.cell[0,0], 0], [0, 10]]
my_model = tb_model(1,2, lat=sc, orb=xyz)

t = -2.7
list_indx = []
for ia in geom:
    idx = geom.close(ia, R=[0.1, 1.43])
    H[ia, idx[0]] = 0
    H[ia, idx[1]] = t
    for i in idx[1]:
        list_indx.append(i)
        if i < len(geom):
            # Divide hopping by two because it is counting two times (see doc for tb_model.set_hop method)
            my_model.set_hop(t/2, ia, i, [0,0], allow_conjugate_pair=True)
        elif i >= len(geom):
            f = i/len(geom)
            my_model.set_hop(t/2, i-f*len(geom), ia, [1-2*(f/2),0], allow_conjugate_pair=True)

#my_model.display()

# plot band structure
fig_band,   ax_band   = plt.subplots()

num_kpt = 100
wf_kpt=wf_array(my_model,[num_kpt])

# create k mesh over 1D Brillouin zone
(k_vec,k_dist,k_node)=my_model.k_path([[-0.5],[0.5]],num_kpt,report=False)

# solve model on all of these k-points
(evals,evec)=my_model.solve_all(k_vec,eig_vectors=True)


ev_sisl = np.zeros((len(k_vec), len(geom)))
for ik, k in enumerate(k_vec):
    ev_sisl[ik, :] = H.eigh(k=[k,0,0])

# plot band structure for all three bands and compare to sisl bands
for band in range(evals.shape[0]):
    ax_band.plot(k_dist,evals[band,:],"k-",linewidth=0.5)
    ax_band.plot(k_dist,ev_sisl[:, band],"r-",linewidth=0.5)
    
# finish plot for band structure
ax_band.set_title("Band structure")
ax_band.set_xlabel("Path in k-vector")
ax_band.set_ylabel("Band energies")
fig_band.tight_layout()
fig_band.savefig("bands.pdf")

# store wavefunctions (eigenvectors)
for i_kpt in range(num_kpt):
    wf_kpt[i_kpt]=evec[:,i_kpt,:]

# impose periodic boundary condition along k-space direction only
# (so that |psi_nk> at k=0 and k=1 have the same phase)
wf_kpt.impose_pbc(0,0)

# compute Berry phase along k-direction for all occ. bands
phase=wf_kpt.berry_phase(range(len(geom)/2),0)
print('Berry phase:', phase)
