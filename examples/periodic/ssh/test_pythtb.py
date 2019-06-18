from __future__ import print_function
import numpy as np
import sys
import matplotlib.pyplot as plt
from pythtb import *

xyz = [[0.0],[1.0],[2.5],[3.5]]
sc = [[5.0]]

my_model = tb_model(1,1,lat=sc, orb=xyz)

# Range t2>t1 -> trivial, t1>t2 -> topological
t1 = -0.5
t2 = -1.0
my_model.set_hop(t1, 0, 1, [0])
my_model.set_hop(t2, 1, 2, [0])
my_model.set_hop(t1, 2, 3, [0])
my_model.set_hop(t2, 3, 0, [1])
#my_model.display()

# plot band structure
fig_band,   ax_band   = plt.subplots()

num_kpt = 100
wf_kpt=wf_array(my_model,[num_kpt])


# create k mesh over 1D Brillouin zone
(k_vec,k_dist,k_node)=my_model.k_path([[-0.5],[0.5]],num_kpt,report=False)

# solve model on all of these k-points
(evals,evec)=my_model.solve_all(k_vec,eig_vectors=True)

# plot band structure for all three bands
for band in range(evals.shape[0]):
    ax_band.plot(k_dist,evals[band,:],"k-",linewidth=0.5)

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

# compute Berry phase along k-direction for each lambda
phase=wf_kpt.berry_phase([0,1],0)
print('Berry phase:', phase)
