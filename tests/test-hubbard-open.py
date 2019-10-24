from __future__ import print_function
import sisl
import numpy as np
import sys
import matplotlib.pyplot as plt
import Hubbard.geometry as geometry
import Hubbard.hamiltonian as hh
import Hubbard.sp2 as sp2

# Build one dimensional chain geometry
one_dim_chain = sisl.Geometry([[0,0,0], [1.0,0,0]], atom=sisl.Atom(6), sc=sisl.SuperCell([1.0, 10, 10], nsc=[3,1,1]))
# and Hamiltonian
H_elec = sp2(one_dim_chain, t1=1.0, t2=0, t3=0)

# Hubbard Hamiltonian of elecs
MFH_elec = hh.HubbardHamiltonian(H_elec, U=0)

# Converge Electrode Hamiltonians
dn = MFH_elec.converge()

# Central region is a repetition of the electrodes
HC = H_elec.tile(3,axis=0)
HC.set_nsc([1,1,1])
MFH_HC = hh.HubbardHamiltonian(HC, U=0)

# Map electrodes in the device region
elec_indx = [range(len(H_elec)),range(len(HC)-len(H_elec), len(HC))]

# Pass the electrode Hamiltonians as input
dn = MFH_HC.iterate3(MFH_elec.H, elec_indx)
print('Nup: ', MFH_HC.nup.sum())
print('Ndn: ', MFH_HC.ndn.sum())