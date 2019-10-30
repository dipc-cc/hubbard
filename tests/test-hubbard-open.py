from __future__ import print_function
import sisl
import numpy as np
import sys
import matplotlib.pyplot as plt
import Hubbard.geometry as geometry
import Hubbard.hamiltonian as hh
import Hubbard.sp2 as sp2

# Set U for the whole calculation
U = 0.0

# Build zigzag GNR
ZGNR = geometry.zgnr(2)

# and 1NN TB Hamiltonian
H_elec = sp2(ZGNR, t1=2.7, t2=0, t3=0)

# Hubbard Hamiltonian of elecs
MFH_elec = hh.HubbardHamiltonian(H_elec, U=U, nkpt=[101, 1, 1])

# Converge Electrode Hamiltonians
dn = MFH_elec.converge(method=2)

# Central region is a repetition of the electrodes without PBC
HC = MFH_elec.tile(3,axis=0)
HC.H.sc.set_nsc([1,1,1])

# MFH object
MFH_HC = hh.HubbardHamiltonian(HC.H, U=U)

# Map electrodes in the device region
elec_indx = [range(len(H_elec)), range(len(HC.H)-len(H_elec), len(HC.H))]

# Pass the electrode Hamiltonians as input and do one iteration
dn = MFH_HC.iterate3(MFH_elec, elec_indx)

# Check if the total number of electrons is correct
print('Nup: ', MFH_HC.nup.sum())
print('Ndn: ', MFH_HC.ndn.sum())
