from __future__ import print_function
import sisl
import numpy as np
import sys
import matplotlib.pyplot as plt
import Hubbard.geometry as geometry
import Hubbard.hamiltonian as hh
import Hubbard.sp2 as sp2

# Set U for the whole calculation
U = 3.0

# Build zigzag GNR
ZGNR = geometry.zgnr(2)

# and 3NN TB Hamiltonian
H_elec = sp2(ZGNR, t1=2.7, t2=0.2, t3=0.18)

# Hubbard Hamiltonian of elecs
MFH_elec = hh.HubbardHamiltonian(H_elec, U=U, nkpt=[101, 1, 1])

# Converge Electrode Hamiltonians
dn = MFH_elec.converge(method=2)

# Central region is a repetition of the electrodes without PBC
HC = H_elec.tile(3,axis=0)
HC.set_nsc([1,1,1])

# Map electrodes in the device region
elec_indx = [range(len(H_elec)), range(len(HC.H)-len(H_elec), len(HC.H))]

# MFH object
MFH_HC = hh.HubbardHamiltonian(HC.H, DM=MFH_elec.DM.tile(3,axis=0), U=U, elecs=MFH_elec, elec_indx=elec_indx)

# Create fermi-level distribution
kT = 0.025
dist = sisl.get_distribution('fermi_dirac', smearing=kT)
# Shift central region Hamiltonian with EF
Ef = MFH_HC.H.fermi_level(q=[MFH_HC.Nup, MFH_HC.Ndn], distribution=dist)
MFH_HC.H.shift(-Ef)
print(Ef)


# Converge using iterative method 3
dn = MFH_HC.converge(method=3, steps=10)
print(MFH_HC.nup.sum(), MFH_HC.ndn.sum())
