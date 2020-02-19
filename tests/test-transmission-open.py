from __future__ import print_function
import sisl
import numpy as np
import sys
import matplotlib.pyplot as plt
import Hubbard.geometry as geometry
import Hubbard.density as density
import Hubbard.hamiltonian as hh
import Hubbard.sp2 as sp2
import Hubbard.plot as plot

# Set U for the whole calculation
U = 3.0
kT = 0.025

# Build zigzag GNR
ZGNR = geometry.zgnr(2)

# and 3NN TB Hamiltonian
H_elec = sp2(ZGNR, t1=2.7, t2=0.2, t3=0.18)

# Hubbard Hamiltonian of elecs
MFH_elec = hh.HubbardHamiltonian(H_elec, U=U, nkpt=[102, 1, 1], kT=kT)

# Converge Electrode Hamiltonians
dn = MFH_elec.converge(density.dm)

dist = sisl.get_distribution('fermi_dirac', smearing=kT)
Ef_elec = MFH_elec.H.fermi_level(MFH_elec.mp, q=MFH_elec.q, distribution=dist)
# Shift each electrode with its Fermi-level and write it to netcdf file
MFH_elec.H.shift(-Ef_elec)
MFH_elec.H.write('MFH_elec.nc')

# Central region is a repetition of the electrodes without PBC
HC = H_elec.tile(3,axis=0)
HC.set_nsc([1,1,1])

# Map electrodes in the device region
elec_indx = [range(len(H_elec)), range(-len(H_elec), 0)]

# MFH object
MFH_HC = hh.HubbardHamiltonian(HC.H, DM=MFH_elec.DM.tile(3,axis=0), U=U, kT=kT)

# First create NEGF object
negf = density.NEGF(MFH_HC, [MFH_elec, MFH_elec], elec_indx, elec_dir=['-A', '+A'])
# Converge using Green's function method to obtain the densities
dn = MFH_HC.converge(negf.dm_open, steps=1)
print('Nup, Ndn: ', MFH_HC.dm.sum(axis=1))

# Shift device with its Fermi level and write nc file
MFH_HC.H.shift(negf.Ef)
MFH_HC.H.write('MFH_HC.nc')

# TBtrans RUN and plot transmission
import os
print('Clean TBtrans output from previous run')
os.system('rm device.TBT*')
os.system('rm fdf*')
print('Runing TBtrans')
os.system('tbtrans RUN.fdf > RUN.out')

tbt_up = sisl.get_sile('device.TBT_UP.nc')
tbt_dn = sisl.get_sile('device.TBT_DN.nc')

p = plot.Plot()
p.axes.plot(tbt_up.E, tbt_up.transmission(0,1), label=r'$\sigma=\uparrow$')
p.axes.plot(tbt_dn.E, tbt_dn.transmission(0,1), label=r'$\sigma=\downarrow$')
p.axes.legend()
p.set_xlim(-10,10)
p.set_xlabel('Energy [eV]')
p.set_ylabel('Transmission [a.u.]')
p.savefig('transmission.pdf')