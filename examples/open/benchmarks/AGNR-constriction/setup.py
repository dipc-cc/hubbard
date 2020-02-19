from __future__ import print_function
import sisl
import numpy as np
import Hubbard.geometry as geometry
import Hubbard.hamiltonian as hh
import Hubbard.sp2 as sp2
import Hubbard.plot as plot
import Hubbard.density as density
import os

'''
This script benchmarks the 9-AGNR constriction of ref. [Y. Hancock, et al., PRB 81, 245402 (2010)]
i.e. the system of fig. 3(b) and the transmissions shown in fig. 4(a)
To describe the system model D of this paper is used (t1=2.7, t2=0.2, t3=0.18, U=2.0 eV)
'''

# Set U and kT for the whole calculation
U = 2.0
kT = 0.025

# Build zigzag GNR
AGNR = geometry.agnr(9)

# and 3NN TB Hamiltonian
H_elec = sp2(AGNR, t1=2.7, t2=0.2, t3=0.18)

# Hubbard Hamiltonian of elecs
MFH_elec = hh.HubbardHamiltonian(H_elec, U=U, nkpt=[102, 1, 1],  kT=0.025)
# Initial densities
MFH_elec.read_density('elec_density.nc')

# Converge Electrode Hamiltonians
dn = MFH_elec.converge(density.dm)

# Write also densities for future calculations
MFH_elec.write_density('elec_density.nc')
# Plot spin polarization of electrodes
p = plot.SpinPolarization(MFH_elec, colorbar=True)
p.savefig('spin_elecs.pdf')

# Find Fermi level of reservoirs and write to netcdf file
dist = sisl.get_distribution('fermi_dirac', smearing=kT)
Ef_elecs = MFH_elec.H.fermi_level(MFH_elec.mp, q=MFH_elec.q, distribution=dist)
MFH_elec.H.shift(-Ef_elecs)
MFH_elec.H.write('MFH_elec.nc')

# Central region is a repetition of the electrodes without PBC
HC = H_elec.tile(11,axis=0)
HC = HC.remove([89,106,107,86,85,102,105,104,103,124,120,101,100,123])
HC.set_nsc([1,1,1])
HC.geom.write('device.xyz')

# Map electrodes in the device region
elec_indx = [range(len(H_elec)), range(len(HC.H)-len(H_elec), len(HC.H))]

# MFH object
MFH_HC = hh.HubbardHamiltonian(HC.H, U=U, kT=kT)
# Initial densities
success = MFH_HC.read_density('HC_density.nc')
if not success:
    # Converge without OBC to have initial density
    MFH_HC.converge(density, tol=1e-5)

# First create NEGF object
negf = density.NEGF(MFH_HC, [MFH_elec, MFH_elec], elec_indx, elec_dir=['-A', '+A'])
# Converge using Green's function method to obtain the densities
dn = MFH_HC.converge(negf.dm_open, steps=1, tol=1e-5)

MFH_HC.H.shift(-negf.Ef)
MFH_HC.H.write('MFH_HC.nc')
print('Nup, Ndn: ', MFH_HC.dm.sum(axis=1))
# Write also densities for future calculations
MFH_HC.write_density('HC_density.nc')
# Plot spin polarization of electrodes
p = plot.SpinPolarization(MFH_HC, colorbar=True)
p.savefig('spin_HC.pdf')

# RUN TBtrans and plot transmissions
print('Clean TBtrans output from previous run')
os.system('rm device.TBT*')
os.system('rm fdf*')
print('Running TBtrans')
os.system('tbtrans RUN.fdf > RUN.out')

tbt_up = sisl.get_sile('device.TBT_UP.nc')
tbt_dn = sisl.get_sile('device.TBT_DN.nc')

p = plot.Plot()
p.axes.plot(tbt_up.E, tbt_up.transmission(0,1), color='k', label=r'$\sigma=\uparrow$')
p.axes.plot(tbt_dn.E, tbt_dn.transmission(0,1), '--', color='r', label=r'$\sigma=\downarrow$')
p.axes.legend()
p.set_xlim(-2,2)
p.set_xlabel('Energy [eV]')
p.set_ylabel('Transmission [a.u.]')
p.savefig('transmission.pdf')