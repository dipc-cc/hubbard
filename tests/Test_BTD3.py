import sisl
import numpy as np
import sys
import matplotlib.pyplot as plt
from hubbard import HubbardHamiltonian, density, NEGF, sp2, plot
from scipy import sparse as sp
import time
from ase.visualize import view
# Set U for the whole calculation
U = 3.0
kT = 0.025

class dummy_class:
    def __init__(self, pivot, btd):
        self._piv  = pivot
        self._btd  = btd
        lpivot = list(pivot)
        self._ipiv = np.array([lpivot.index(i) for i in range(len(pivot))])
    def pivot(self):  return self._piv
    def ipivot(self): return self._ipiv
    def btd(self):    return self._btd

# Build zigzag GNR

tx, ty = 400,12
useBTD = True

ZGNR = sisl.geom.zgnr(ty)

# and 3NN TB Hamiltonian
H_elec = sp2(ZGNR, t1=2.7, t2=0.2, t3=0.18)#.tile(ty, 1)

# Hubbard Hamiltonian of elecs
MFH_elec = HubbardHamiltonian(H_elec, U=U, nkpt=[102, 1, 1], kT=kT)
# Initialize spin densities
MFH_elec.set_polarization([0], dn=[-1]) # Ensure we break symmetry
# Converge Electrode Hamiltonians
#dn = MFH_elec.converge(density.calc_n, mixer=sisl.mixing.PulayMixer(weight=.7, history=7), tol=1e-10)

dist = sisl.get_distribution('fermi_dirac', smearing=kT)
Ef_elec = MFH_elec.H.fermi_level(MFH_elec.mp, q=MFH_elec.q, distribution=dist)
print("Electrode Ef = ", Ef_elec)
# Shift each electrode with its Fermi-level and write it to netcdf file
MFH_elec.H.shift(-Ef_elec)
MFH_elec.H.write('MFH_elec.nc')

# Central region is a repetition of the electrodes without PBC
HC = H_elec.tile(tx, axis=0)
HC.set_nsc([1, 1, 1])
# The pristine Block partitioning is easy in this case

#We place a hole in the center of the device
nHoles = 46

step = 10
Cl = np.array([HC.center() + np.array([step*i,0,0]) for i in range(-nHoles//2+1, nHoles//2+1)
               ])

Cl = Cl.reshape(nHoles,3)

Rl = np.array([3.0,
               ]*nHoles)

def Holes(r):
    Di = np.linalg.norm(r - Cl, axis = 1)
    if (Di<Rl).any():
        return True
    return False

# indices to be removed
ridx = [i for i in range(HC.na) if Holes(HC.xyz[i])]
# indices to stay
sidx = [i for i in range(HC.na) if i not in ridx]

HC = HC.sub(sidx)


print('Number of atoms: ', HC.na)

view(HC.geom.toASE())

# In[]

def get_btd_partition_of_matrix(_csr, start):
    from scipy.sparse.csgraph import reverse_cuthill_mckee as rcm
    from scipy import sparse as sp
    assert _csr.shape[0] == _csr.shape[1]
    p = rcm(_csr)
    no = _csr.shape[0]
    csr = _csr[p,:][:,p]
    Part = [0, start]
    while True:
        sl = slice(Part[-2], Part[-1])
        coup = csr[sl,sl.stop:]
        i,j,v = sp.find(coup)
        if len(v)==0:
            break
        else:
            Part += [Part[-1] + np.max(j)+1]
    btd = [Part[i+1] - Part[i] for i in range(len(Part)-1)]
    return p, btd, Part


# Map electrodes in the device region
elec_indx = [range(len(H_elec)), range(-len(H_elec), 0)]

E_Ms = []
for eidx in elec_indx:
    M = sp.csr_matrix(HC.shape[0:2])
    for _i in eidx:
        for _j in eidx:
            M[_i,_j] = 1.0
    E_Ms += [M]

NonZeros = HC.Hk() + sum(E_Ms)
NonZeros = abs(NonZeros)
#NonZeros = NonZeros + NonZeros.T



from Block_matrices.Block_matrices import test_partition_2d_sparse_matrix as testP
piv,btd,part = get_btd_partition_of_matrix(NonZeros, 40)
f,S = testP(NonZeros[:,piv][piv,:], part)
print(f)


tbt = dummy_class(piv, btd)
Ov  = sp.identity(HC.no)

# MFH object
MFH_HC = HubbardHamiltonian(HC.H, #n=np.tile(MFH_elec.n, tx), 
                            U=U, kT=kT)

# First create NEGF object
if useBTD == False:
    tbt = None
    Ov  = None

negf = NEGF(MFH_HC, [(MFH_elec, '-A'), (MFH_elec, '+A')], elec_indx, 
            tbt = tbt, Ov = Ov
            )


time_start = time.time()

# Converge using Green's function method to obtain the densities
#dn = MFH_HC.converge(negf.calc_n_open, 
#                     steps=1, mixer=sisl.mixing.PulayMixer(weight=.1), tol=0.1)
dn = MFH_HC.converge(negf.calc_n_open, tol = 1e-6,
                     steps=1, mixer=sisl.mixing.PulayMixer(weight=1., history=7), 
                     print_info=True)

# In[]

plt.scatter(MFH_HC.geometry.xyz[:,0], MFH_HC.geometry.xyz[:,1], s = (MFH_HC.n[0]-0.48)*200)
#cooment in when script has been run without btd algo
# if useBTD:
#     nprev = np.load('PrevRun_n.npy')
#     assert np.allclose(nprev, MFH_HC.n)
# if useBTD == False:
#     np.save('PrevRun_n.npy',MFH_HC.n)

print('Nup, Ndn: ', MFH_HC.n.sum(axis=1))
print("time:", time.time()-time_start)

# Shift device with its Fermi level and write nc file
MFH_HC.H.write('MFH_HC.nc', Ef=negf.Ef)

# TBtrans RUN and plot transmission
import os
print('Clean TBtrans output from previous run')
os.system('rm device.TBT*')
os.system('rm fdf*')
print('Running TBtrans')
os.system('tbtrans RUN.fdf > RUN.out')

tbt_up = sisl.get_sile('device.TBT_UP.nc')
tbt_dn = sisl.get_sile('device.TBT_DN.nc')

p = plot.Plot()
p.axes.plot(tbt_up.E, tbt_up.transmission(0, 1), label=r'$\sigma=\uparrow$')
p.axes.plot(tbt_dn.E, tbt_dn.transmission(0, 1), label=r'$\sigma=\downarrow$')
p.axes.legend()
p.set_xlim(-10, 10)
p.set_xlabel('Energy [eV]')
p.set_ylabel('Transmission [a.u.]')
p.savefig('transmission.pdf')


