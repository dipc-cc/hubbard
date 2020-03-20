import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.density as density
import Hubbard.sp2 as sp2
import sys
import numpy as np
import sisl

# Build sisl Geometry object
mol = sisl.get_sile('clar-goblet.xyz').read_geometry()
mol.sc.set_nsc([1, 1, 1])
mol = mol.move(-mol.center(what='xyz'))

# 3NN tight-binding model
Hsp2 = sp2(mol, t1=2.7, t2=0.2, t3=.18, dim=2)
H = hh.HubbardHamiltonian(Hsp2)
H.random_density()
H.set_polarization(up=[6], dn=[28])
dm_AFM = H.dm

f = open('FM-AFM.dat', 'w')

mixer = sisl.mixing.PulayMixer(0.7, history=7)

for u in np.arange(5, 0, -0.25):
    # We approach the solutions for different U values
    H.U = u
    mixer.clear()

    # AFM case first
    success = H.read_density('clar-goblet.nc') # Try reading, if we already have density on file
    if not success:
        H.dm = dm_AFM.copy()

    dn = H.converge(density.dm_insulator, tol=1e-10, mixer=mixer)
    eAFM = H.Etot
    H.write_density('clar-goblet.nc')
    dm_AFM = H.dm.copy()

    if u == 3.5:
        p = plot.SpinPolarization(H, colorbar=True, vmax=0.4, vmin=-0.4)
        p.savefig('spin_pol_U%i_AFM.pdf'%(H.U*1000))

    # Now FM case
    H.q[0] += 1 # change to two more up-electrons than down
    H.q[1] -= 1
    success = H.read_density('clar-goblet.nc') # Try reading, if we already have density on file
    if not success:
        H.random_density()
        H.set_polarization(up=[6, 28])
    mixer.clear()
    dn = H.converge(density.dm_insulator, tol=1e-10, mixer=mixer)
    eFM = H.Etot
    H.write_density('clar-goblet.nc')

    if u == 3.5:
        p = plot.SpinPolarization(H, colorbar=True, vmax=0.4, vmin=-0.4)
        p.savefig('spin_pol_U%i_FM.pdf'%(H.U*1000))

    # Revert the imbalance for next loop
    H.q[0] -= 1
    H.q[1] += 1

    f.write('%.4f %.8f\n'%(H.U, eFM-eAFM))

f.close()

U, FM_AFM = np.loadtxt('FM-AFM.dat').T
p = plot.Plot()
p.axes.plot(U, FM_AFM, 'o')
p.set_ylabel(r'$E_{FM}-E_{AFM}$ [eV]')
p.set_xlabel(r'$U$ [eV]')
p.savefig('FM_AFM.pdf')
