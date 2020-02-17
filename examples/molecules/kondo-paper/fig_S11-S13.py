import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.sp2 as sp2
import Hubbard.density as dm
import sys
import numpy as np

import sisl

# Build sisl Geometry object
mol = sisl.get_sile('junction-2-2.XV').read_geometry()
mol.sc.set_nsc([1,1,1])
mol = mol.move(-mol.center(what='xyz')).rotate(220, [0,0,1])
Hsp2 = sp2(mol, t1=2.7, t2=0.2, t3=.18)

# 3NN tight-binding model
H = hh.HubbardHamiltonian(Hsp2)

for u in [0.0, 3.5]:
    H.U = u
    if H.U == 0:
        lab = 'Fig_S12'
    else:
        lab = 'Fig_S13'
        H.read_density('fig_S11-S13.nc') # Try reading, if we already have density on file
        H.converge(dm.dm_insulator)
        H.write_density('fig_S11-S13.nc')

    # Plot Eigenspectrum
    p = plot.Spectrum(H, ymax=0.12)
    p.set_title(r'3NN, $U=%.2f$ eV'%H.U)
    p.savefig('Fig_S11_eigenspectrum_U%i.pdf'%(H.U*100))

    # Plot HOMO and LUMO level wavefunctions for up- and down-electrons
    spin = ['up', 'dn']
    N = [H.Nup, H.Ndn]
    for i in range(2):
        ev, evec = H.eigh(eigvals_only=False, spin=i)
        H.find_midgap()
        ev -= H.midgap

        f = 1

        v = evec[:, N[i]-1]
        j = np.argmax(abs(v))
        wf = f*v**2*np.sign(v[j])*np.sign(v)
        p = plot.Wavefunction(H, wf, ext_geom=mol, realspace=True, vmax=0.0006, vmin=-0.0006)
        p.set_title(r'$E = %.3f$ eV'%(ev[N[i]-1]))
        p.axes.axis('off')
        p.savefig(lab+'_HOMO-%s.pdf'%spin[i])

        v = evec[:, N[i]]
        j = np.argmax(abs(v))
        wf = f*v**2*np.sign(v[j])*np.sign(v)
        p = plot.Wavefunction(H, wf, ext_geom=mol, realspace=True, vmax=0.0006, vmin=-0.0006)
        p.set_title(r'$E = %.3f$ eV'%(ev[N[i]]))
        p.axes.axis('off')
        p.savefig(lab+'_LUMO-%s.pdf'%spin[i])
