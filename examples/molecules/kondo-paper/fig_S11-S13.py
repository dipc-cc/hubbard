from hubbard import HubbardHamiltonian, plot, sp2, density
import sys
import numpy as np

import sisl

# Build sisl Geometry object
mol = sisl.get_sile('junction-2-2.XV').read_geometry()
mol.sc.set_nsc([1, 1, 1])
mol = mol.move(-mol.center(what='xyz')).rotate(220, [0, 0, 1])
Hsp2 = sp2(mol, t1=2.7, t2=0.2, t3=.18)

# 3NN tight-binding model
H = HubbardHamiltonian(Hsp2)

# Create mixer
mixer = sisl.mixing.PulayMixer(0.7, history=12)

for u in [0.0, 3.5]:
    H.U = u
    if np.allclose(H.U, 0):
        lab = 'Fig_S12'
    else:
        lab = 'Fig_S13'
        success = H.read_density('fig_S11-S13.nc') # Try reading, if we already have density on file
        if not success:
            H.set_polarization([77], dn=[23])
        mixer.clear()
        H.converge(density.calc_n_insulator, mixer=mixer)
        H.write_density('fig_S11-S13.nc')

    # Plot Eigenspectrum
    p = plot.Spectrum(H, ymax=0.12)
    p.set_title(r'3NN, $U=%.2f$ eV'%np.average(H.U))
    p.savefig('Fig_S11_eigenspectrum_U%i.pdf'%(np.average(H.U)*100))

    # Plot HOMO and LUMO level wavefunctions for up- and down-electrons
    spin = ['up', 'dn']
    N = H.q
    midgap = H.find_midgap()
    for i in range(2):
        ev, evec = H.eigh(eigvals_only=False, spin=i)
        ev -= midgap

        f = 1

        v = evec[:, int(round(N[i]))-1]
        j = np.argmax(abs(v))
        wf = f*v**2*np.sign(v[j])*np.sign(v)
        rs = False
        if not rs:
            wf *= -5000
        p = plot.Wavefunction(H, wf, ext_geom=mol, realspace=rs, vmax=0.0006, vmin=-0.0006)
        p.set_title(r'$E = %.3f$ eV'%(ev[int(round(N[i]))-1]))
        p.axes.axis('off')
        p.savefig(lab+'_HOMO-%s.pdf'%spin[i])

        v = evec[:, int(round(N[i]))]
        j = np.argmax(abs(v))
        wf = f*v**2*np.sign(v[j])*np.sign(v)
        if not rs:
            wf *= -5000
        p = plot.Wavefunction(H, wf, ext_geom=mol, realspace=rs, vmax=0.0006, vmin=-0.0006)
        p.set_title(r'$E = %.3f$ eV'%(ev[int(round(N[i]))]))
        p.axes.axis('off')
        p.savefig(lab+'_LUMO-%s.pdf'%spin[i])
