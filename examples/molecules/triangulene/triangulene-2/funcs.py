import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.sp2 as sp2
import sys
import numpy as np
import sisl

def read(fn, file='/molecule.xyz'):

    # Build sisl Geometry object
    mol = sisl.get_sile(fn+file).read_geometry()
    mol = mol.move(-mol.center(what='xyz'))
    if file[-3:] == '.XV':
        mol = mol.rotate(180, v=[0,1,0]) # Rotate molecule to coincide with the experimental image
    mol.sc.set_nsc([1,1,1])

    # 3NN tight-binding model
    Hsp2 = sp2(mol, t1=2.7, t2=0.2, t3=.18, dim=2)
    H = hh.HubbardHamiltonian(Hsp2)
    return mol, H

def plot_spin(H, mol, label):
    p = plot.SpinPolarization(H, ext_geom=mol, colorbar=True, vmax=0.4, vmin=-0.4)
    p.set_colorbar_yticklabels(['%.1f'%i for i in p.get_colorbar_yticks()], fontsize=25)
    p.axes.axis('off')
    p.savefig(label)
    p.close()

def plot_spectrum(fn, H, mol, ncfile):

    H.read_density(fn+ncfile)
        
    dn = H.converge()

    H.find_midgap()
    ev_up, evec_up = H.eigh(spin=0, eigvals_only=False)
    ev_dn, evec_dn = H.eigh(spin=1, eigvals_only=False)

    if (H.Nup+H.Ndn) %2 != 0:
        H.midgap = max(ev_up[H.Nup-1], ev_dn[H.Ndn-1])
    ev_up -= H.midgap
    ev_dn -= H.midgap

    p = plot.Wavefunction(H, evec_up[:, H.Nup-1], ext_geom=mol, realspace=True, vmax=0.002, vmin=-0.002)
    p.axes.set_title(r'$E_{\uparrow}=%.2f$ eV'%(ev_up[H.Nup-1]), fontsize=30, y=-0.1)
    p.axes.axis('off')
    p.savefig(fn+'/U%i_state%i_up.pdf'%(H.U*100, H.Nup-1))

    p = plot.Wavefunction(H, evec_up[:, H.Nup], ext_geom=mol, realspace=True, vmax=0.002, vmin=-0.002)
    p.axes.set_title(r'$E_{\uparrow}=%.2f$ eV'%(ev_up[H.Nup]), fontsize=30, y=-0.1)
    p.axes.axis('off')
    p.savefig(fn+'/U%i_state%i_up.pdf'%(H.U*100, H.Nup))

    p = plot.Wavefunction(H, evec_dn[:, H.Ndn-1], ext_geom=mol, realspace=True, vmax=0.002, vmin=-0.002)
    p.axes.set_title(r'$E_{\downarrow}=%.2f$ eV'%(ev_dn[H.Ndn-1]), fontsize=30, y=-0.1)
    p.axes.axis('off')
    p.savefig(fn+'/U%i_state%i_dn.pdf'%(H.U*100,H.Ndn-1))
    
    p = plot.Wavefunction(H, evec_dn[:, H.Ndn], ext_geom=mol, realspace=True, vmax=0.002, vmin=-0.002)
    p.axes.set_title(r'$E_{\downarrow}=%.2f$ eV'%(ev_dn[H.Ndn]), fontsize=30, y=-0.1)
    p.axes.axis('off')
    p.savefig(fn+'/U%i_state%i_dn.pdf'%(H.U*100,H.Ndn))

    p = plot.Spectrum(H, ymin=0.02, ymax=0.09, fontsize=25)
    p.axes.set_yticklabels(['%.2f'%i for i in p.axes.get_yticks()], fontsize=20)
    p.axes.set_xticklabels(p.axes.get_xticks(), fontsize=20)
    p.savefig(fn+'/U%i_spectrum.pdf'%(H.U*100))
    p.close()

    # Plot DOS at E=HOMO
    HOMO = max(ev_up[H.Nup-1], ev_dn[H.Ndn-1])
    DOS = H.DOS(HOMO, eta=1e-3, spin=[0,1])
    p = plot.DOS_distribution(H, DOS, ext_geom=mol, realspace=True, z=1.1, colorbar=True)
    p.set_colorbar_ylabel(label=r'$\rho(\textbf{r})$ [$e$\AA$^{-2}$]', fontsize=30)
    p.set_colorbar_yticklabels([format(i, ".0e") for i in p.get_colorbar_yticks()], fontsize=25)
    p.axes.axis('off')
    p.set_title('E $= %.2f$ eV'%(HOMO), fontsize=30)
    p.savefig(fn+'/U%i_DOS.pdf'%(H.U*100))

