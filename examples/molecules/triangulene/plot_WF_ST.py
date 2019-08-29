import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.sp2 as sp2
import sys
import numpy as np
import sisl

# Folder name (triangulene-1, -2 or -3)
fn = sys.argv[1]

# Build sisl Geometry object
mol = sisl.get_sile(fn+'/molecule.xyz').read_geometry()
mol = mol.move(-mol.center(what='xyz'))
mol.sc.set_nsc([1,1,1])

# 3NN tight-binding model
Hsp2 = sp2(mol, t1=2.7, t2=0.2, t3=.18, dim=2)
H = hh.HubbardHamiltonian(Hsp2)

if 'triangulene-1' in fn or 'triangulene-2' in fn:
    H.Nup += 1
    H.Ndn -= 1
    fn_nc = fn+'triangulene-FM.nc'
else:
    fn_nc = fn+'trianglene-AFM.nc'
H.polarize_sublattices()

for u in [0., 3.5]:

    H.U = u
    H.read_density(fn_nc)
    
    # Plot wavefunctions
    dn = H.converge()
    H.find_midgap()
    ev_up, evec_up = H.eigh(eigvals_only=False, spin=0)
    ev_up -= H.midgap
    ev_dn, evec_dn = H.eigh(eigvals_only=False, spin=1)
    ev_dn -= H.midgap

    p = plot.Wavefunction(H, 3000*evec_up[:, H.Nup-2], ext_geom=mol)
    p.set_title(r'$E_{\uparrow}=%.2f$ meV, $U=%.1f$ eV'%(ev_up[H.Nup-2]*1000, H.U))
    p.savefig(fn+'/U%i_state%i_up.pdf'%(H.U*100, H.Nup-2))

    p = plot.Wavefunction(H, 3000*evec_up[:, H.Nup-1], ext_geom=mol)
    p.set_title(r'$E_{\uparrow}=%.2f$ meV, $U=%.1f$ eV'%(ev_up[H.Nup-1]*1000, H.U))
    p.savefig(fn+'/U%i_state%i_up.pdf'%(H.U*100, H.Nup-1))

    p = plot.Wavefunction(H, 3000*evec_up[:, H.Nup], ext_geom=mol)
    p.set_title(r'$E_{\uparrow}=%.2f$ meV, $U=%.1f$ eV'%(ev_up[H.Nup]*1000, H.U))
    p.savefig(fn+'/U%i_state%i_up.pdf'%(H.U*100, H.Nup))

    p = plot.Wavefunction(H, 3000*evec_dn[:, H.Ndn-1], ext_geom=mol)
    p.set_title(r'$E_{\downarrow}=%.2f$ meV, $U=%.1f$ eV'%(ev_dn[H.Ndn-1]*1000, H.U))
    p.savefig(fn+'/U%i_state%i_dn.pdf'%(H.U*100,H.Ndn-1))
    
    p = plot.Wavefunction(H, 3000*evec_dn[:, H.Ndn], ext_geom=mol)
    p.set_title(r'$E_{\downarrow}=%.2f$ meV, $U=%.1f$ eV'%(ev_dn[H.Ndn]*1000, H.U))
    p.savefig(fn+'/U%i_state%i_dn.pdf'%(H.U*100,H.Ndn))

    p = plot.Spectrum(H)
    p.savefig(fn+'/U%i_spectrum.pdf'%(H.U*100))
    p.close()

# Plot FM-AFM energies
dat = np.loadtxt(fn+'/FM-AFM.dat')

p = plot.Plot()
p.axes.plot(dat[:,0], dat[:,1], label='FM-AFM')
p.set_xlabel(r'$U$ [eV]')
p.set_ylabel(r'$E_\mathrm{FM}-E_\mathrm{AFM}$ [eV]')
p.axes.legend()
p.savefig(fn+'/FM-AFM.pdf')
