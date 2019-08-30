import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.sp2 as sp2
import sys
import numpy as np
import sisl

# Build sisl Geometry object
fn = 'dimer-pentagon/'
mol = sisl.get_sile(fn+'molecule.XV').read_geometry()
mol = mol.move(-mol.center(what='xyz'))
mol = mol.rotate(180, v=[0,1,0]) # Rotate molecule to coincide with the experimental image
mol.sc.set_nsc([1,1,1])

# 3NN tight-binding model
Hsp2 = sp2(mol, t1=2.7, t2=0.2, t3=.18, dim=2)
H = hh.HubbardHamiltonian(Hsp2)
H.polarize_sublattices()
f = open(fn+'/AFM-FM1-FM2.dat', 'w')

nup_AFM, ndn_AFM = H.nup*1, H.ndn*1
nup_FM1, ndn_FM1 = H.nup*1, H.ndn*1
nup_FM2, ndn_FM2 = H.nup*1, H.ndn*1

for u in np.linspace(5.0, 0.0, 21):

    H.U = u

    # AFM case first
    ncf = fn+'/triangulene-AFM.nc'
    H.nup, H.ndn = nup_AFM, ndn_AFM
    H.read_density(ncf)
    dn = H.converge(tol=1e-10, fn=ncf)
    eAFM = H.Etot
    H.write_density(ncf)
    nup_AFM, ndn_AFM = H.nup*1, H.ndn*1

    if u == 3.0:
        p = plot.SpinPolarization(H, ext_geom=mol, colorbar=True, vmax=0.4)
        #p.annotate()
        p.savefig(fn+'/AFM-pol-%i.pdf'%(u*100))
        p.close()


    # Now FM1 case
    ncf = fn+'/triangulene-FM1.nc'
    H.Nup += 1 # change to two more up-electrons than down
    H.Ndn -= 1

    H.nup, H.ndn = nup_FM1, ndn_FM1
    H.read_density(ncf)
    dn = H.converge(tol=1e-10, fn=ncf)
    eFM1 = H.Etot
    H.write_density(ncf)
    nup_FM1, ndn_FM1 = H.nup*1, H.ndn*1

    if u == 3.0:
        p = plot.SpinPolarization(H, ext_geom=mol, colorbar=True, vmax=0.4)
        #p.annotate()
        p.savefig(fn+'/FM1-pol-%i.pdf'%(u*100))
        p.close()


    # Now FM2 case
    ncf = fn+'/triangulene-FM2.nc'
    H.Nup += 1 # change to four more up-electrons than down
    H.Ndn -= 1

    H.nup, H.ndn = nup_FM2, ndn_FM2
    H.read_density(ncf)
    dn = H.converge(tol=1e-10, fn=ncf)
    eFM2 = H.Etot
    H.write_density(ncf)
    nup_FM2, ndn_FM2 = H.nup*1, H.ndn*1

    if u == 3.0:
        p = plot.SpinPolarization(H, ext_geom=mol, colorbar=True, vmax=0.4)
        #p.annotate()
        p.savefig(fn+'/FM2-pol-%i.pdf'%(u*100))
        p.close()

    # Revert imbalance
    H.Nup -= 2
    H.Ndn += 2

    f.write('%.4f %.8f %.8f %.8f\n'%(H.U, eAFM, eFM1, eFM2))

f.close()

data = np.loadtxt(fn+'AFM-FM1-FM2.dat')
p = plot.Plot()
p.axes.plot(data[:,0], data[:,2]-data[:,1], 'o', label='S$_{1}$-S$_{0}$')
p.axes.plot(data[:,0], data[:,3]-data[:,1], 'o', label='S$_{2}$-S$_{0}$')
p.set_xlabel(r'U [eV]')
p.set_ylabel(r'E$_{S_{i}}$-E$_{S_{0}}$ [eV]')
p.axes.legend()
p.savefig(fn+'figS9.pdf')

H.polarize_sublattices()

# Plot WF, spectrum and DOS of the GS (in this case FM1)
for u in [0., 3.0]:
    if u > 0:
        H.Nup += 1
        H.Ndn -= 1
        
    H.U = u

    H.read_density(fn+'/triangulene-FM1.nc')
        
    dn = H.converge()

    H.find_midgap()
    ev_up, evec_up = H.eigh(spin=0, eigvals_only=False)
    ev_up -= H.midgap
    ev_dn, evec_dn = H.eigh(spin=1, eigvals_only=False)
    ev_dn -= H.midgap

    p = plot.Wavefunction(H, evec_up[:, H.Nup-2], ext_geom=mol, realspace=True, vmax=0.0006)
    p.set_title(r'$E_{\uparrow}=%.2f$ meV, $U=%.1f$ eV'%(ev_up[H.Nup-2]*1000, H.U))
    p.savefig(fn+'/U%i_state%i_up.pdf'%(H.U*100, H.Nup-2))

    p = plot.Wavefunction(H, evec_up[:, H.Nup-1], ext_geom=mol, realspace=True, vmax=0.0006)
    p.set_title(r'$E_{\uparrow}=%.2f$ meV, $U=%.1f$ eV'%(ev_up[H.Nup-1]*1000, H.U))
    p.savefig(fn+'/U%i_state%i_up.pdf'%(H.U*100, H.Nup-1))

    p = plot.Wavefunction(H, evec_up[:, H.Nup], ext_geom=mol, realspace=True, vmax=0.0006)
    p.set_title(r'$E_{\uparrow}=%.2f$ meV, $U=%.1f$ eV'%(ev_up[H.Nup]*1000, H.U))
    p.savefig(fn+'/U%i_state%i_up.pdf'%(H.U*100, H.Nup))

    p = plot.Wavefunction(H, evec_dn[:, H.Ndn-1], ext_geom=mol, realspace=True, vmax=0.0006)
    p.set_title(r'$E_{\downarrow}=%.2f$ meV, $U=%.1f$ eV'%(ev_dn[H.Ndn-1]*1000, H.U))
    p.savefig(fn+'/U%i_state%i_dn.pdf'%(H.U*100,H.Ndn-1))
    
    p = plot.Wavefunction(H, evec_dn[:, H.Ndn], ext_geom=mol, realspace=True, vmax=0.0006)
    p.set_title(r'$E_{\downarrow}=%.2f$ meV, $U=%.1f$ eV'%(ev_dn[H.Ndn]*1000, H.U))
    p.savefig(fn+'/U%i_state%i_dn.pdf'%(H.U*100,H.Ndn))

    p = plot.Spectrum(H)
    p.savefig(fn+'/U%i_spectrum.pdf'%(H.U*100))
    p.close()

    E = ev_up[H.Nup-1] 
    p = plot.DOS_distribution(H, ext_geom=mol, E=E, realspace=True)
    p.set_title('E $= %.2f$ eV'%(E))
    p.savefig(fn+'/U%i_DOS.pdf'%(H.U*100))
