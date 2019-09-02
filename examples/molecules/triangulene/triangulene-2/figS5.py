import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.sp2 as sp2
import sys
import numpy as np
import sisl

# First do H-passivation/2H-pos-1-2 molecule

# Build sisl Geometry object
fn = 'H-passivation/2H-pos-1-2/'
mol = sisl.get_sile(fn+'/molecule.xyz').read_geometry()
mol = mol.move(-mol.center(what='xyz'))
mol.sc.set_nsc([1,1,1])

# 3NN tight-binding model
Hsp2 = sp2(mol, t1=2.7, t2=0.2, t3=.18, dim=2)
H = hh.HubbardHamiltonian(Hsp2)
H.polarize_sublattices()
H.U = 3.0
ncf = fn+'/triangulene.nc'
H.read_density(ncf)
dn = H.converge(tol=1e-10, fn=ncf)
H.write_density(ncf)

# Plot spin polarization
p = plot.SpinPolarization(H, ext_geom=mol, colorbar=True, vmax=0.4)
p.axes.axis('off')
p.savefig(fn+'/pol-%i.pdf'%(H.U*100))

# Plot DOS at HOMO 
ev_up, evec_up = H.eigh(spin=0, eigvals_only=False)
ev_dn, evec_dn = H.eigh(spin=1, eigvals_only=False)
H.find_midgap()
ev_up -= H.midgap
ev_dn -= H.midgap
HOMO = max(ev_up[H.Nup-1], ev_dn[H.Ndn-1])
p = plot.DOS_distribution(H, ext_geom=mol, E=HOMO, realspace=True)
p.set_title('E $= %.2f$ eV'%(HOMO), fontsize=30)
p.axes.axis('off')
p.savefig(fn+'/U%i_DOS.pdf'%(H.U*100))

# Then do the rest of H-passivated molecules
p = plot.Plot()
for i, fn in enumerate(['pos-1/', 'pos-2/', 'pos-3/', 'pos-4/', 'pos-5/']):
    # Build sisl Geometry object
    fn = 'H-passivation/'+fn
    mol = sisl.get_sile(fn+'/molecule.xyz').read_geometry()
    mol = mol.move(-mol.center(what='xyz'))
    mol.sc.set_nsc([1,1,1])

    # 3NN tight-binding model
    Hsp2 = sp2(mol, t1=2.7, t2=0.2, t3=.18, dim=2)
    H = hh.HubbardHamiltonian(Hsp2)
    H.polarize_sublattices()
    f = open(fn+'/tot_energy.dat', 'w')

    nup, ndn = H.nup*1, H.ndn*1

    U, E = [], []
    for u in np.linspace(5.0, 0.0, 21):

        H.U = u
        U.append(u)

        ncf = fn+'/triangulene.nc'
        H.nup, H.ndn = nup, ndn
        H.read_density(ncf)
        dn = H.converge(tol=1e-10, fn=ncf)
        e = H.Etot
        E.append(e)
        H.write_density(ncf)
        nup, ndn = H.nup*1, H.ndn*1
        
        if u == 3.0:
            p_pol = plot.SpinPolarization(H, ext_geom=mol, colorbar=True, vmax=0.4)
            p_pol.axes.axis('off')
            p_pol.savefig(fn+'/pol-%i.pdf'%(u*100))
            p_pol.close()

            if i==2:
                # Plot DOS at HOMO for pos-3 H passivated molecule
                ev_up, evec_up = H.eigh(spin=0, eigvals_only=False)
                ev_dn, evec_dn = H.eigh(spin=1, eigvals_only=False)
                H.find_midgap()
                if (H.Nup+H.Ndn) %2 != 0:
                    H.midgap = max(ev_up[H.Nup-1], ev_dn[H.Ndn-1])
                ev_up -= H.midgap
                ev_dn -= H.midgap
                HOMO = max(ev_up[H.Nup-1], ev_dn[H.Ndn-1])
                p_dos = plot.DOS_distribution(H, ext_geom=mol, E=HOMO, realspace=True)
                p_dos.set_title('E $= %.2f$ eV'%(HOMO), fontsize=30)
                p_dos.axes.axis('off')
                p_dos.savefig(fn+'/U%i_DOS.pdf'%(H.U*100))
                p_dos.close()

        f.write('%.4f %.8f\n'%(H.U, e))

    f.close()
    if i == 0:
        E0 = np.loadtxt('H-passivation/pos-1/tot_energy.dat')[:,1]
    else:
        p.axes.plot(U, E-E0, 'o', label='pos$_{%i}$-pos$_{1}$'%(i+1))
        p.set_xlabel(r'U [eV]', fontsize=25)
        p.axes.set_xticks([0,1,2,3,4])
        p.axes.set_xticklabels(p.axes.get_xticks(), fontsize=20)
        p.axes.set_yticklabels(['%.1f'%i for i in p.axes.get_yticks()], fontsize=20)
        p.set_ylabel(r'E$_{i}$-E$_{1}$ [eV]', fontsize=25)
        p.set_xlim(-0.1,4.1)
p.axes.legend()
p.savefig('figS5.pdf')