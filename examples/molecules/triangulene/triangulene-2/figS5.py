import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.sp2 as sp2
import sys
import numpy as np
import sisl

p = plot.Plot()
pos = [21,18,28,32,34]
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

        # AFM case first
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
            #p.annotate()
            p_pol.axes.set_title('pos-%i'%(i+1), fontsize=50, y=-0.2)
            p_pol.axes.axis('off')
            p_pol.savefig(fn+'/pol-%i.pdf'%(u*100))
            p_pol.close()

        f.write('%.4f %.8f\n'%(H.U, e))

    f.close()
    if i == 0:
        E0 = np.loadtxt('H-passivation/pos-1/tot_energy.dat')[:,1]
    else:
        p.axes.plot(U, E-E0, 'o', label='pos$_{%i}$-pos$_{1}$'%(i+1))
        p.set_xlabel(r'U [eV]')
        p.set_ylabel(r'E$_{i}$-E$_{1}$ [eV]')
p.axes.legend()
p.savefig('figS5.pdf')