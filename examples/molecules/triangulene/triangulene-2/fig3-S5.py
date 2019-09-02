import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.sp2 as sp2
import sys
import numpy as np
import sisl
import funcs

#### First do H-passivation/2H-pos-1-2 molecule
fn = 'H-passivation/2H-pos-1-2/'
mol, H = funcs.read(fn)

H.polarize_sublattices()
H.U = 3.0
ncf = fn+'/triangulene.nc'
H.read_density(ncf)
dn = H.converge(tol=1e-10, fn=ncf)
H.write_density(ncf)

# Plot spin polarization
funcs.plot_spin(H, mol, fn+'/pol-%i.pdf'%(H.U*100))
# Plot DOS at HOMO 
funcs.plot_spectrum(fn, H, mol, fn+'/triangulene.nc')

#### Then do the rest of H-passivated molecules
p = plot.Plot()
for i, fn in enumerate(['pos-1/', 'pos-2/', 'pos-3/', 'pos-4/', 'pos-5/']):
    # Build sisl Geometry object
    fn = 'H-passivation/'+fn
    mol, H = funcs.read(fn)

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
            funcs.plot_spin(H, mol, fn+'/pol-%i.pdf'%(u*100))

            if i==2:
                funcs.plot_spectrum(fn, H, mol, fn+'/triangulene.nc')

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