import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.sp2 as sp2
import sys
import numpy as np
import sisl
import funcs

fnlist = ['./molecule.xyz', '2-blocks/molecule.xyz', '3-blocks/molecule.xyz', '4-blocks/molecule.xyz']
for n, fni in enumerate(fnlist):
    fn = fni.split('/')[0]
    if n>0:
        nblocks = int(fn[0])
    else:
        nblocks=1
    mol, H = funcs.read(fni)
    H.geom.write('tmp.xyz')
    # Set initial polarization for triplet and singlet configurations
    dn = list(np.arange(19,19+42*(nblocks-1)+1,42)) + list(np.arange(21,21+42*(nblocks-1)+1, 42))
    dn.sort()
    up = list(np.arange(42,42+42*(nblocks-1)+1,42)) + list(np.arange(40,40+42*(nblocks-1)+1, 42)) 
    up.sort()
    up[-1] = up[-1]+1
    up[-2] = up[-2]+1

    H.set_polarization(up, dn=dn)
    nup_AFM, ndn_AFM = H.nup*1, H.ndn*1

    H.Nup +=1
    H.Ndn -=1

    up = up + [19, 21]
    dn = [i for i in dn if i not in up]
    H.set_polarization(up, dn=dn)
    nup_FM, ndn_FM = H.nup*1, H.ndn*1

    H.Nup -=1
    H.Ndn +=1

    f = open(fn+'/FM-AFM.dat', 'w')

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

        funcs.plot_spin(H, mol, fn+'/AFM-pol-%i.pdf'%(H.U*100))
        if u in [0., 3.0]:
            # Plot sepctrum at HOMO and LUMO
            funcs.plot_spectrum(fn, H, mol, fn+'/triangulene-AFM.nc')

        # Now FM case
        ncf = fn+'/triangulene-FM.nc'
        H.Nup += 1 # change to two more up-electrons than down
        H.Ndn -= 1

        H.nup, H.ndn = nup_FM, ndn_FM
        H.read_density(ncf)
        dn = H.converge(tol=1e-10, fn=ncf)
        eFM = H.Etot
        H.write_density(ncf)
        nup_FM, ndn_FM = H.nup*1, H.ndn*1

        funcs.plot_spin(H, mol, fn+'/FM-pol-%i.pdf'%(H.U*100))

        # Revert the imbalance for next loop
        H.Nup -= 1
        H.Ndn += 1

        f.write('%.4f %.8f %.8f\n'%(H.U, eFM, eAFM))

    f.close()

    # Plot FM-AFM energies
    FM_AFM = np.loadtxt(fn+'/FM-AFM.dat')
    U_list = FM_AFM[np.where(FM_AFM[:,0]<4.3)[0], 0] # First column are U values
    S1 = FM_AFM[np.where(FM_AFM[:,0]<4.3)[0], 1] # FM energies
    S0 = FM_AFM[np.where(FM_AFM[:,0]<4.3)[0], 2] # AFM energies
    p = plot.Plot()
    p.axes.plot(U_list, S1-S0, 'o')
    p.set_xlabel(r'$U$ [eV]')
    p.set_ylabel(r'$E_\mathrm{FM}-E_\mathrm{AFM}$ [eV]')
    p.savefig(fn+'/FM-AFM.pdf')
