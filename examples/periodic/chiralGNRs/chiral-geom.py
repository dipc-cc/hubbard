import sisl
import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.sp2 as sp2
import netCDF4 as NC
import numpy as np
import os

def analyze(H0, directory, nx=51):
    H = hh.HubbardHamiltonian(H0, U=0.0, nkpt=[nx, 1, 1])
    if H.U > 0:
        H.random_density()
        H.converge()
    ymax = 8.0
    p = plot.Bandstructure(H, ymax=ymax)
    # Zak all filled bands
    zak = H.get_Zak_phase(Nx=nx)
    z2 = int(round(np.abs(1-np.exp(1j*zak))/2))
    p.set_title(r'[%s]'%directory)
    #p.axes.annotate(r'$\gamma=%.4f$'%zak, (0.4, 0.50), size=22, backgroundcolor='w')
    tol = 0.05
    p.axes.annotate(r'$\mathbf{Z_2=%i}$' % z2, (0., 0.9*ymax), size=22, backgroundcolor='k', color='w')
    p.axes.annotate(r'$\phi/\pi=%.2f$' % (zak/np.pi), (0.5, 0.9*ymax), size=16)
    p.savefig(directory+'/bands_1NN.pdf')

def analyze_edge(H0, directory):
    # Create 15 length ribbon
    geom = H0.geometry
    # Identify edge sites along the lower ribbon border
    geom.set_nsc([3,1,1])
    sites = []
    for ia in geom:
        idx = geom.close(ia, R=[0.1, 1.43])
        if len(idx[1]) == 2 :
            if geom.xyz[ia, 1] < 0:
                sites.append(ia)

    # Eigenvectors and eigenvalues in 1NN model for finite ribbon
    H = hh.HubbardHamiltonian(H0, U=0.)
    ev, evec = H.eigh(eigvals_only=False, spin=0)
    ev -= H.midgap

    p = plot.Plot()
    y1 = np.absolute(evec[sites, H.Nup-1] )**2
    y2 = np.absolute(evec[sites, H.Nup] )**2
    x = geom.xyz[sites, 0]
    zipped = list(zip(x, y1, y2))
    zipped.sort(key = lambda t: t[0])
    x, y1, y2 = np.array(zipped)[:,0], np.array(zipped)[:,1], np.array(zipped)[:,2]
    p.axes.plot(x, y1, '-or', label=r'HOMO')
    p.axes.plot(x, y2, '--ob', label=r'LUMO')
    p.axes.legend(fontsize=13)
    p.set_ylabel(r'$|\Psi_{n}(x_{edge})|^{2}$ [a.u.]', fontsize=23)
    p.set_xlabel(r'$x$ [\AA]', fontsize=23)
    p.set_title('[%s]'%directory, fontsize=23)
    p.savefig(directory+'/1NN_edge_wf.pdf')

    if True:
        # Plot edge sites?
        v = np.zeros(len(H.geom))
        v[sites] = 1.
        p = plot.GeometryPlot(H.geom, cmap='Reds', figsize=(10,3))
        p.__orbitals__(v, vmax=1.0, vmin=0)
        p.set_title(r'Edge sites of [%s]'%directory, fontsize=23)
        p.savefig(directory+'/edge_sites.pdf')

def plot_states(H0, directory):
    band_lab = ['VB', 'CB']
    k_lab = ['G', 'X']
    k_lab2 = ['\Gamma', 'X']
    H = hh.HubbardHamiltonian(H0, U=0.)
    for ik, k in enumerate([0, 0.5]):
        VB, CB = H.Nup-1, H.Nup
        ev, evec = H.eigh(k=[k, 0, 0], eigvals_only=False, spin=0)
        for ib, band in enumerate([VB, CB]):
            p = plot.Wavefunction(H, 3000*evec[:, band], colorbar=True)
            sym = H.band_sym(evec[:, band])[0]
            p.set_title(r'[%s]: $ E_{%s}=%.1f$ meV'%(directory, k_lab2[ik],ev[band]*1000), fontsize=23)
            p.axes.annotate(r'$\mathbf{Sym}=%.1f$'%(sym), (p.xmin+0.2, 0.87*p.ymax), size=18, backgroundcolor='k', color='w')
            p.savefig(directory+'/%s_%s.pdf'%(band_lab[ib], k_lab[ik]))

def gap_exp(H0, directory, L=np.arange(1,31)):
    H = hh.HubbardHamiltonian(H0, U=0.)
    ev = np.zeros((len(np.linspace(0,0.5,51)), len(H0)))
    for ik, k in enumerate(np.linspace(0,0.5,51)):
        ev[ik,:] = H.H.eigh(k=[k,0,0],spin=0)
    bg = min(ev[:, H.Nup] - ev[:, H.Nup-1])
    HL = []
    HL_1 = []
    for pu in L:
        ribbon = H0.tile(pu, axis=0)
        ribbon.set_nsc([1,1,1])
        H = hh.HubbardHamiltonian(ribbon, U=0.)
        ev = H.eigh(spin=0)
        HL.append(ev[H.Nup]-ev[H.Nup-1])
        HL_1.append(ev[H.Nup+1]-ev[H.Nup-2])
    
    p = plot.Plot(figsize=(10,6))
    p.set_title('HOMO-LUMO gap fitting [%s]'%(directory), fontsize=23)
    p.axes.axhline(y=bg, linestyle='--', color='k', linewidth=0.75, label='Inf. Bandgap: %.3f eV'%(bg))
    from scipy.optimize import curve_fit
    # Define fitting functions
    def exp_fit(x, a, b):
        return -a * x - b
    def pol_fit(x, a, b, c):
        return a * x**(-b) + c
     
    x, y = L, HL
    p.axes.plot(x, y, 'ok', label='LUMO-HOMO')
    #p.axes.plot(x, HL_1, '^b', label='(L+1)-(H-1)')
    popt, pcov = curve_fit(pol_fit, x, y)
    p.axes.plot(x, pol_fit(x, *popt), color='r', label=r'fit $ax^{-b}+c$: a=%.3f, b=%.3f, c=%.3f'%tuple(popt))
    popt, pcov = curve_fit(exp_fit, x[3:], np.log(y[3:]))
    p.axes.plot(x[3:], np.exp(-x[3:]*popt[0] - popt[1]), color='g', label=r'fit: $e^{-\alpha x - \beta}: \alpha=%.3f, \beta=%.3f$'%tuple(popt))
    p.axes.legend(fontsize=16)
    p.set_xlabel(r'ch-GNR Length [p.u.]', fontsize=23)
    p.axes.set_xticks(np.arange(2,max(L),max(L)/6))
    p.set_ylabel(r'Energy Gap [eV]', fontsize=23)
    p.axes.set_yscale('log')
    p.savefig(directory+'/gap_fit.pdf')

def phase_diagram(w=8):

    import Hubbard.geometry as geometry

    def get_Zak_phase(geom):
        H0 = sp2(geom, t1=2.7, t2=0., t3=0.)
        H = hh.HubbardHamiltonian(H0, U=0.)
        zak = H.get_Zak_phase(Nx=101)
        if not (abs(zak)>1e-3 or abs(abs(zak)-np.pi)>1e-3):
            print('zak:', zak)
            print('Warning: Zak phase not quantized!')
        z2 = int(round(np.abs(1-np.exp(1j*zak))/2))
        # Obtain band gap
        ev = np.zeros((len(np.linspace(0,0.5,51)), len(H.geom)))
        for ik, k in enumerate(np.linspace(0,0.5,51)):
            ev[ik,:] = H.eigh(k=[k,0,0],spin=0)

        bg = min(ev[:, H.Nup] - ev[:, H.Nup-1])
        return bg, z2

    nlist = np.arange(2, 11)
    mlist = np.arange(1,int(w/2)+1)

    # Build empty matrix to store bandgap and Z2
    band_gap_matrix = np.zeros((len(nlist), len(mlist),  2))
    for i_n, n in enumerate(nlist):
        for i_m, m in enumerate(mlist):
            print(n,m,w)
            geom = geometry.cgnr(n,m,w)
            bg, z2  = get_Zak_phase(geom)
            band_gap_matrix[i_n, i_m, 0] = bg
            band_gap_matrix[i_n, i_m, 1] = z2
    
    fn = 'band_gap_zak_W%i.nc'%w
    ncf = NC.Dataset(fn, 'w')

    # Create dimensions
    ncf.createDimension('nlen', len(nlist))
    ncf.createDimension('mlen', len(mlist))
    ncf.createDimension('data', 2)
    # Create variables
    ncf.createVariable('BG', 'f8', ('nlen', 'mlen', 'data'))
    ncf.createVariable('mlist', 'i8', ('mlen'))
    ncf.createVariable('nlist', 'i8', ('nlen'))

    ncf['BG'][:] = band_gap_matrix 
    ncf['mlist'][:] = mlist
    ncf['nlist'][:] = nlist

    ncf.close()

def plot_band_gap_imshow(w=8, figsize=(8,7)):
    fn = 'band_gap_zak_W%i.nc'%w
    try:
        ncf = NC.Dataset(fn, 'r')
    except:
        phase_diagram(w=w)
        ncf = NC.Dataset(fn, 'r')
    band_gap_matrix = ncf['BG'][:]
    m = ncf['mlist'][:]
    n = ncf['nlist'][:]

    p = plot.Plot(figsize=figsize)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(p.axes)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    import matplotlib.colors as mcolors
    bg = band_gap_matrix[:,:,0]
    z2 = band_gap_matrix[:,:,1]

    bg = ((-1)**z2)*bg # multiply band gap by (-1)^z2
    lim = max(abs(np.max(bg)), abs(np.min(bg)))
    extent = [0, len(n), 0, len(m)]
    sc = p.axes.imshow(bg.T, cmap='seismic', origin='lower', vmax=lim, vmin=-lim, extent=extent, 
                        norm=mcolors.SymLogNorm(linthresh=0.03), aspect='equal')
    p.set_xlabel('n', fontsize=25)
    p.set_ylabel('m', fontsize=25)
    p.axes.set_xticks(np.arange(0, len(n))+0.5)
    p.axes.set_xticklabels(range(int(min(n)), int(max(n))+1), fontsize=20)
    p.axes.set_yticks(np.arange(0, len(m))+0.5)
    p.axes.set_yticklabels(range(int(min(m)), int(max(m))+1), fontsize=20)
    p.fig.colorbar(sc, cax=cax, label=r'$E$ [eV]')
    p.set_title(r'Band gap for $W=%i$'%w, fontsize=25)
    p.savefig('W%i_band_gap_imshow.pdf'%w)
