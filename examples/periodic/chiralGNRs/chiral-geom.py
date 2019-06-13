import sisl
import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import numpy as np
import os

def cgnr(n, m, w, d=1.42):
    "Generation of chiral GNR geometry (periodic along x-axis)"
    g0 = sisl.geom.graphene()
    g = sisl.geom.graphene(orthogonal='True')
    g = g.tile(n+1, 1)
    g = g.remove(3).remove(0)
    g = g.repeat(w/2, 0)
    g.cell[1] += g0.cell[1]
    if m > 1:
        g.cell[1, 0] += 3*(m-1)*d
    cs = np.cos(np.pi/3)
    sn = np.sin(np.pi/3)
    A1 = d*(1.+cs)*(2.*(m-1)+1.)
    A2 = d*(n+0.5)*sn*2.
    theta = np.arctan(A2/A1)
    gr = g.rotate(theta*360/(2*np.pi), v=[0,0,1])
    gr.set_sc([A2*np.sin(theta)+A1*np.cos(theta), 10, 10])
    gr.set_nsc([3,1,1])
    # Move center-of-mass to origo
    gr = gr.move(-gr.center())
    return gr

def analyze(geom, directory, nx=1001):
    geom.write(directory+'/cgnr.xyz')
    geom.repeat(3, 0).write(directory+'/cgnr-rep.xyz')
    H = hh.HubbardHamiltonian(geom, t1=2.7, t2=0., t3=0., U=0.0, kmesh=[nx, 1, 1])
    ymax = 8.0
    p = plot.Bandstructure(H, ymax=ymax)
    p.set_title(r'%s: $n_x=%i$'%(directory,nx))
    # Zak one band only
    zak1 = H.get_Zak_phase(Nx=nx, sub=H.Nup)
    z21 = int(round(np.abs(1-np.exp(1j*zak1))/2))
    zak1b = H.get_Zak_phase(Nx=nx+1, sub=H.Nup)
    z21b = int(round(np.abs(1-np.exp(1j*zak1b))/2))
    assert z21 == z21b
    # Zak all filled bands
    zak = H.get_Zak_phase(Nx=nx)
    z2 = int(round(np.abs(1-np.exp(1j*zak))/2))
    print('%s: z21=%i [z2-all=%i]'%(directory, z21,z2))
    #assert z21 == z2
    #p.axes.annotate(r'$\gamma=%.4f$'%zak, (0.4, 0.50), size=22, backgroundcolor='w')
    tol = 0.05
    if np.abs(zak) < tol or np.abs(np.abs(zak)-np.pi) < tol:
        # Only append Z2 when appropriate:
        p.axes.annotate(r'$\mathbf{Z_2=%i (%i)}$'%(z21, z2), (0., 0.9*ymax), size=22, backgroundcolor='k', color='w')
    p.savefig(directory+'/bands_1NN.pdf')

def analyze_edge(geom, directory):
    # Create 15 length ribbon
    geom = geom.tile(15, axis=0)
    # Identify edge sites along the lower ribbon border
    sites = []
    for ia in geom:
        idx = geom.close(ia, R=[0.1, 1.43])
        if len(idx[1]) == 2 :
            if geom.xyz[ia, 1] < 0:
                sites.append(ia)

    # Eigenvectors and eigenvalues in 1NN model for finite ribbon
    geom.set_nsc([1,1,1])
    H = hh.HubbardHamiltonian(geom, t1=2.7, t2=0., t3=0., U=0.)        
    ev, evec = H.H.eigh(eigvals_only=False,spin=0)
    ev -= H.midgap

    p = plot.Plot()
    y1 = np.absolute(evec[sites, H.Nup-1] )**2
    y2 = np.absolute(evec[sites, H.Nup] )**2
    x = geom.xyz[sites, 0]
    zipped = zip(x, y1, y2)
    zipped.sort(key = lambda t: t[0])
    x, y1, y2 = np.array(zipped)[:,0], np.array(zipped)[:,1], np.array(zipped)[:,2]
    p.axes.plot(x, y1, '-or', label=r'HOMO')
    p.axes.plot(x, y2, '--ob', label=r'LUMO')
    p.axes.legend(fontsize=13)
    p.set_ylabel(r'$|\Psi_{n}(x_{edge})|^{2}$ [a.u.]')
    p.set_xlabel(r'$x$ [\AA]')
    p.set_title('[%s]'%directory)
    p.savefig(directory+'/1NN_edge_wf.pdf')

    if True:
        # Plot edge sites?
        v = np.zeros(len(H.geom))
        v[sites] = 1.
        p = plot.GeometryPlot(H, cmap='Reds', figsize=(10,3))
        p.__orbitals__(v, vmax=1.0, vmin=0)
        p.set_title(r'Edge sites of [%s]'%directory)
        p.savefig(directory+'/edge_sites.pdf')

def band_sym(H, band=None, k=[0,0,0]):
    # It returns the parity of a band at a desired k-point
    geom = H.geom
    # Diagonalize Hamiltonian and store the eigenvectors obtained at the desired k-point
    ev, evec_0 = H.H.eigh(k=k, eigvals_only=False,spin=0)
    # Obtain sites of rotated geometry
    geom_rot = geom.rotate(180, v=[0,0,1])
    geom_rot = geom_rot.move(-geom_rot.center())
    atom_list = []
    for ia in geom_rot:
        for ib in geom:
            if np.allclose(geom.xyz[ib], geom_rot.xyz[ia]):
                atom_list.append(ib)
    if not band:
        band = len(H)/2-1 # VB
    sym = (evec_0[atom_list, band] * evec_0[:, band]).sum()
    return sym

def print_band_sym(geom):
    H = hh.HubbardHamiltonian(geom, t1=2.7, t2=0., t3=0., U=0.)
    VB = len(H)/2-1
    CB = len(H)/2
    VB_G = band_sym(H,k=[0.0,0,0], band=VB)
    VB_X = band_sym(H,k=[0.5,0,0], band=VB)
    CB_G = band_sym(H,k=[0.0,0,0], band=CB)
    CB_X = band_sym(H,k=[0.5,0,0], band=CB)
    VB_1_G = band_sym(H,k=[0.0,0,0], band=VB-1)
    VB_1_X = band_sym(H,k=[0.5,0,0], band=VB-1)
    CB_1_G = band_sym(H,k=[0.0,0,0], band=CB+1)
    CB_1_K = band_sym(H,k=[0.5,0,0], band=CB+1)
    print(' Band     Gamma        X \n')
    print(' VB-1 : %.2f+%.2fi %.2f+%.2fi \n'%(VB_1_G.real, VB_1_G.imag, VB_1_X.real, VB_1_X.imag))
    print(' VB : %.2f+%.2fi %.2f+%.2fi \n'%(VB_G.real, VB_G.imag, VB_X.real, VB_X.imag))
    print(' CB : %.2f+%.2fi %.2f+%.2fi \n'%(CB_G.real, CB_G.imag, CB_X.real, CB_X.imag))
    print(' CB+1 : %.2f+%.2fi %.2f+%.2fi \n'%(CB_1_G.real, CB_1_G.imag, CB_1_K.real, CB_1_K.imag))   

def plot_states(geom, directory):
    band_lab = ['VB', 'CB']
    k_lab = ['G', 'X']
    k_lab2 = ['\Gamma', 'X']
    for ik, k in enumerate([0,0.5]):
        H = hh.HubbardHamiltonian(geom, t1=2.7, t2=0., t3=0., U=0)
        VB, CB = H.Nup-1, H.Nup
        ev, evec = H.eigh(k=[k,0,0],eigvals_only=False, spin=0)
        for ib, band in enumerate([VB, CB]):
            p = plot.Wavefunction(H, 3000*evec[:, band], colorbar=True)
            sym = band_sym(H, band=band, k=[k,0,0])
            p.set_title(r'[%s]: $ E_{%s}=%.1f$ meV'%(directory, k_lab2[ik],ev[band]*1000))
            p.axes.annotate(r'$\mathbf{Sym}=%.1f$'%(sym), (p.xmin+0.2, 0.87*p.ymax), size=18, backgroundcolor='k', color='w')
            p.savefig(directory+'/%s_%s.pdf'%(band_lab[ib], k_lab[ik]))

def gap_exp(geom, directory, L=np.arange(1,31)):
    H = hh.HubbardHamiltonian(geom, t1=2.7, t2=0., t3=0., U=0.)
    ev = np.zeros((len(np.linspace(0,0.5,51)), len(H)))
    for ik, k in enumerate(np.linspace(0,0.5,51)):
        ev[ik,:] = H.H.eigh(k=[k,0,0],spin=0)
    bg = min(ev[:, H.Nup] - ev[:, H.Nup-1])
    HL = []
    HL_1 = []
    for pu in L:
        ribbon = geom.tile(pu, axis=0)
        ribbon.set_nsc([1,1,1])
        H = hh.HubbardHamiltonian(ribbon, t1=2.7, t2=0., t3=0., U=0.)
        ev = H.H.eigh(spin=0)
        HL.append(ev[H.Nup]-ev[H.Nup-1])
        HL_1.append(ev[H.Nup+1]-ev[H.Nup-2])
    
    p = plot.Plot(figsize=(10,6))
    p.set_title('HOMO-LUMO gap fitting [%s]'%(directory))
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
    p.set_xlabel(r'ch-GNR Length [p.u.]')
    p.axes.set_xticks(np.arange(2,max(L),max(L)/6))
    p.set_ylabel(r'Energy Gap [eV]')
    p.axes.set_yscale('log')
    p.savefig(directory+'/gap_fit.pdf')

