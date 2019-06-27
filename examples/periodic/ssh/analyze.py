from __future__ import print_function
import sisl
import numpy as np
import sys
import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import Hubbard.geometry as geometry
import os
import matplotlib.pyplot as plt

phase = sys.argv[1]
if phase in 'topological':
    d1, d2 = 2.0, 1.0
if phase in 'trivial':
    d1, d2 = 1.0, 2.0
geom = geometry.ssh(d1=d1, d2=d2)
geom = geom.move(geom.center(what='xyz'))

H = sisl.Hamiltonian(geom, dim=2)
for ia in geom:
    idx = geom.close(ia, R=[0.1, 1.1, 2.1])
    H[ia, idx[0]] = 0.
    H[ia, idx[1]] = 1.0 # 1NN
    H[ia, idx[2]] = 0.5 # 2NN

def analyze(H, nx=501):

    H = hh.HubbardHamiltonian(H, U=0.0, nkpt=[nx, 1, 1])
    if H.U > 0:
        H.random_density()
        H.converge()
    ymax = 2.0
    p = plot.Bandstructure(H, ymax=ymax)
    # Zak all filled bands
    zak = H.get_Zak_phase(Nx=nx)
    z2 = int(round(np.abs(1-np.exp(1j*zak))/2))
    p.set_title(r'[SSH-%s]'%phase)
    #p.axes.annotate(r'$\gamma=%.4f$'%zak, (0.4, 0.50), size=22, backgroundcolor='w')
    tol = 0.05
    p.axes.annotate(r'$\mathbf{Z_2=%i}$' % z2, (0., 0.9*ymax), size=22, backgroundcolor='k', color='w')
    p.axes.annotate(r'$\phi/\pi=%.2f$' % (zak/np.pi), (0.5, 0.9*ymax), size=16)
    p.savefig('SSH-%s_bands.pdf'%phase)

def analyze_edge(H):

    # Eigenvectors and eigenvalues in 1NN model for finite ribbon
    H = H.tile(15,axis=0)
    H.set_nsc([1,1,1])
    geom = H.geom
    ev, evec = H.eigh(eigvals_only=False)
    N = int(len(H)/2)
    p = plot.Plot()
    y1 = np.absolute(evec[:, N-1] )**2
    y2 = np.absolute(evec[:, N] )**2
    x = geom.xyz[:, 0]
    zipped = list(zip(x, y1, y2))
    zipped.sort(key = lambda t: t[0])
    x, y1, y2 = np.array(zipped)[:,0], np.array(zipped)[:,1], np.array(zipped)[:,2]
    p.axes.plot(x, y1, '-or', label=r'HOMO')
    p.axes.plot(x, y2, '--ob', label=r'LUMO')
    p.axes.legend(fontsize=13)
    p.set_ylabel(r'$|\Psi_{n}(x_{edge})|^{2}$ [a.u.]', fontsize=23)
    p.set_xlabel(r'$x$ [\AA]', fontsize=23)
    p.set_title(r'[SSH-%s]'%phase, fontsize=23)
    p.savefig('SSH-%s_edge_wf.pdf'%phase)
    p.close()

def band_symm(H, band=None, k=[0,0,0]):
    # It returns the parity of the VB and CB at Gamma and the edge of the BZ (X)
    geom = H.geom
    # Diagonalize Hamiltonian and store the eigenvectors obtained at Gamma and X
    ev, evec_0 = H.H.eigh(k=k, eigvals_only=False,spin=0)
    # Obtain sites of rotated symmetry
    geom_rot = geom.rotate(180, v=[0,0,1])
    geom_rot = geom_rot.move(-geom_rot.center())
    atom_list = []
    for ia in geom_rot:
        for ib in geom:
            if np.allclose(geom.xyz[ib], geom_rot.xyz[ia]):
                atom_list.append(ib)
    # Dot product between VB and CB of the rotated (by 180^o) and not rotated systems
    if not band:
        band = int(len(H)/2)-1
    symm = (np.conjugate(evec_0[atom_list, band]) * evec_0[:, band]).sum()
    return symm 

def plot_states(H, kpoints=[0.0,0.5]):
    #H = H.tile(2, axis=0)
    Hub = hh.HubbardHamiltonian(H)
    band_lab = ['VB', 'CB']
    k_lab = ['G', 'X']
    k_lab2 = ['\Gamma', 'X']
    N = int(len(H)/2)
    for ik, k in enumerate(kpoints):
        VB, CB = N-1, N
        ev, evec = H.eigh(k=[k,0,0],eigvals_only=False, spin=0)
        for ib, band in enumerate([VB, CB]):
            p = plot.Wavefunction(Hub, 3000*evec[:, band], colorbar=True, vmin=0)
            symm = band_symm(H, band=band, k=[k,0,0])
            p.set_title(r'[SSH-%s]: $ E_{%s}=%.1f$ eV'%(phase, k_lab2[ik],ev[band]), fontsize=23)
            p.axes.annotate(r'$\mathbf{Sym}=%.1f$'%(symm), (p.xmin+0.2, 0.87*p.ymax), size=18, backgroundcolor='k', color='w')
            p.savefig('SSH-%s_%s_%s.pdf'%(phase, band_lab[ib], k_lab[ik]))
            p.close()

def gap_exp(H, L=np.arange(1,31)):
    ev = np.zeros((len(np.linspace(0,0.5,51)), len(H)))
    for ik, k in enumerate(np.linspace(0,0.5,51)):
        ev[ik,:] = H.eigh(k=[k,0,0])
    N = int(len(H)/2)
    bg = min(ev[:, N] - ev[:, N-1])
    HL = []
    for pu in L:
        ribbon = H.tile(pu, axis=0)
        N = int(len(ribbon)/2)
        ribbon.set_nsc([1,1,1])
        ev = ribbon.eigh()
        HL.append(ev[N]-ev[N-1])
    
    p = plot.Plot(figsize=(10,6))
    p.set_title('HOMO-LUMO gap fitting [SSH-%s]'%phase, fontsize=23)
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
    ribbon.geometry.write('SSH-%s_ribbon.xyz'%phase)
    popt, pcov = curve_fit(pol_fit, x, y)
    p.axes.plot(x, pol_fit(x, *popt), color='r', label=r'fit $ax^{-b}+c$: a=%.3f, b=%.3f, c=%.3f'%tuple(popt))
    popt, pcov = curve_fit(exp_fit, x[5:], np.log(y[5:]))
    p.axes.plot(x[5:], np.exp(-x[5:]*popt[0] - popt[1]), color='g', label=r'fit: $e^{-\alpha x - \beta}: \alpha=%.3f, \beta=%.3f$'%tuple(popt))
    p.axes.legend(fontsize=16)
    p.set_xlabel(r'ch-GNR Length [p.u.]', fontsize=23)
    p.axes.set_xticks(np.arange(2,max(L),max(L)/6))
    p.set_ylabel(r'Energy Gap [eV]', fontsize=23)
    p.axes.set_yscale('log')
    p.savefig('SSH-%s_gap_fit.pdf'%phase)
    p.close()

def open_boundary(h, xlim=0.1):
    # Obtain the bulk and surface states of a Hamiltonian h
    from scipy import linalg as la
    H = h.copy()
    H.set_nsc([1,1,1])
    se_A = sisl.RecursiveSI(h, '-A')
    se_B = sisl.RecursiveSI(h, '+A')
    egrid = np.linspace(-xlim,xlim,500)
    dos_surf = []
    DOS_bulk = []
    for E in egrid:
        SEr_A = se_A.self_energy(E, eps=1e-50)
        SEr_B = se_B.self_energy(E, eps=1e-50)
        gs_A = la.inv((E+1e-4j)*np.identity(len(H)) - H.Hk().todense() - (SEr_A) )
        dos_surf.append(-(1/np.pi)*np.trace(gs_A).imag )
        #G = la.inv((E+1e-4j)*np.identity(len(H)) - H.Hk().todense() - (SEr_A+SEr_B) )
        G = se_A.green(E) # Sisl function to obtain BULK greens function
        DOS_bulk.append( -(1/np.pi)*np.trace(G).imag )

    p = plot.Plot()
    p.axes.plot(egrid, dos_surf, label='Surface DOS')
    p.axes.plot(egrid, DOS_bulk, label='Bulk DOS')
    p.axes.legend()
    #p.set_ylim(0,50)
    p.set_xlabel(r'Energy [eV]', fontsize=23)
    p.set_ylabel(r'DOS [eV$^{-1}$]', fontsize=23)
    p.set_title(r'Density of states [SSH-%s]'%phase, fontsize=23)
    p.savefig('SSH-%s_DOS.pdf'%phase)

open_boundary(H, xlim=1.0)
gap_exp(H)
analyze(H)
analyze_edge(H)
plot_states(H)
