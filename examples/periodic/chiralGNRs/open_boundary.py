import sisl
import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import numpy as np
from scipy import linalg as la
import os

def open_boundary(h, directory, xlim=0.1, U=0, model='1NN'):
    # Obtain the bulk and surface states of a Hamiltonian h
    H = hh.HubbardHamiltonian(h, U=U)
    if U>0:
        dn = H.converge()
    Hsurf = H.H.copy()
    Hsurf.set_nsc([1,1,1])
    se_A = sisl.RecursiveSI(H.H, '-A')
    se_B = sisl.RecursiveSI(H.H, '+A')
    H.find_midgap()
    egrid = np.linspace(H.midgap-xlim, H.midgap+xlim,500)
    dos_surf = []
    DOS_bulk = []
    eta = 1e-3
    for E in egrid:
        # Find self-energies for both spin channels
        SEr_A_up = se_A.self_energy(E, spin=0)
        SEr_A_dn = se_A.self_energy(E, spin=1)
        SEr_B_up = se_B.self_energy(E, spin=0)
        SEr_B_dn = se_B.self_energy(E, spin=1)
        # Find surface and bulk Greens function for both spin channels
        gs_A_up = la.inv((E+eta*1j)*np.identity(len(Hsurf)) - Hsurf.Hk(spin=0).todense() - (SEr_A_up) )
        gs_A_dn = la.inv((E+eta*1j)*np.identity(len(Hsurf)) - Hsurf.Hk(spin=1).todense() - (SEr_A_dn) )
        dos_surf.append(-(1/np.pi)*np.trace(gs_A_up).imag-(1/np.pi)*np.trace(gs_A_dn).imag )
        G_up = la.inv((E+eta*1j)*np.identity(len(Hsurf)) - Hsurf.Hk(spin=0).todense() - (SEr_A_up+SEr_B_up) )
        G_dn = la.inv((E+eta*1j)*np.identity(len(Hsurf)) - Hsurf.Hk(spin=1).todense() - (SEr_A_dn+SEr_B_dn) )
        DOS_bulk.append( -(1/np.pi)*np.trace(G_up).imag -(1/np.pi)*np.trace(G_dn).imag)

    p = plot.Plot()
    p.axes.plot(egrid-H.midgap, dos_surf, 'r', label='Surface DOS')
    p.axes.plot(egrid-H.midgap, DOS_bulk, 'k', label='Bulk DOS')
    p.axes.legend()
    #p.set_ylim(0,50)
    p.set_xlabel(r'Energy [eV]')
    p.set_ylabel(r'DOS [eV$^{-1}$]')
    p.set_title(r'Density of states [%s]'%directory)
    p.savefig(directory+'/%s_DOS_U%i.pdf'%(model, U*100))
