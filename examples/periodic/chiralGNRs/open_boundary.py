import sisl
import Hubbard.hamiltonian as hh
import Hubbard.plot as plot
import numpy as np
from scipy import linalg as la
import os

def open_boundary(h, directory, xlim=0.1):
    # Obtain the bulk and surface states of a Hamiltonian h
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
    p.set_xlabel(r'Energy [eV]')
    p.set_ylabel(r'DOS [eV$^{-1}$]')
    p.set_title(r'Density of states [%s]'%directory)
    p.savefig(directory+'/DOS.pdf')
