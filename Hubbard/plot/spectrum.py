from __future__ import print_function

import matplotlib.pyplot as plt
from Hubbard.plot import Plot
from Hubbard.plot import GeometryPlot
import numpy as np


class Spectrum(Plot):

    def __init__(self, HubbardHamiltonian, k=[0, 0, 0], xmax=10, ymax=0, **keywords):

        Plot.__init__(self, **keywords)
        self.axes.fill_between([-xmax, 0], 0, 1.0, facecolor='k', alpha=0.1)
        lmax = 0.0
        for ispin in range(2):
            ev, L = HubbardHamiltonian.calc_orbital_charge_overlaps(k=k, spin=ispin)
            ev -= HubbardHamiltonian.midgap
            L = np.diagonal(L)
            lmax = max(max(L), lmax)
            plt.plot(ev, L, 'rg'[ispin]+'.+'[ispin], label=[r'$\sigma=\uparrow$', r'$\sigma=\downarrow$'][ispin])
            
            if 'annotate' in keywords:
                if keywords['annotate'] != False:
                    for i in range(len(ev)):
                        self.axes.annotate(i, (ev[i], L[i]), fontsize=6)
        self.axes.legend()
        self.set_xlabel(r'$E_{\alpha\sigma}-E_\mathrm{mid}$ (eV)')
        self.set_ylabel(r'$\eta_{\alpha\sigma}=\int dr |\psi_{\alpha\sigma}|^4$')
        self.set_xlim(-xmax, xmax)
        if ymax == 0:
            self.set_ylim(0, lmax+0.01)
        else:
            self.set_ylim(0, ymax)


class LDOSmap(Plot):

    def __init__(self, HubbardHamiltonian, k=[0, 0, 0], spin=0, axis=0,
                 nx=501, gamma_x=1.0, dx=5.0, ny=501, gamma_e=0.05, ymax=10., vmax=None,
                 **keywords):

        Plot.__init__(self, **keywords)
        ev, evec = HubbardHamiltonian.eigh(k=k, eigvals_only=False, spin=spin)
        ev -= HubbardHamiltonian.midgap
        coord = HubbardHamiltonian.geom.xyz[:, axis]

        xmin, xmax = min(coord)-dx, max(coord)+dx
        ymin, ymax = -ymax, ymax
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)

        dat = np.zeros((len(x), len(y)))
        for i, evi in enumerate(ev):
            de = gamma_e/((y-evi)**2+gamma_e**2)/np.pi
            dos = np.zeros(len(x))
            for j, vj in enumerate(evec[:, i]):
                dos += abs(vj)**2*gamma_x/((x-coord[j])**2+gamma_x**2)/np.pi
            dat += np.outer(dos, de)
        intdat = np.sum(dat)*(x[1]-x[0])*(y[1]-y[0])
        print('Integrated LDOS spectrum (states within plot):', intdat)
        cm = plt.cm.hot
        self.axes.imshow(dat.T, extent=[xmin, xmax, ymin, ymax], cmap=cm, origin='lower', vmax=vmax)
        if axis==0:
            self.set_xlabel(r'$x$ (\AA)')
        elif axis==1:
            self.set_xlabel(r'$y$ (\AA)')
        elif axis==2:
            self.set_xlabel(r'$z$ (\AA)')
        self.set_ylabel(r'$E-E_\mathrm{midgap}$ (eV)')
        self.set_xlim(xmin, xmax)
        self.set_ylim(ymin, ymax)
        self.axes.set_aspect('auto')


class DOS_distribution(GeometryPlot):

    def __init__(self, HubbardHamiltonian, E, eta=1e-3, spin=[0,1], f=300, sites=[], ext_geom=None, **keywords):

        # Set default keywords
        if 'realspace' in keywords:
            if 'facecolor' not in keywords:
                keywords['facecolor'] = 'None'
            if 'cmap' not in keywords:
                keywords['cmap'] = 'Greys'
        else:
            if 'cmap' not in keywords:
                keywords['cmap'] = plt.cm.bwr

        GeometryPlot.__init__(self, HubbardHamiltonian.geom, ext_geom=ext_geom, **keywords)

        DOS = HubbardHamiltonian.DOS(E, eta=eta, spin=spin)
        x = HubbardHamiltonian.geom[:, 0]
        y = HubbardHamiltonian.geom[:, 1]

        if 'realspace' in keywords:
            self.__realspace__(DOS, density=True, vmin=0, **keywords)
            self.imshow.set_cmap(plt.cm.afmhot)

        else:
            self.axes.scatter(x, y, f*DOS, 'b')

        for i, s in enumerate(sites):
            self.axes.text(x[s], y[s], '%i'%i, fontsize=15, color='r')

    
class DOS(Plot):
    def __init__(self, HubbardHamiltonian, egrid, eta=1e-3, spin=[0,1], sites=[], **keywords):

        Plot.__init__(self, **keywords)
        
        DOS = HubbardHamiltonian.DOS(egrid, eta=eta, spin=spin)

        if np.any(sites):
            offset = 0.*np.average(DOS[sites[0]])
            for i, s in enumerate(sites):
                self.axes.plot(egrid, DOS[s]+offset*i, label='%i'%i)

        else:
            TDOS = DOS.sum(axis=0)
            plt.plot(egrid,TDOS,label='TDOS')

        self.set_xlabel(r'E-E$_\mathrm{midgap}$ [eV]')
        self.set_ylabel(r'DOS [1/eV]')
        self.axes.legend()
