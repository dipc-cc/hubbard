import matplotlib.pyplot as plt
from hubbard.plot import Plot
import sisl
import numpy as np

__all__ = ['Bandstructure']


class Bandstructure(Plot):
    """ Plot the bandstructure for the `HubbardHamiltonian` object along certain path of the Brillouin Zone

    Parameters
    ----------
    bz: list of tuples or `sisl.BandStructure`, optional
        k points to build the path along the first Brillouin Zone with their corresponding string labels ``[(k1, label1),(k2, label2)]``,
        or directly pass the `sisl.BandStructure` object
    projection: list, optional
        sites to project the bands onto
    scale: float, optional
        in case ``projection!=None``, scale controls the size of the error bar to plot the projection onto the bands
    ymax: float, optional
        maximum point in the vertical axis
    spin: int, or list of int optional
        to plot the bandstructure corresponding to the spin index (0, or 1) or plot simultaneusly both spin indices by passing `spin=[0,1]`
        Defaults to 0
    c: str, or list of str optional
        color for the bandstructure lines. If a list is passed it plots each spin component with the selected colors,
        otherwise it defaults to red and green for spin indices 0 and 1, respectively. See `matplotlib color setting <https://matplotlib.org/stable/tutorials/colors/colors.html>`_
    """

    def __init__(self, HH, bz=[([0., 0., 0.], r'$\Gamma$'), ([0.5, 0., 0.], r'X')],  ymax=4., projection=None, scale=1, c='r', spin=0, **kwargs):

        # Set default kwargs
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (4, 8)

        super().__init__(**kwargs)

        self.set_ylabel(r'$E_{nk}-E_F$ (eV)')
        self.set_ylim(-ymax, ymax)

        self.add_bands(HH, bz=bz, projection=projection, scale=scale, c=c, spin=spin)

    def add_bands(self, HH, bz=[([0., 0., 0.], r'$\Gamma$'), ([0.5, 0., 0.], r'X')], projection=None, scale=1, c='r', spin=0):

        if isinstance(bz, sisl.BandStructure):
            band = bz
        else:
            path, labels = map(list, zip(*bz))
            band = sisl.BandStructure(HH.H, path, 101, labels)

        lk = band.lineark()
        xticks, xticks_labels = band.lineartick()
        if not isinstance(spin, (list,tuple,np.ndarray)):
            spin = [spin]
        if not isinstance(c, (list, tuple)):
                c = [c]
        if len(spin)>1 and len(c)==1:
            if isinstance(c, list):
                c = c + ['g.']
            elif isinstance(c, tuple):
                c = c + ('g.',)

        ev = np.empty([len(spin), len(lk), HH.sites])
        # Set energy reference to the Fermi level
        Ef = HH.fermi_level(q=HH.q)
        for i,s in enumerate(spin):
            ev[i] = band.apply.array.eigh(spin=s)
            ev[i] -= Ef[s]
            # Add spin-up component to plot (top layer)
            self.axes.plot(lk, ev[i], c[i])

        # Loop over k
        if projection != None:
            pdos = np.empty([len(spin), len(lk), HH.sites])
            for ispin in spin:
                for ik, k in enumerate(band.k):
                    _, evec = HH.eigh(k, eigvals_only=False, spin=ispin)
                    v = evec[tuple(projection), :]
                    pdos[ispin, ik] = np.diagonal(np.dot(np.conjugate(v).T, v).real)

        # Fat bands?
        if projection != None:
            for i in range(HH.sites):
                self.axes.errorbar(lk, ev[0, :, i], yerr=scale*pdos[0, :, i], alpha=.4, color='Grey')
        # Figure aspects
        self.axes.xaxis.set_ticks(xticks)
        self.axes.set_xticklabels(xticks_labels)
        # Adjust borders
        self.fig.subplots_adjust(left=0.2, top=.95, bottom=0.1, right=0.95)
