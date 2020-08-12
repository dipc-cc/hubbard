import matplotlib.pyplot as plt
from Hubbard.plot import Plot
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
    """

    def __init__(self, HH, bz=[([0.,0.,0.], r'$\Gamma$'), ([0.5,0.,0.], r'X')],  ymax=4.,  projection=None, scale=1, c='r', **keywords):

        # Set default keywords
        if 'figsize' not in keywords:
            keywords['figsize'] = (4, 8)

        super().__init__(**keywords)

        self.set_ylabel(r'$E_{nk}-E_\mathrm{mid}$ (eV)')
        self.set_ylim(-ymax, ymax)

        self.add_bands(HH, bz=bz, projection=projection, scale=scale, c=c)

    def add_bands(self, HH, bz=[([0.,0.,0.], r'$\Gamma$'), ([0.5,0.,0.], r'X')], projection=None, scale=1, c='r'):

        if isinstance(bz, sisl.BandStructure):
            band = bz
        else:
            path, labels = map(list, zip(*bz))
            band = sisl.BandStructure(HH.H, path, 101, labels)

        lk = band.lineark()
        xticks, xticks_labels = band.lineartick()
        ev = np.empty([2, len(lk), HH.sites])
        ev[0] = band.apply.array.eigh(spin=0)
        ev[1] = band.apply.array.eigh(spin=1)
        # Loop over k
        if projection != None:
            pdos = np.empty([2, len(lk), HH.sites])
            for ispin in range(2):
                for ik, k in enumerate(band.k):
                    _, evec = HH.eigh(k, eigvals_only=False, spin=ispin)
                    v = evec[tuple(projection)]
                    pdos[ispin, ik] = np.diagonal(np.dot(np.conjugate(v).T, v).real)
        # Set energy reference to the Fermi level
        Ef = HH.fermi_level(q=HH.q)
        ev[0] -= Ef[0]
        ev[1] -= Ef[1]
        # Make plot
        if not np.allclose(ev[0], ev[1]):
            # Add spin-down component to plotapply.
            plt.plot(lk, ev[1], 'g.')
        # Fat bands?
        if projection != None:
            for i in range(HH.sites):
                plt.errorbar(lk, ev[0, :, i], yerr=scale*pdos[0, :, i], alpha=.4, color='Grey')
        # Add spin-up component to plot (top layer)
        plt.plot(lk, ev[0], color=c)
        plt.gca().xaxis.set_ticks(xticks)
        plt.gca().set_xticklabels(xticks_labels)
        # Adjust borders
        plt.subplots_adjust(left=0.2, top=.95, bottom=0.1, right=0.95)
