from __future__ import print_function

from Hubbard.plot import GeometryPlot

class Charge(GeometryPlot):

    def __init__(self, HubbardHamiltonian):
        
        GeometryPlot.__init__(self, HubbardHamiltonian)

        # Compute total charge on each site
        charge = HubbardHamiltonian.nup + HubbardHamiltonian.ndn

        # Set values for the pi-network
        self.ppi.set_array(charge)

        # Set colorbar limits
        self.ppi.set_clim(min(charge), max(charge))

        # Write file
        fn = HubbardHamiltonian.get_label()+'-chg.pdf'
        self.savefig(fn)

        # Close plot
        self.close()
