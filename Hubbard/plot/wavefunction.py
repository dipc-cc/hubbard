from __future__ import print_function

import matplotlib.pyplot as plt
from Hubbard.plot import GeometryPlot


class Wavefunction(GeometryPlot):

    def __init__(self, HubbardHamiltonian, wf, **keywords):

        x = HubbardHamiltonian.pi_geom[:, 0]
        y = HubbardHamiltonian.pi_geom[:, 1]

        GeometryPlot.__init__(self, HubbardHamiltonian, cmap=plt.cm.bwr, **keywords)

        self.axes.scatter(x, y, wf.real, 'r') # pos. part, marker AREA is proportional to data
        self.axes.scatter(x, y, -wf.real, 'g') # neg. part
        # Write file
        if 'filename' not in keywords:
            fn = HubbardHamiltonian.get_label()+'-wf.pdf'
        else:
            fn = keywords['filename']
        self.savefig(fn)
        self.close()
