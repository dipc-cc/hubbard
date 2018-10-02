from __future__ import print_function

import matplotlib.pyplot as plt
from Hubbard.plot import GeometryPlot


class Wavefunction(GeometryPlot):

    def __init__(self, HubbardHamiltonian, wf, **keywords):

        # Set default keywords
        if 'cmap' not in keywords:
            keywords['cmap'] = plt.cm.bwr

        GeometryPlot.__init__(self, HubbardHamiltonian, **keywords)

        x = HubbardHamiltonian.geom[:, 0]
        y = HubbardHamiltonian.geom[:, 1]

        assert len(x) == len(wf)

        self.axes.scatter(x, y, wf.real, 'r') # pos. part, marker AREA is proportional to data
        self.axes.scatter(x, y, -wf.real, 'g') # neg. part
