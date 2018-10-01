from __future__ import print_function

import matplotlib.pyplot as plt
from Hubbard.plot import GeometryPlot


class Wavefunction(GeometryPlot):

    def __init__(self, HubbardHamiltonian, wf, **keywords):

        GeometryPlot.__init__(self, HubbardHamiltonian, cmap=plt.cm.bwr, **keywords)

        x = HubbardHamiltonian.pi_geom[:, 0]
        y = HubbardHamiltonian.pi_geom[:, 1]

        assert len(x) == len(wf)

        self.axes.scatter(x, y, wf.real, 'r') # pos. part, marker AREA is proportional to data
        self.axes.scatter(x, y, -wf.real, 'g') # neg. part
