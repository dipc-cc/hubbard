from __future__ import print_function

import matplotlib.pyplot as plt
from Hubbard.plot import GeometryPlot
import sisl
import numpy as np


class Wavefunction(GeometryPlot):

    def __init__(self, HubbardHamiltonian, wf, **keywords):

        # Set default keywords
        if 'realspace' in keywords:
            if 'facecolor' not in keywords:
                keywords['facecolor'] = 'None'
            if 'cmap' not in keywords:
                keywords['cmap'] = 'Greys'
        else:
            if 'cmap' not in keywords:
                keywords['cmap'] = plt.cm.bwr

        GeometryPlot.__init__(self, HubbardHamiltonian, **keywords)

        if 'realspace' in keywords:
            self.__realspace__(wf, **keywords)

            # Create custom map to differenciate it from polarization cmap
            import matplotlib.colors as mcolors
            custom_map = mcolors.LinearSegmentedColormap.from_list(name='custom_map', colors =['g', 'white', 'red'], N=100)
            self.imshow.set_cmap(custom_map)

        else:
            x = HubbardHamiltonian.geom[:, 0]
            y = HubbardHamiltonian.geom[:, 1]

            assert len(x) == len(wf)

            self.axes.scatter(x, y, wf.real, 'r') # pos. part, marker AREA is proportional to data
            self.axes.scatter(x, y, -wf.real, 'g') # neg. part

