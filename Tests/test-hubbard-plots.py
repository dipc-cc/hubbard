import Hubbard.hubbard as HH
import numpy as np

# Test all plot functionalities of Hubbard module
# using a reference molecule (already converged)

H = HH.Hubbard('mol-ref/mol-ref.XV', U=3.5)

# Produce plots
H.plot_spectrum()
H.plot_charge()
H.plot_polarization()
H.plot_rs_polarization()
H.plot_wf(EnWindow=0.25, ispin=0)
H.plot_wf(EnWindow=0.25, ispin=1)
H.plot_wf(EnWindow=0.25, ispin=1, density=False)
H.plot_rs_wf(EnWindow=0.25, ispin=0, z=0.5)
H.plot_rs_wf(EnWindow=0.25, ispin=1, z=0.5)
H.plot_rs_wf(EnWindow=0.25, ispin=1, z=0.5, density=False)

