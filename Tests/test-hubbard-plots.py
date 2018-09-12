import Hubbard.hubbard as HH

# Test all plot functionalities of Hubbard module
# using a reference molecule (already converged)

H = HH.Hubbard('mol-ref/mol-ref.XV')
H.U = 3.5
H.read()
H.plot_polarization()
H.plot_rs_polarization()
H.plot_wf(EnWindow=0.25,ispin=0)
H.plot_wf(EnWindow=0.25,ispin=1)
H.plot_rs_wf(EnWindow=0.25,ispin=0, z=0.5)
H.plot_rs_wf(EnWindow=0.25,ispin=1, z=0.5)