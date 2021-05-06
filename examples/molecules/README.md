Simulating a molecule with electron correlations
================================================

In this example we will create the HubbardHamiltonian object of a finite
molecule, and find the self-consistent solution.

We will use as a reference the molecule of ref. [Nature Communications
10, 200 (2019)](https://www.nature.com/articles/s41467-018-08060-6) and
compare the simulation with the experimental system.

You can also navigate through the [molecule examples
section](https://github.com/dipc-cc/hubbard/tree/master/examples/molecules),
where more molecular geometries suitable to compute with the mean-field
Hubbard model can be found.

1.  We will start by building the tight-binding (TB) Hamiltonian for an
    sp2 carbon-based molecule, by first reading the geometry file stored
    in this
    [file](https://github.com/dipc-cc/hubbard/blob/master/examples/molecules/kondo-paper/junction-2-2.XV)
    (which you can download). You can find the parameters used to model
    this sp2 TB Hamiltonian in the [Supp.
    Material](https://www.nature.com/articles/s41467-018-08060-6#Sec12)
    of the paper referenced above.
2.  We will build the HubbardHamiltonian object, which will allow us to
    use the routines stored in this class to converge the mean-field
    Hubbard Hamiltonian until we find the self-consistent solution.
3.  We then can manipulate and obtain different magnetic states to
    compare the total energies that will tell us which one is the
    groundstate.
4.  You can try to reproduce more calculations by looking at the [Supp.
    Material](https://www.nature.com/articles/s41467-018-08060-6#Sec12)
    of the above referenced paper.

``` {.sourceCode .python}
import sisl
from hubbard import HubbardHamiltonian, sp2, density, plot

# Build sisl.Geometry object from the 'XV' file (previously downloaded)
g = sisl.get_sile('junction-2-2.XV').read_geometry()

# Build sisl.Hamiltonian object using the sp2 function
Hsp2 = sp2(g, t1=2.7, t2=0.2, t3=0.18)

# Build the HubbardHamiltonian object with U=3.5 at a temperature of kT=1e-5
# in units of the Boltzmann constant
HH = HubbardHamiltonian(Hsp2, U=3.5, kT=1e-5)

# Let's initiate with a custom spin density distribution
# by placing up-spin components in atom 23 and down spin components in atom 77
HH.set_polarization([23],dn=[77])

# Converge until a tolerance of tol=1e-10
dn = HH.converge(density.calc_n, tol=1e-10)

# Save total energy
E_0 =  HH.Etot

# Let's visualize the final result:
p = plot.SpinPolarization(HH, colorbar=True, vmax=0.4, vmin=-0.4)
p.savefig('spin.pdf')

# Until now, in absence of specifications we have found the antiferromagnetic solution
# (same up- and down-spin particles). But we can compute the solution for the
# ferromagnetic case, by imposing Qup = Qdn + 2 and converge again

HH.q[0] += 1
HH.q[1] -= 1

# Converge until a tolerance of tol=1e-10
dn = HH.converge(density.calc_n, tol=1e-10)

# Let's visualize some the final result:
p = plot.SpinPolarization(HH, colorbar=True, vmax=0.4, vmin=-0.4)
p.savefig('spin-ferro.pdf')

# Compare energies between the two calculations
print('E_FM - E_AFM : ', HH.Etot - E_0, ' eV')
```
