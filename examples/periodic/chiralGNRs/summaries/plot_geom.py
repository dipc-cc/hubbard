import Hubbard.plot as plot
import Hubbard.geometry as geom
import Hubbard.sp2 as sp2
import matplotlib.pyplot as plt

n,m,w = 3,1,8
geom = geom.cgnr(n,m,w).tile(3,axis=0)
Hsp2 = sp2(geom, t1=2.7, t2=0, t3=0)
p = plot.Bonds(Hsp2, figsize=(10,6), cmap=plt.cm.hot)
p.axes.axis('off')
p.savefig('bonds.pdf')