import Hubbard.geometry as geom

for w in [8, 9]:

    for n in [3, 4]:
        for m in [1, 2]:
            geometry = geom.cgnr(n, m, w)
            geometry.tile(2, axis=0).write('cgnr-%i-%i-%i.xyz' % (n, m, w))

geometry = geom.add_Hatoms(geometry, sp3=[15])
geometry.tile(2, axis=0).write('cgnr-H.xyz')

geometry = geom.ssh()
geometry.tile(2, axis=0).write('ssh.xyz')
