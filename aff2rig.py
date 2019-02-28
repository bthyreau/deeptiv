import numpy as np
import sys
m=np.loadtxt(sys.argv[1])
u,s,vt = np.linalg.svd(m[:3,:3])
m2 = np.dot(u, vt)
outtfm = '#Insight Transform File V1.0\n#Transform 0\nTransform: AffineTransform_double_3_3\nParameters: ' + \
" ".join(["%4.8f" % x for x in m2.ravel()] + ["%4.8f" % x for x in m[:3,3].ravel()]) + \
'\nFixedParameters: 0 0 0\n'

open(sys.argv[1] + ".rigid.tfm", "w").write(outtfm)
