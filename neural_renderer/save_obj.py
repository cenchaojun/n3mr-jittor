import os
import string

import numpy as np
import scipy.misc

def save_obj(filename, vertices, faces, textures=None):
    assert textures == None # TODO: allow save textures
    assert vertices.ndim == 2
    assert faces.ndim == 2

    with open(filename, 'w') as f:
        f.write('# %s\n' % os.path.basename(filename))
        f.write('#\n')
        f.write('\n')

        for vertex in vertices.numpy():
            f.write('v %.8f %.8f %.8f\n' % (vertex[0], vertex[1], vertex[2]))
        f.write('\n')

        for face in faces.numpy():
            f.write('f %d %d %d\n' % (face[0] + 1, face[1] + 1, face[2] + 1))