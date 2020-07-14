import os

import jittor as jt

import neural_renderer as nr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

def to_minibatch(data, batch_size=4, target_num=2):
    ret = []
    for d in data:
        d2 = jt.zeros(d.shape).unsqueeze(0).broadcast([batch_size, *d.shape])
        d2[target_num] = d
        ret.append(d2)
    return ret

def load_teapot_batch(batch_size=4, target_num=2):
    vertices, faces = nr.load_obj(os.path.join(data_dir, 'teapot.obj'))
    textures = jt.ones((faces.shape[0], 4, 4, 4, 3))
    vertices, faces, textures = to_minibatch((vertices, faces, textures), batch_size, target_num)
    return vertices.float32(), faces.float32(), textures.float32()