"""
Example 4. Finding camera parameters.
"""
import os
import argparse
import glob

import jittor as jt
from jittor import nn
import numpy as np
from skimage.io import imread, imsave
import tqdm
import imageio

import neural_renderer as nr
from pdb import set_trace as st
jt.flags.use_cuda = 1

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

class Model(nn.Module):
    def __init__(self, filename_obj, filename_ref=None):
        super(Model, self).__init__()
        # load .obj
        vertices, faces = nr.load_obj(filename_obj)
        self.vertices = vertices.unsqueeze(0).float32().stop_grad()
        self.faces = faces.unsqueeze(0).float32().stop_grad()

        # create textures
        texture_size = 2
        self.textures = jt.ones((1, self.faces.shape[1], texture_size, texture_size, texture_size, 3)).float32().stop_grad()

        # load reference image
        self.image_ref = jt.array((imread(filename_ref).max(-1) != 0).astype(np.float32)).stop_grad()

        # camera parameters
        self.camera_position = jt.array([6,10,-14]).float32()

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.eye = self.camera_position
        self.renderer = renderer

    def execute(self):
        image = self.renderer(self.vertices, self.faces, mode='silhouettes')
        loss = jt.sum((image - self.image_ref.unsqueeze(0)).sqr())
        return loss


def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)
    writer.close()

def make_reference_image(filename_ref, filename_obj):
    model = Model(filename_obj)

    model.renderer.eye = nr.get_points_from_angles(2.732, 30, -15)
    images, _, _ = model.renderer.render(model.vertices, model.faces, jt.tanh(model.textures))
    image = images.numpy()[0]
    imsave(filename_ref, image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'example4_ref.png'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example4_result.gif'))
    parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    if args.make_reference_image:
        make_reference_image(args.filename_ref, args.filename_obj)

    model = Model(args.filename_obj, args.filename_ref)

    # optimizer = chainer.optimizers.Adam(alpha=0.1)
    optimizer = nn.Adam(model.parameters(), lr=0.1)
    loop = tqdm.tqdm(range(1000))
    for i in loop:
        loss = model()
        # for p in model.parameters():
        #     if not p.is_stop_grad():
        #         print(p, jt.grad(loss,p))
        optimizer.step(loss)
        # if i == 20:
        #     break
        images, _, _ = model.renderer(model.vertices, model.faces, jt.tanh(model.textures))
        image = images.numpy()[0].transpose(1,2,0)
        imsave('/tmp/_tmp_%04d.png' % i, image)
        loop.set_description('Optimizing (loss %.4f)' % loss.data)
        if loss.data < 70:
            break
    make_gif(args.filename_output)


if __name__ == '__main__':
    main()
