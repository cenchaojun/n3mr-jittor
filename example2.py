"""
Example 2. Optimizing vertices.
"""
from __future__ import division
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
    def __init__(self, filename_obj, filename_ref):
        super(Model, self).__init__()

        # load .obj
        vertices, faces = nr.load_obj(filename_obj)
        self.vertices = vertices.unsqueeze(0).float32()
        self.faces = faces.unsqueeze(0).float32().stop_grad()

        # create textures
        texture_size = 2
        self.textures = jt.ones((1, self.faces.shape[1], texture_size, texture_size, texture_size, 3)).float32().stop_grad()

        # load reference image
        self.image_ref = jt.array(imread(filename_ref).astype(np.float32).mean(-1) / 255.).unsqueeze(0).stop_grad()

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at')
        self.renderer = renderer

    def execute(self):
        self.renderer.eye = nr.get_points_from_angles(2.732, 0, 90)
        image = self.renderer(self.vertices, self.faces, mode='silhouettes')
        loss = jt.sum((image - self.image_ref.unsqueeze(0)).sqr())
        print("[*] loss: ", loss)
        return loss

def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imageio.imread(filename))
            os.remove(filename)
    writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'example2_ref.png'))
    parser.add_argument(
        '-oo', '--filename_output_optimization', type=str, default=os.path.join(data_dir, 'example2_optimization.gif'))
    parser.add_argument(
        '-or', '--filename_output_result', type=str, default=os.path.join(data_dir, 'example2_result.gif'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    model = Model(args.filename_obj, args.filename_ref)

    optimizer = nn.Adam(model.parameters(), lr=1e-3)
    # optimizer = nn.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    # optimizer.setup(model)
    # loop = tqdm.tqdm(range(300))
    # for i in loop:
    for i in range(300):
        # loop.set_description('Optimizing')
        # optimizer.target.cleargrads()
        print("=" * 30)
        print("=" * 30)
        loss = model()
        for p in model.parameters():
            print(p.is_stop_grad() ,p.min(), p.max())

        optimizer.step(loss)
        images = model.renderer(model.vertices, model.faces, mode='silhouettes')
        image = images.numpy()[0,0]
        imsave('/tmp/_tmp_%04d.png' % i, image)
    make_gif(args.filename_output_optimization)

    # draw object
    loop = tqdm.tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        model.renderer.eye = nr.get_points_from_angles(2.732, 0, azimuth)
        images, _, _ = model.renderer(model.vertices, model.faces, model.textures)
        image = images.numpy()[0].transpose((1, 2, 0))
        imsave('/tmp/_tmp_%04d.png' % num, image)
    make_gif(args.filename_output_result)


if __name__ == '__main__':
    main()
