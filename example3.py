"""
Example 3. Optimizing textures.
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
jt.flags.use_cuda = 1

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')
np.random.seed(1)

class Model(nn.Module):
    def __init__(self, filename_obj, filename_ref):
        super(Model, self).__init__()
        vertices, faces = nr.load_obj(filename_obj)
        self.vertices = vertices.unsqueeze(0).float32().stop_grad()
        self.faces = faces.unsqueeze(0).float32().stop_grad()
        # create textures
        texture_size = 4
        self.textures = jt.zeros((1, self.faces.shape[1], texture_size, texture_size, texture_size, 3)).float32()

        # load reference image
        self.image_ref = jt.array(imread(filename_ref).astype('float32') / 255.).permute(2,0,1).unsqueeze(0).stop_grad()

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.perspective = False
        renderer.light_intensity_directional = 0.0
        renderer.light_intensity_ambient = 1.0
        self.renderer = renderer


    def execute(self):
        num = np.random.uniform(0, 360)
        self.renderer.eye = nr.get_points_from_angles(2.732, 0, num)
        image, _, _ = self.renderer(self.vertices, self.faces, jt.tanh(self.textures))
        loss = jt.sum((image - self.image_ref).sqr())
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
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'example3_ref.png'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example3_result.gif'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    model = Model(args.filename_obj, args.filename_ref)

    optimizer = nn.Adam(model.parameters(), lr=0.1, betas=(0.5,0.999))
    loop = tqdm.tqdm(range(300))
    for num in loop:
        loop.set_description('Optimizing')
        loss = model()
        optimizer.step(loss)

    # draw object
    loop = tqdm.tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        model.renderer.eye = nr.get_points_from_angles(2.732, 0, azimuth)
        images, _, _ = model.renderer(model.vertices, model.faces, jt.tanh(model.textures))
        image = images.numpy()[0].transpose((1, 2, 0))
        imsave('/tmp/_tmp_%04d.png' % num, image)
    make_gif(args.filename_output)


if __name__ == '__main__':
    main()
