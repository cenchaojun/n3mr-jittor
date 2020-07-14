"""
Example 1. Drawing a teapot from multiple viewpoints.
"""
import os
import argparse

import numpy as np
import tqdm
import imageio

import jittor as jt
jt.flags.use_cuda = 1

import neural_renderer as nr
from pdb import set_trace as st
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--filename_input', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-o', '--filename_output', type=str, default=os.path.join(data_dir, 'example1.gif'))
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    # other settings
    camera_distance = 2.732
    elevation = 30
    texture_size = 2

    # load .obj
    vertices, faces = nr.load_obj(args.filename_input)
    
    vertices = vertices.unsqueeze(0)  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces.unsqueeze(0)  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = jt.ones((1, faces.shape[1], texture_size, texture_size, texture_size, 3), dtype="float32")

    # to gpu

    # create renderer
    renderer = nr.Renderer(camera_mode='look_at')

    # draw object
    loop = tqdm.tqdm(range(0, 360, 4))
    writer = imageio.get_writer(args.filename_output, mode='I')
    import cv2
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
        images, _, _ = renderer(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
        image = images.numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
        writer.append_data((255*image).astype(np.uint8))
        # cv2.imwrite(f"{num}.png", (255*image).astype(np.uint8))
    writer.close()

if __name__ == '__main__':
    main()