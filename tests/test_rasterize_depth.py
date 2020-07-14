import unittest
import os

import jittor as jt
import numpy as np
from skimage.io import imread

import neural_renderer as nr
from . import utils

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')

class TestRasterizeDepth(unittest.TestCase):
    def test_forward_case1(self):
        """Whether a silhouette by neural renderer matches that by Blender."""

        # load teapot
        vertices, faces, _ = utils.load_teapot_batch()

        # create renderer
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.image_size = 256
        renderer.anti_aliasing = False

        images = renderer(vertices, faces, mode='depth')
        images = images.numpy()
        image = images[2]
        image = image != image.max()

        # load reference image by blender
        ref = imread(os.path.join(data_dir, 'teapot_blender.png'))
        ref = (ref.min(axis=-1) != 255).astype(np.float32)

        assert(np.allclose(ref, image))

    def test_forward_case2(self):
        # load teapot
        vertices, faces, _ = utils.load_teapot_batch()

        # create renderer
        renderer = nr.Renderer(camera_mode='look_at')
        renderer.image_size = 256
        renderer.anti_aliasing = False

        images = renderer(vertices, faces, mode='depth')
        images = images.numpy()
        image = images[2]
        image[image == image.max()] = image.min()
        image = (image - image.min()) / (image.max() - image.min())

        ref = imread(os.path.join(data_dir, 'test_depth.png')).astype(np.float32) / 255.

        assert(np.allclose(image, ref, atol=1e-2))

    def test_backward_case1(self):
        vertices = [
            [-0.9, -0.9, 2.],
            [-0.8, 0.8, 1.],
            [0.8, 0.8, 0.5]]
        faces = [[0, 1, 2]]

        renderer = nr.Renderer(camera_mode='look_at')
        renderer.image_size = 64
        renderer.anti_aliasing = False
        renderer.perspective = False
        renderer.camera_mode = 'none'

        vertices = jt.array(np.array(vertices, np.float32))
        faces = jt.array(np.array(faces, np.int32))
        vertices, faces = utils.to_minibatch((vertices, faces))

        images = renderer(vertices, faces, mode='depth')
        loss = jt.sum((images[0, 15, 20] - 1).sqr())
        loss.backward()
        grad = jt.grad(loss, vertices)
        grad2 = jt.zeros(grad.shape)

        for i in range(3):
            for j in range(3):
                eps = 1e-3
                vertices2 = vertices.clone()
                vertices2[i, j] += eps
                images = renderer.render_depth(vertices2, faces)
                loss2 = jt.sum((images[0, 15, 20] - 1).sqr())
                grad2[i, j] = ((loss2 - loss) / eps)

        assert(np.allclose(grad.numpy(), grad2.numpy(), atol=1e-3))


if __name__ == '__main__':
    unittest.main()
