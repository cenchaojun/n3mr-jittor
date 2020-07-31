import numpy as np
import jittor as jt
from jittor import nn

def lighting(faces, textures, intensity_ambient=0.5, intensity_directional=0.5,
             color_ambient=(1, 1, 1), color_directional=(1, 1, 1), direction=(0, 1, 0)):

    bs, nf = faces.shape[:2]

    # arguments
    # make sure to convert all inputs to float tensors
    color_ambient = jt.array(color_ambient, "float32")
    color_directional = jt.array(color_directional, "float32")
    direction = jt.array(direction, "float32")

    if len(color_ambient.shape) == 1:
        color_ambient = color_ambient.unsqueeze(0)
    if len(color_directional.shape) == 1:
        color_directional = color_directional.unsqueeze(0)
    if len(direction.shape) == 1:
        direction = direction.unsqueeze(0)

    # create light
    light = jt.zeros((bs, nf, 3), "float32")

    
    # ambient light
    if intensity_ambient != 0:
        light += intensity_ambient * color_ambient.unsqueeze(1)

    # directional light
    if intensity_directional != 0:
        faces = faces.reshape((bs * nf, 3, 3))
        v10 = faces[:, 0] - faces[:, 1]
        v12 = faces[:, 2] - faces[:, 1]

        normals = jt.normalize(jt.cross(v10, v12), eps=1e-5)
        normals = normals.reshape((bs, nf, 3))

        if len(direction.shape) == 2:
            direction = direction.unsqueeze(1)
        cos = nn.relu(jt.sum(normals * direction, dim=2))
        # may have to verify that the next line is correct
        light += intensity_directional * (color_directional.unsqueeze(1) * cos.unsqueeze(2))
    # apply
    light = light.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)
    textures *= light
    return textures
