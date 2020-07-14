import jittor as jt


def vertices_to_faces(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3)
    :return: [batch size, number of faces, 3, 3]
    """
    assert (len(vertices.shape) == 3)
    assert (len(faces.shape) == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    faces = faces + (jt.index(bs)[0] * nv).unsqueeze(-1).unsqueeze(-1)
    vertices = vertices.reshape((bs * nv, 3))
    return vertices[faces]