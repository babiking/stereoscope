import numpy as np
from plyfile import PlyData, PlyElement


def write_vertices_to_ply(vertices, ply_file):
    vertices = np.array(vertices,
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    element = PlyElement.describe(data=vertices, name='vertex', comments=['vertices'])

    PlyData(elements=[element]).write(ply_file)