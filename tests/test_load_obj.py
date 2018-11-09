import unittest
import os
import math

import numpy as np
from imageio import imsave
import torch
import torch.nn.functional as F

import neural_renderer as nr


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


def position_rotation_from_angles(distance, elevation, azimuth, at=[0, 0, 0], up=[0, 1, 0]):
    """
    Gives a position and a rotation from elevation and azimuth parameters
    """
    elevation = math.radians(elevation)
    azimuth = math.radians(azimuth)
    position = [distance * math.cos(elevation) * math.sin(azimuth),
                distance * math.sin(elevation),
                -distance * math.cos(elevation) * math.cos(azimuth)]

    at = torch.tensor(at).float()[None, :]
    position = torch.tensor(position).float()[None, :]
    up = torch.tensor(up).float()[None, :]

    z_axis = F.normalize(at - position, eps=1e-5)
    x_axis = F.normalize(torch.cross(up, z_axis), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis), eps=1e-5)

    r = torch.cat((x_axis[:, :], y_axis[:, :], z_axis[:, :]), dim=0)

    rotation = r.transpose(1, 0)
    return position, rotation


class TestCore(unittest.TestCase):
    def test_tetrahedron(self):
        vertices_ref = np.array(
            [
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [0., 0., 0.]],
            'float32')
        faces_ref = np.array(
            [
                [1, 3, 2],
                [3, 1, 0],
                [2, 0, 1],
                [0, 2, 3]],
            'int32')

        obj_file = os.path.join(data_dir, 'tetrahedron.obj')
        vertices, faces = nr.load_obj(obj_file, False)
        assert (np.allclose(vertices_ref, vertices))
        assert (np.allclose(faces_ref, faces))
        vertices, faces = nr.load_obj(obj_file, True)
        assert (np.allclose(vertices_ref * 2 - 1.0, vertices))
        assert (np.allclose(faces_ref, faces))

    def test_teapot(self):
        obj_file = os.path.join(data_dir, 'teapot.obj')
        vertices, faces = nr.load_obj(obj_file)
        assert (faces.shape[0] == 2464)
        assert (vertices.shape[0] == 1292)

    def test_texture(self):
        position, rotation = position_rotation_from_angles(2, 15, 30)
        camera = nr.Camera(position=position, rotation=rotation)
        renderer = nr.Renderer(camera=camera)

        vertices, faces, textures = nr.load_obj(
            os.path.join(data_dir, '1cde62b063e14777c9152a706245d48/model.obj'), load_texture=True)

        images = renderer(vertices[None, :, :], faces[None, :, :], textures[None, :, :, :, :, :]).permute(0,2,3,1).detach().cpu().numpy()
        image = (images[0] * 255).astype(np.uint8)
        imsave(os.path.join(data_dir, 'car.png'), image)

        vertices, faces, textures = nr.load_obj(
            os.path.join(data_dir, '4e49873292196f02574b5684eaec43e9/model.obj'), load_texture=True, texture_size=16)
        position, rotation = position_rotation_from_angles(2, 15, -90)
        renderer.camera.position = position
        renderer.camera.rotation = rotation
        images = renderer(vertices[None, :, :], faces[None, :, :], textures[None, :, :, :, :, :]).permute(0,2,3,1).detach().cpu().numpy()
        image = (images[0] * 255).astype(np.uint8)
        imsave(os.path.join(data_dir, 'display.png'), image)


if __name__ == '__main__':
    unittest.main()
