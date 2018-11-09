from __future__ import division

import torch


def projection(vertices, camera):
    '''
    camera.rotation, camera.position: batch_size * 3 * 3, batch_size * 1 * 3 camera parameters
    returns the vertices from the camera orientation
    '''
    device = vertices.device
    batch_size = vertices.size()[0]

    # Assume batch_size is 1 when missing.
    if camera.position is None:
        camera.position = torch.tensor([0, 0, 1]).float().view(1, 1, 3).expand(batch_size, 1, 3)

    if camera.rotation is None:
        camera.rotation = torch.eye(3).view(1, 3, 3).expand(batch_size, 3, 3)

    camera.rotation = camera.rotation.to(device)
    camera.position = camera.position.to(device)

    vertices = torch.matmul(vertices, camera.rotation) + camera.position
    return vertices
