"""
Example 1. Drawing a teapot from multiple viewpoints.
"""
import os
import argparse

import torch
import numpy as np
import tqdm
import imageio

import neural_renderer as nr

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
    texture_size = 2

    # load .obj
    vertices, faces = nr.load_obj(args.filename_input)
    vertices = vertices[None, :, :]  # [num_vertices, XYZ] -> [batch_size=1, num_vertices, XYZ]
    faces = faces[None, :, :]  # [num_faces, 3] -> [batch_size=1, num_faces, 3]

    # create texture [batch_size=1, num_faces, texture_size, texture_size, texture_size, RGB]
    textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()

    # to gpu

    # create renderer
    renderer = nr.Renderer()

    with imageio.get_writer(args.filename_output, mode='I') as writer:
        loop = tqdm.tqdm(range(0, 360, 4))
        for num, azimuth in enumerate(loop):
            loop.set_description('Drawing')

            renderer.camera.position = torch.tensor([0, 0, camera_distance]).float().reshape(1, 1, 3)
            angle = azimuth / 360 * 2 * np.pi
            axis = angle * torch.tensor([0, 1, 0]).float().view(1, 1, 3)

            renderer.camera.rotation = nr.rotation_from_axis(axis)

            images = renderer(vertices, faces, textures)  # [batch_size, RGB, image_size, image_size]
            image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
            writer.append_data((255*image).astype(np.uint8))


if __name__ == '__main__':
    main()
