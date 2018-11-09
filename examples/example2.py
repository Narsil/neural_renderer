"""
Example 2. Optimizing vertices.
"""
from __future__ import division
import os
import argparse
import glob

import torch
import torch.nn as nn
import numpy as np
from imageio import imread, imsave, get_writer
import tqdm

import neural_renderer as nr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


class Model(nn.Module):
    def __init__(self, filename_obj, filename_ref):
        super(Model, self).__init__()

        # load .obj
        vertices, faces = nr.load_obj(filename_obj)
        self.vertices = nn.Parameter(vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])

        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # load reference image
        image_ref = torch.from_numpy(imread(filename_ref).astype(np.float32).mean(-1) / 255.)[None, ::]
        self.register_buffer('image_ref', image_ref)

        # setup renderer
        renderer = nr.Renderer()
        self.renderer = renderer

    def forward(self):
        angle = np.pi / 2
        axis = angle * torch.tensor([0, 1, 0]).float().view(1, 1, 3)
        self.renderer.camera.position = torch.tensor([0, 0, 2.732]).float().reshape(1, 1, 3)
        self.renderer.camera.rotation = nr.rotation_from_axis(axis)

        images = self.renderer(self.vertices, self.faces, mode='silhouettes')
        loss = torch.sum((images - self.image_ref[None, :, :])**2)
        return images, loss


def make_gif(filename):
    with get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)


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
    model.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    # optimizer.setup(model)
    loop = tqdm.tqdm(range(300))
    for i in loop:
        loop.set_description('Optimizing')
        # optimizer.target.cleargrads()
        optimizer.zero_grad()
        images, loss = model()
        loss.backward()
        optimizer.step()
        image = images.detach().cpu().numpy()[0]
        image = (image * 255).astype(np.uint8)
        imsave('/tmp/_tmp_%04d.png' % i, image)
    make_gif(args.filename_output_optimization)

    # draw object
    loop = tqdm.tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')

        angle = azimuth / 360 * 2 * np.pi
        axis = angle * torch.tensor([0, 1, 0]).float().view(1, 1, 3)
        model.renderer.camera.rotation = nr.rotation_from_axis(axis)

        images = model.renderer(model.vertices, model.faces, model.textures)
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        image = (image * 255).astype(np.uint8)
        imsave('/tmp/_tmp_%04d.png' % num, image)
    make_gif(args.filename_output_result)


if __name__ == '__main__':
    main()
