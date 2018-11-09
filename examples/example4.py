"""
Example 4. Finding camera parameters.
"""
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
    def __init__(self, filename_obj, filename_ref=None):
        super(Model, self).__init__()
        # load .obj
        vertices, faces = nr.load_obj(filename_obj)
        self.register_buffer('vertices', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])

        # create textures
        texture_size = 2
        textures = torch.ones(1, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        self.register_buffer('textures', textures)

        # load reference image
        image_ref = torch.from_numpy((imread(filename_ref).max(-1) != 0).astype(np.float32))
        self.register_buffer('image_ref', image_ref)

        # setup renderer
        renderer = nr.Renderer()

        # camera parameters
        self.camera_position = nn.Parameter(torch.from_numpy(np.array([0.5, 0.5 ,14], dtype=np.float32)))
        renderer.camera.position = self.camera_position

        angle = 0
        axis = angle * torch.tensor([0, 1, 0]).float().view(1, 1, 3)
        renderer.camera.rotation = nr.rotation_from_axis(axis)

        self.renderer = renderer

    def forward(self):
        image = self.renderer(self.vertices, self.faces, mode='silhouettes')
        loss = torch.sum((image - self.image_ref[None, :, :]) ** 2)
        return loss


def make_gif(filename):
    with get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imread(filename))
            os.remove(filename)


def make_reference_image(filename_ref, filename_obj):
    vertices, faces = nr.load_obj(filename_obj)
    vertices = vertices[None, :, :]
    faces = faces[None, :, :]
    texture_size = 2
    textures = torch.ones(1, faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32).cuda()
    renderer = nr.Renderer()

    renderer.camera.position = torch.tensor([0, 0, 2.732]).float()
    angle = 0
    axis = angle * torch.tensor([0, 1, 0]).float().view(1, 1, 3)
    renderer.camera.rotation = nr.rotation_from_axis(axis)

    images = renderer(vertices, faces, torch.tanh(textures))
    image = images.detach().cpu().numpy()[0].transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    imsave(filename_ref, image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'example4_ref.png'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example4_result.gif'))
    parser.add_argument('-mr', '--make_reference_image', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    if args.make_reference_image:
        make_reference_image(args.filename_ref, args.filename_obj)

    model = Model(args.filename_obj, args.filename_ref)
    model.cuda()

    # optimizer = chainer.optimizers.Adam(alpha=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loop = tqdm.tqdm(range(1000))
    for i in loop:
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

        # Optimization happens in silhouette space, here we display texture.
        images = model.renderer(model.vertices, model.faces, torch.tanh(model.textures))
        image = images.detach().cpu().numpy()[0].transpose(1,2,0)
        image = (image * 255).astype(np.uint8)
        imsave('/tmp/_tmp_%04d.png' % i, image)
        loop.set_description('Optimizing (loss %.4f)' % loss.data)
        if loss.item() < 100:
            break
    make_gif(args.filename_output)


if __name__ == '__main__':
    main()
