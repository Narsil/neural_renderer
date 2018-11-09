from __future__ import division
import math

import torch
import torch.nn as nn
import numpy

import neural_renderer as nr


class Renderer(nn.Module):
    def __init__(self, anti_aliasing=True, background_color=[0,0,0],
                 fill_back=True, camera=None,
                 light_intensity_ambient=0.5, light_intensity_directional=0.5,
                 light_color_ambient=[1,1,1], light_color_directional=[1,1,1],
                 light_direction=[0,1,0]):
        super(Renderer, self).__init__()

        if camera is None:
            self.camera = nr.Camera()
        elif isinstance(camera, nr.Camera):
            self.camera = camera
        else:
            t = type(camera)
            raise Exception(f"camera needs to be of type Camera, got {t}")


        self.anti_aliasing = anti_aliasing
        self.background_color = background_color
        self.fill_back = fill_back

        # light
        self.light_intensity_ambient = light_intensity_ambient
        self.light_intensity_directional = light_intensity_directional
        self.light_color_ambient = light_color_ambient
        self.light_color_directional = light_color_directional
        self.light_direction = light_direction

        # rasterization
        self.rasterizer_eps = 1e-3

    def forward(self, vertices, faces, textures=None, mode=None):
        '''
        Implementation of forward rendering method
        The old API is preserved for back-compatibility with the Chainer implementation
        '''
        _textures = textures
        if mode not in [None, 'silhouettes', 'depth']:
            raise ValueError("mode should be one of None, 'silhouettes' or 'depth'")
        
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
            if _textures is not None:
                _textures = torch.cat((_textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)
        
        if textures is not None:
            # lighting
            faces_lighting = nr.vertices_to_faces(vertices, faces)
            _textures = nr.lighting(
                faces_lighting,
                _textures,
                self.light_intensity_ambient,
                self.light_intensity_directional,
                self.light_color_ambient,
                self.light_color_directional,
                self.light_direction)

        # projection
        vertices = nr.projection(vertices, self.camera)
        if self.camera.perspective:
            vertices = nr.perspective(vertices, angle=self.camera.viewing_angle)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)

        if mode is None:
            images = nr.rasterize(
                faces, _textures, self.camera.image_size, self.anti_aliasing, self.camera.near, self.camera.far, self.rasterizer_eps,
                self.background_color)
        elif mode == 'silhouettes':
            images = nr.rasterize_silhouettes(faces, self.camera.image_size, self.anti_aliasing)
        elif mode == 'depth':
            images = nr.rasterize_depth(faces, self.camera.image_size, self.anti_aliasing)

        return images
