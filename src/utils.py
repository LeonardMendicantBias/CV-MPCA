from typing import List
from dataclasses import dataclass, field
import numpy as np
from pysolotools.core.models import BoundingBox2DLabel, BoundingBox2DAnnotation
from pysolotools.core.models import BoundingBox3DLabel, BoundingBox3DAnnotation
from pysolotools.core.models import Frame, Capture
from scipy.spatial.transform import Rotation as R

import torch
from torch import nn


# @dataclass
# class Camera:
#     capture: Capture
#     resolution: np.ndarray
#     position: np.ndarray=field(init=False)
#     intrinsic: np.ndarray=field(init=False)
#     rotation: R=field(init=False)

#     def __post_init__(self):
#         self.resolution = self.resolution[..., np.newaxis]
#         # self.resolution = np.array(self.capture.dimension)[..., np.newaxis]
#         intrinsic = np.array([self.capture.matrix]).reshape((3, 3))
#         intrinsic /= intrinsic[-1,-1]
#         intrinsic *= np.array([-self.resolution[0, 0]/2, self.resolution[1, 0]/2, 1])
#         intrinsic[0, -1] = self.resolution[0]/2
#         intrinsic[1, -1] = self.resolution[1]/2
#         self.intrinsic = intrinsic
#         self.rotation = R.from_quat(self.capture.rotation)
#         self.position = np.array(self.capture.position)[..., np.newaxis]
    
#     def pos_to_pixel(self, position):
#         camera_coordinates = self.rotation.inv().apply((position - self.position).T).T
#         pixel_coordinates = self.intrinsic @ camera_coordinates
#         pixels = pixel_coordinates / pixel_coordinates[-1]
#         return pixels[:2].astype(np.int_)


# def voxelization(
#         space: List[List[float]],  # X, Y, Z
#         voxel: List[float],  # X, Y, Z
#         resolution: List[int],
#         cameras: List[Camera]
#     ):
#         X = np.arange(space[0][0], space[0][1], voxel[0])
#         Y = np.arange(space[1][0], space[1][1], voxel[1])
#         Z = np.arange(space[2][0], space[2][1], voxel[2])
#         voxels = []
#         for x in X:
#             for y in Y:
#                 for z in Z:
#                     voxels.append(np.array([x, y, z]))
#         voxels = np.array(voxels)

#         pos2pix = {}
#         for camera in cameras:
#             pos2pix[camera.capture.id] = camera.pos_to_pixel(voxels.T)


class Camera(nn.Module):
    '''
        Pytorch-based Camera to utilize GPU acceleration
    '''

    def __init__(self, rotation, translation, intrinsic):
        super().__init__()
        # self.rotation = torch.tensor(rotation)
        # self.translation = torch.tensor(translation)
        # self.intrinsic = torch.tensor(intrinsic)
        
        self.register_buffer('rotation', torch.tensor(rotation))
        self.register_buffer('translation', torch.tensor(translation))
        self.register_buffer('intrinsic', torch.tensor(intrinsic))

    def _pos2pix(self, positions, resolution):
        camera_coordinates = (positions - self.translation.unsqueeze(0)) @ self.rotation
        _intrinsic = self.intrinsic * torch.diag(torch.tensor([-.5, .5, 1], device=positions.device)*resolution)
        pixel_coordinates = camera_coordinates @ _intrinsic
        pixels = pixel_coordinates / (pixel_coordinates[:, -1] + 1e-5 ).unsqueeze(-1)  # keep dimension
        pixels += torch.tensor([.5, .5, 0], device=positions.device)*resolution
        return pixels[:, :2].float()#.int()

    def forward(self, positions, resolution):
        if not isinstance(resolution, torch.Tensor):
            resolution = torch.tensor(resolution, device=positions.device)
        return self._pos2pix(positions, resolution)

    @classmethod
    def from_unity(cls, capture: Capture):
        intrinsic = np.array([capture.matrix]).reshape((3, 3))
        intrinsic /= intrinsic[-1,-1]
        
        rotation = R.from_quat(capture.rotation)
        position = np.array(capture.position)

        return cls(
            rotation=rotation.as_matrix(),
            translation=position,
            intrinsic=intrinsic,
        )
