# %%
from typing import List, Tuple
from dataclasses import dataclass, field
import numpy as np
from pysolotools.core.models import BoundingBox2DLabel, BoundingBox2DAnnotation
from pysolotools.core.models import BoundingBox3DLabel, BoundingBox3DAnnotation
from pysolotools.core.models import Frame, Capture
from scipy.spatial.transform import Rotation as R

import torch
from torch import nn, Tensor


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


def _create_voxel(space, voxel_size):
    X = torch.arange(space[0][0], space[0][1], voxel_size[0])
    Y = torch.arange(space[1][0], space[1][1], voxel_size[1])
    Z = torch.arange(space[2][0], space[2][1], voxel_size[2])
    
    grid_x, grid_y, grid_z = torch.meshgrid(X, Y, Z)
    return torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)


@dataclass
class Space:
    '''
        Space to be voxelized (measured in meters)
    '''
    space: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    size: Tuple[float, float, float]
    voxel: Tensor=field(init=False)

    def __post_init__(self):
        s = np.array(self.size)
        self.voxels = self._create_voxel(s)

    def _create_voxel(self, voxel_size):
        X = np.arange(self.space[0][0], self.space[0][1], voxel_size[0])
        Y = np.arange(self.space[1][0], self.space[1][1], voxel_size[1])
        Z = np.arange(self.space[2][0], self.space[2][1], voxel_size[2])
        
        grid_x, grid_y, grid_z = torch.meshgrid(X, Y, Z)
        return torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)


class Camera(nn.Module):
    '''
        Pytorch-based Camera to utilize GPU acceleration
            - X: right, Y: up, Z: forward
            - clockwise rotation. Ref: https://docs.unity3d.com/ScriptReference/UIElements.Rotate.html
    '''

    def __init__(self, rotation, translation, intrinsic):
        super().__init__()
        self.register_buffer('rotation', torch.tensor(rotation, dtype=torch.float))
        self.register_buffer('translation', torch.tensor(translation, dtype=torch.float))
        self.register_buffer('intrinsic', torch.tensor(intrinsic, dtype=torch.float))

    def _pos2pix(self, positions, resolution):
        # camera_coordinates = (positions - self.translation.unsqueeze(0)) @ self.rotation
        # print(positions.shape, self.translation.shape, self.rotation.shape, self.intrinsic.shape)
        camera_coordinates = (positions - self.translation.unsqueeze(0)) @ self.rotation
        _intrinsic = self.intrinsic * torch.diag(torch.tensor([-.5, .5, 1], device=positions.device)*resolution)
        pixel_coordinates = camera_coordinates @ _intrinsic
        # print(pixel_coordinates.shape)
        pixels = pixel_coordinates / (pixel_coordinates[..., -1] + 1e-12 ).unsqueeze(-1)  # keep dimension
        pixels += torch.tensor([.5, .5, 0], device=positions.device)*resolution
        return pixels[..., :2].float()#.int()
    
    def pix2ray(self, pixels, resolution):  # (N, 3) last dimension is 1
        if not isinstance(resolution, torch.Tensor):
            resolution = torch.tensor(resolution, device=pixels.device)

        _intrinsic = self.intrinsic * torch.diag(torch.tensor([-.5, .5, 1], device=pixels.device)*resolution)
        _intrinsic[-1, 0] = resolution[0]/2
        _intrinsic[-1, 1] = resolution[1]/2

        ndc_coor = pixels @ torch.linalg.inv(_intrinsic)#.float()
        rays = ndc_coor @ self.rotation.T
        return rays

    def forward(self, positions, resolution):
        if not isinstance(resolution, torch.Tensor):
            resolution = torch.tensor(resolution, device=positions.device)
        return self._pos2pix(positions, resolution)

    @classmethod
    def from_unity(cls, capture: Capture):
        intrinsic = np.array(capture.matrix).reshape((3, 3))
        intrinsic /= intrinsic[-1,-1]
        
        rotation = R.from_quat(capture.rotation)
        position = np.array(capture.position)

        return cls(
            rotation=rotation.as_matrix(),
            # quaternion=rotation,
            translation=position,
            intrinsic=intrinsic,
        )


if __name__ == '__main__':
    space = [[-11, 11], [0, 3], [-7, 7]]
    voxel = [0.1, 0.1, 0.1]

    X = torch.arange(space[0][0], space[0][1], 3*3*3*voxel[0])
    Y = torch.arange(space[1][0], space[1][1], 3*3*3*voxel[1])
    Z = torch.arange(space[2][0], space[2][1], 3*3*3*voxel[2])

    grid_x, grid_y, grid_z = torch.meshgrid(X, Y, Z)
    grid = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
    print(grid_x.shape, grid_y.shape, grid_z.shape)
    print(grid.shape)

# %%


a, b, c = torch.meshgrid(torch.arange(3), torch.arange(3), torch.arange(3))
grid = torch.stack([a.flatten(), b.flatten(), c.flatten()], dim=1)
