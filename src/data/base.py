from __future__ import annotations
import os
from dataclasses import dataclass

from typing import Dict, List, Tuple, Optional, Callable, Any

import numpy as np
from PIL import Image

from pysolotools.consumers import Solo
from pysolotools.core.models import BoundingBox2DLabel, BoundingBox2DAnnotation
from pysolotools.core.models import BoundingBox3DLabel, BoundingBox3DAnnotation as BBox3DAnno
from pysolotools.core.models import Frame, Capture
from scipy.spatial.transform import Rotation as R

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
# from torch.utils.data import ConcatDataset, DataLoader
from collections import OrderedDict
import torchvision.transforms as T
from scipy.io import savemat, loadmat

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import matplotlib.pyplot as plt

# %matplotlib ipympl

from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader



@dataclass
class Object:
    instanceId: str | List[str]
    category: int | List[int]
    position: np.ndarray | List[np.ndarray]

    @classmethod
    def batchify(cls, object_list: List[Object]) -> Object:
        return cls(
            instanceId=torch.stack([obj.instanceId for obj in object_list]),
            category=torch.stack([obj.category for obj in object_list]),
            position=torch.stack([obj.position for obj in object_list]),
        )

    @classmethod
    def from_capture(cls, capture: Capture, category_lookup: Dict[str, int]) -> Object:
        '''
            visible objects in a capture/image
        '''
        anno = [anno for anno in capture.annotations if isinstance(anno, BBox3DAnno)][0]
        if len(anno.values) == 0: return None

        r = R.from_quat(capture.rotation)
        return cls(
            instanceId=[str(bbox.instanceId) for bbox in anno.values],
            category=[category_lookup[bbox.labelName] for bbox in anno.values],
            position=[
                np.array(r.apply(bbox.translation)+capture.position)
                for bbox in anno.values
            ],
        )

@dataclass
class Coordinate:
    x: float
    y: float
    z: float

    def norm(self, other):
        return np.linalg.norm(np.array([self.x, self.y, self.z]) - np.array([other.x, other.y, other.z]))

@dataclass
class Path:
    path: List[Coordinate]

x, y, z = None, None, None

coordinate: (x, y, z)
path: list(coordinate 1, coordinate 2, ..., destination coordinate)
paths: list(path 1, path 2, ..., last path)

cluster = np.zeros(len(paths))
cluster_count = 1
for i, path in enumerate(paths):
    if i == 0: continue

    for j in range(i-1):
        # If the current path paths[i] is similar to the paths before that (paths[j]), put paths[i] in cluster[j]
        # This is done by marking the cluster number in cluster array
        if len(paths[j]) == len(paths[i]) and np.linalg.norm(np.array(paths[i])-np.array(paths[j]))/(len(paths[i])-1) <= 1:
            cluster[i] = cluster[j]
            break
                        
        # If no similar paths, add the paths[i] into a new cluster
        if j == i-1:
            cluster[i] = cluster_count
            cluster_count += 1

ind_paths = []
for cl in range(1, cluster_count):
    cluster_paths = paths[np.where(cluster == cl)]     # Find out the paths in the same cluster cl
    distance = np.zeros(len(cluster_paths))        # Measure the sum of the distance of a path in the cluster cl with other paths in the cluster. Distance is given by the norm of subtraction of the paths.
    for i in range(len(cluster_paths)):
        for j in range(len(cluster_paths)):
            if j != i:
                distance[i] += np.linalg.norm(np.array(cluster_paths[i])-np.array(cluster_paths[j]))

   # The path with the minimum sum distance in a cluster becomes the representative path of that cluster.
    minimum_distance_path = np.argmin(distance)
    ind_paths.append(cluster_paths[minimum_distance_path])

# for each established cluster, collect the corresponding paths
path_ids = cluster[cluster == idx]
cluster_paths = [paths[i] for i in path_ids]

# The representative path is the path with the minimum distance from other paths in the cluster
# A matrix, minimum distance
    
