from dataclasses import dataclass

from typing import Dict, List, Tuple, Optional, Callable, Any

import numpy as np
from PIL import Image

from pysolotools.consumers import Solo
from pysolotools.core.models import BoundingBox2DLabel, BoundingBox2DAnnotation
from pysolotools.core.models import BoundingBox3DLabel, BoundingBox3DAnnotation
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
    instanceId: int
    category: int
    position: Tuple[float]
    los: np.ndarray

    def __post_init__(self):
        if not isinstance(self.category, Tensor):
            self.category = torch.tensor(self.category, dtype=torch.float)
        if not isinstance(self.position, Tensor):
            self.position = torch.tensor(self.position, dtype=torch.float)
        if not isinstance(self.los, Tensor):
            self.los = torch.tensor(self.los, dtype=torch.float)
    
    def distance(self, other, alphas=[1, 1, 1]):
        '''
            Measure the distance between two objects for Hungarian Matching
        '''
        # print(other.category, self.category)
        cat_delta = F.binary_cross_entropy_with_logits(other.category, self.category)
        pos_delta = F.mse_loss(other.position, self.position)
        los_delta = F.binary_cross_entropy_with_logits(other.los, self.los)
        return alphas[0]*cat_delta + alphas[1]*pos_delta + alphas[2]*los_delta

    @classmethod
    def batchify(cls, objects: List):
        return cls(
            instanceId=[obj.instanceId for obj in objects],
            category=torch.stack([obj.category for obj in objects]),
            position=torch.stack([obj.position for obj in objects]),
            los=torch.stack([obj.los for obj in objects])
        )


class UnityDataset(VisionDataset):

    def __init__(self, to_matlab=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.samples = []
        solo = Solo(data_path=self.root)

        category_lookup = {}
        category_lookup.update({v: k-1 for k, v in solo.categories().items()})
        
        capture_lookup = {}

        # collecting paths and labels
        for frame_idx, frame in enumerate(solo.frames()):
            print(f'\rLoading: {100*frame_idx/len(solo.frames()):.2f}%', end='')
            paths = {
                capture.id: f'{self.root}/sequence.{frame.sequence}/{capture.filename}'
                for capture in frame.captures
            }

            objects = {}
            # for each camera
            for capture in frame.captures:
                # update camera2idx dictionary
                if capture.id not in capture_lookup:
                    capture_lookup[capture.id] = len(capture_lookup)

                anno_3d = [
                    anno for anno in capture.annotations 
                    if isinstance(anno, BoundingBox3DAnnotation)
                ][0]
                if len(anno_3d.values) == 0: break
                r = R.from_quat(capture.rotation)
                for bbox in anno_3d.values:
                    # add new UE
                    if bbox.instanceId not in objects:
                        objects[bbox.instanceId] = Object(
                            instanceId=bbox.instanceId,
                            category=np.zeros(len(category_lookup)),
                            position=np.array(r.apply(bbox.translation)+capture.position),
                            los=np.zeros(len(frame.captures))
                        )
                    
                    # update UE's information
                    objects[bbox.instanceId].category[category_lookup[bbox.labelName]] = 1
                    # if capture has bbox, then it is in line of sight
                    objects[bbox.instanceId].los[capture_lookup[capture.id]] = 1
            else:
                self.samples.append((paths, list(objects.values())))

            if to_matlab:
                # savemat('./sbss.mat', {"position": np.array([
                #     [8, 2.8, 6.5],
                #     [8, 2.8, -6.5],
                #     [1, 2.8, 6.5],
                #     [1, 2.8, -6.5],
                #     [-5, 2.8, 6.5],
                #     [-5, 2.8, -6.5],
                #     [-10.5, 2.8, 0],
                # ])})
                # savemat('./ues.mat', {"position": np.array(ue_pos), "blockage": np.ones((5, 9))})
                pass
            # if frame_idx == 1: break
        else:
            category_lookup['background'] = len(category_lookup)
            print(f'\rDataset loaded succesfully.')

        self.category_lookup = category_lookup
        self.capture_lookup = capture_lookup
        self.captures = frame.captures

    def _build_transform(self):
        transform = create_transform(
            input_size=[1024, 800],
            is_training=True,
            interpolation='bicubic',
        )
        transform.transforms[0] = transforms.RandomCrop([800, 1024], padding=4)
        return transform
        
    def __getitem__(self, index: int) -> Tuple[Dict[str, Image.Image], Object]:
        path_dict, targets = self.samples[index]
        images = {key: Image.open(path).convert('RGB') for key, path in path_dict.items()}
        if self.transform is not None:
            images = {key: self.transform(image) for key, image in images.items()}
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return path_dict, images, targets
    
    def __len__(self) -> int: return len(self.samples)

    @staticmethod
    def _collate_fn(data: List[Tuple[Dict[str, Image.Image], Object]]):
        image_list, object_list = zip(*data)

        image_tensor = {
            key: torch.stack([images[key] for images in image_list])
            for key in image_list[0].keys()
        }

        return image_tensor, object_list  # obj

    @classmethod
    def from_unity(cls, root):
        return cls(
            root=root,
            transform=Swin_V2_T_Weights.DEFAULT.transforms(),
            target_transform=Object.batchify
        )
    
    @classmethod
    def from_unity_to_loader(cls, root, batch_size=4, num_workers=0):
        return DataLoader(
            cls.from_unity(root=root),
            batch_size=batch_size, num_workers=num_workers, shuffle=False,
            collate_fn=cls._collate_fn, pin_memory=True
        )


class UnityMatDataset(VisionDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.samples = []
        solo = Solo(data_path=self.root)

        # category_lookup = {'background': 0}
        category_lookup = {}
        category_lookup.update({v: k-1 for k, v in solo.categories().items()})
        
        capture_lookup = {}

        for frame_idx, frame in enumerate(solo.frames()):
            paths = {
                capture.id: f'{self.root}/sequence.{frame.sequence}/{capture.id}.mat'
                for capture in frame.captures
            }

            objects = {}
            # for each camera
            for capture in frame.captures:
                # update camera2idx dictionary
                if capture.id not in capture_lookup:
                    capture_lookup[capture.id] = len(capture_lookup)

                anno_3d = [
                    anno for anno in capture.annotations 
                    if isinstance(anno, BoundingBox3DAnnotation)
                ][0]
                if len(anno_3d.values) == 0: break
                r = R.from_quat(capture.rotation)
                for bbox in anno_3d.values:
                    # add new UE
                    if bbox.instanceId not in objects:
                        objects[bbox.instanceId] = Object(
                            instanceId=bbox.instanceId,
                            category=np.zeros(len(category_lookup)),
                            position=np.array(r.apply(bbox.translation)+capture.position),
                            los=np.zeros(len(frame.captures))
                        )
                    
                    # update UE's information
                    objects[bbox.instanceId].category[category_lookup[bbox.labelName]] = 1
                    # if capture has bbox, then it is in line of sight
                    objects[bbox.instanceId].los[capture_lookup[capture.id]] = 1
            else:
                self.samples.append((paths, list(objects.values())))

            print(f'\rLoading: {100*frame_idx/len(solo.frames()):.2f}%', end='')
        else:
            category_lookup['background'] = len(category_lookup)
            print(f'\rDataset loaded succesfully.')

        self.category_lookup = category_lookup
        self.capture_lookup = capture_lookup
        self.captures = frame.captures

    def __getitem__(self, index: int) -> Tuple[Dict[str, Image.Image], Object]:
        path_dict, targets = self.samples[index]
        visual_features = {}  # [key, (M, D, H, W)]  # M: number of cameras
        for key, path in path_dict.items():
            mat_file = loadmat(path)
            visual_features[key] = {
                f'feat{i}': torch.tensor(mat_file[str(i)]).permute(2, 0, 1)
                for i in range(4)  # load the extracted hierachical features
            }
        
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return path_dict, visual_features, targets
    
    def __len__(self) -> int: return len(self.samples)

    @staticmethod
    def _collate_fn(data: List[Tuple[Dict[str, Image.Image], Object]]):
        paths, feature_list, object_list = zip(*data)

        features = {}
        camera_keys = list(feature_list[0].keys())
        for camera_key in camera_keys:
            feat_keys = list(feature_list[0][camera_key].keys())
            features[camera_key] = {
                feat_key: torch.stack([
                    feats[camera_key][feat_key]
                    for feats in feature_list
                ])
                for feat_key in feat_keys
            }

        return features, object_list  # obj

    @classmethod
    def from_unity(cls, root):
        return cls(
            root=root,
            transform=Swin_V2_T_Weights.DEFAULT.transforms(),
            target_transform=Object.batchify
        )
    
    @classmethod
    def from_unity_to_loader(cls, root, batch_size=4, num_workers=0):
        return DataLoader(
            cls.from_unity(root=root),
            batch_size=batch_size, num_workers=num_workers, shuffle=False,
            collate_fn=cls._collate_fn, pin_memory=True
        )

if __name__ == '__main__':
    dataset = UnityDataset(
        root='D:/Unity/dataset/solo_12',
        transform=Swin_V2_T_Weights.DEFAULT.transforms(),
        target_transform=lambda ue: ue.to_target()
    )
    for sample in dataset:
        images, ues = sample
        for key, image in images.items():
            print(key, image.shape)
        for ue in ues:
            print(ue)
        break