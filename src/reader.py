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


import wireless_env as nr


@dataclass
class Object:
    instanceId: str | List[str]
    category: int | List[int]
    position: np.ndarray | List[np.ndarray]

    @classmethod
    def batchify(cls, objects: List):
        return cls(
            instanceId=[obj.instanceId for obj in objects],
            category=torch.stack([obj.category for obj in objects]),
            position=torch.stack([obj.position for obj in objects]),
            los=torch.stack([obj.los for obj in objects])
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
class UE(Object):
    required_rate: float | List[float]
    los: np.ndarray | List[np.ndarray]
    ca: np.ndarray | List[np.ndarray]
    
    @staticmethod
    def collate_fn(data: List[MultiViewSample]):
        camera_keys = list(data[0].image_paths.keys())
        return {
            key: key
            for key in camera_keys
        }
    
    @classmethod
    def from_frame(cls,
        min_req: float, max_req: float,
        power: float, N_h: int, N_v: int, freq: int,
        V_max: int, noise: float,
        frame: Frame,
        category_lookup: Dict[str, int], capture_lookup: Dict[str, int]
    ) -> UE:
        object_dict = {
            capture.id: Object.from_capture(capture, category_lookup)
            for capture in frame.captures
        }  # extract objects from each capture/image
        instanceIds = set([idx for obj in object_dict.values() for idx in obj.instanceId if obj is not None])

        # generate an placeholder UE
        ues = cls(
            instanceId=instanceIds,
            category=np.zeros((len(instanceIds), len(category_lookup))),
            position=np.zeros((len(instanceIds), 3)),
            # required_rate=np.random.rand(1)*(max_req-min_req) + min_req,
            required_rate=0,
            los=np.zeros((len(instanceIds), len(frame.captures))),
            ca=np.zeros((len(instanceIds), len(frame.captures))),  # ???
        )  # K of UEs
        
        for i, idx in enumerate(instanceIds):
            for camera_idx, objs in object_dict.items():
                idx_ = objs.instanceId.index(idx) if idx in objs.instanceId else None
                if idx_ is None: continue

                ues.category[i] = objs.category[idx_]
                ues.position[i] = objs.position[idx_]
                ues.los[i, capture_lookup[camera_idx]] = 1

        sbs_positions = np.array([capture.position for capture in frame.captures])
        M, K = sbs_positions.shape[0], len(object_dict)

        distance, theta, phi = nr.cart2sph(ues.position.reshape([1, K, 3]) - sbs_positions.reshape([M, 1, 3]))
        perfect_aod = (theta, phi)
        _, beam_gain = nr.AoD_to_beamgain(sbs_positions, ues.position, perfect_aod, power, ues.los.T, N_h, N_v, freq)
        ues.ca = nr.cell_association(ues.required_rate, beam_gain, V_max, noise)
    
        return ues


@dataclass
class MultiViewSample:
    # images: Dict[str, torch.Tensor]
    image_paths: Dict[str, str] | List[Dict[str, str]]

    @classmethod
    def from_frame(cls, root: str, frame: Frame):
        return cls(
            image_paths={
                capture.id: f'{root}/sequence.{frame.sequence}/step0.{capture.id}.png'
                for capture in frame.captures
            }
        )

@dataclass
class MultiViewSampleDS(MultiViewSample):
    images: torch.Tensor
    
    @classmethod
    def collate_fn(cls, data: List[MultiViewSample]):
        keys = list(data[0].image_paths.keys())
        return cls(

        )
    
    @classmethod
    def from_mv_sample(cls, mv_sample: MultiViewSample, transform: T.Compose=None):
        images = {key: Image.open(path).convert('RGB') for key, path in mv_sample.image_paths.items()}
        if transform is not None:
            images = {key: transform(image) for key, image in images.items()}
        return cls(
            image_paths=mv_sample.image_paths,
            images=images
        )


class UnityDataset(VisionDataset):

    def __init__(self, min_req=1, max_req=10, *args, is_parallel, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_parallel = is_parallel
        self.min_req, self.max_req = min_req, max_req

        self.samples = []
        solo = Solo(data_path=self.root)
        for frame in solo.frames(): pass
        self.capture_lookup = {}
        for capture in frame.captures:
            self.capture_lookup[capture.id] = len(self.capture_lookup)
        self.category_lookup = {v: k-1 for k, v in solo.categories().items()}

        # if is_parallel: map_fn = map
        # else: map_fn = map
        # self.samples = [item for item in map(self._read_frame, solo.frames()) if item is not None]
        
        for frame_idx, frame in enumerate(solo.frames()):
            print(f'\rLoading: {100*frame_idx/len(solo.frames()):.2f}%', end='')
            self.samples.append((
                MultiViewSample.from_frame(self.root, frame),
                UE.from_frame(min_req, max_req, frame, self.category_lookup, self.capture_lookup)
            ))
            if frame_idx == 4: break
        else:
            print(f'\rDataset loaded succesfully.')

        self.category_lookup['background'] = len(self.category_lookup)
        
    def _read_frame(self, frame: Frame):
        mv_imgs = MultiViewSample.from_capture(self.root, frame)
        ues = UE.from_frame(self.min_req, self.max_req, frame, self.category_lookup, self.capture_lookup)
        return mv_imgs, ues

    def __getitem__(self, index: int) -> MultiViewSample:
        path_dict, targets = self.samples[index]
        return MultiViewSampleDS.from_mv_sample(path_dict, self.transform), targets

    def __len__(self) -> int: return len(self.samples)

    @staticmethod
    def _collate_fn(data: List[Tuple[Dict[str, Image.Image], Object]]):
        image_list, object_list = zip(*data)

        return MultiViewSampleDS.collate_fn(image_list), Object.collate_fn(object_list)
    
    @classmethod
    def from_unity(cls, root, min_req, max_req, is_parallel=False):
        return cls(
            root=root, min_req=min_req, max_req=max_req, is_parallel=is_parallel,
            transform=Swin_V2_T_Weights.DEFAULT.transforms(),
            target_transform=Object.batchify
        )

    @classmethod
    def from_unity_to_loader(cls, root, min_req=0, max_req=0, batch_size=4, num_workers=0):
        return DataLoader(
            cls.from_unity(root=root, min_req=min_req, max_req=max_req, is_parallel=num_workers!=0),
            batch_size=batch_size, num_workers=num_workers, shuffle=True,
            collate_fn=cls._collate_fn, pin_memory=True
        )

class UnityDataset2(VisionDataset):

    def __init__(self, min_req=1, max_req=10, *args, **kwargs):
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
                    if isinstance(anno, BBox3DAnno)# and anno.labelName == 'phone'
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
                            los=np.zeros(len(frame.captures)),
                            required_rate=np.random.rand(1)*(max_req-min_req) + min_req
                        )
                    
                    # update UE's information
                    objects[bbox.instanceId].category[category_lookup[bbox.labelName]] = 1
                    # if capture has bbox, then it is in line of sight
                    objects[bbox.instanceId].los[capture_lookup[capture.id]] = 1
            else:
                self.samples.append((paths, list(objects.values())))

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
        # if self.target_transform is not None:
        targets = targets

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
        category_lookup = {v: k-1 for k, v in solo.categories().items()}
        
        capture_lookup = {}

        for frame_idx, frame in enumerate(solo.frames()):
            print(f'\rLoading: {100*frame_idx/len(solo.frames()):.2f}%', end='')
            img_paths ={
                # capture.id: f'{self.root}/sequence.{frame.sequence}/{capture.id}.mat'
                capture.id: f'{self.root}/sequence.{frame.sequence}/{capture.id}'
                for capture in frame.captures
            }
            obj_path = f'{self.root}/sequence.{frame.sequence}'
            # objects = loadmat(f'{obj_path}/objects.mat')
            if not os.path.isfile(f'{obj_path}/objects.mat'): continue

            # if frame_idx == 2: break 
            self.samples.append((img_paths, obj_path))

            
        # else:
        category_lookup['background'] = len(category_lookup)
        print(f'\rDataset loaded succesfully.')

        self.category_lookup = category_lookup
        self.capture_lookup = capture_lookup
        self.captures = frame.captures

    def __len__(self) -> int: return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Dict[str, Image.Image], Object]:
        img_paths, obj_path = self.samples[index]
        visual_features = {}  # [key, (M, D, H, W)]  # M: number of cameras
        for key, path in img_paths.items():
            mat_file = loadmat(f'{path}.mat')
            visual_features[key] = {
                f'feat{i}': torch.tensor(mat_file[str(i)]).permute(2, 0, 1)
                for i in range(4)  # load the extracted hierachical features
            }
        
        obj = loadmat(f'{obj_path}/objects.mat')
        objects = Object(
            instanceId=torch.tensor(obj['instanceId'][0], dtype=torch.long),
            category=torch.tensor(obj['category'], dtype=torch.float),
            position=torch.tensor(obj['position'], dtype=torch.float),
            los=torch.tensor(obj['los'], dtype=torch.float),
            required_rate=torch.tensor(obj['required_rate'][:, 0], dtype=torch.float),
            ca=torch.tensor(obj['ca'], dtype=torch.float),
        )

        return visual_features, objects, obj['file_paths']

    @staticmethod
    def _collate_fn(data: List[Tuple[Dict[str, Image.Image], Object]]):
        feature_list, object_list, file_path_list = zip(*data)

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

        return features, object_list, file_path_list

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
            batch_size=batch_size, num_workers=num_workers, shuffle=True,
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