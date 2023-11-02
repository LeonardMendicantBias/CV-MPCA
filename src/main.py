# %%
from dataclasses import dataclass, field

from typing import Dict, List, Tuple, Optional, Callable, Any

import math
import numpy as np
from PIL import Image

from pysolotools.consumers import Solo
from pysolotools.converters.solo2coco import SOLO2COCOConverter
from pysolotools.core.models import KeypointAnnotationDefinition, RGBCameraCapture
from pysolotools.core.models import BoundingBox2DLabel, BoundingBox2DAnnotation
from pysolotools.core.models import BoundingBox3DLabel, BoundingBox3DAnnotation
from pysolotools.core.models import Frame, Capture
from scipy.spatial.transform import Rotation as R

from torchvision.datasets import ImageFolder
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
# from torch.utils.data import ConcatDataset, DataLoader
from collections import OrderedDict
from torch.utils.data import DataLoader

import torch
from torch import nn, Tensor
from torch.nn import functional as F

import lightning.pytorch as pl

import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

import reader
import network
from my_trainer import HungarianMatcher, SetCriterion

import os
# %matplotlib ipympl

# %%

if __name__ == "__main__":
    train_folder = "/data/solo_22"
    print('here')
    print(os.path.exists(train_folder))
    # train_loader = reader.UnityDataset.from_unity_to_loader(root=train_folder)
    # trainer = pl.Trainer(max_epochs=200, precision=16)
    # net = network.CVMPCA(
    #     captures=train_loader.dataset.captures,
    #     space_size=[[-5, 5], [0, 3], [-5, 5]],
    #     voxel_size=[1., 1., 1.],
    #     num_neck_layers=1,
    #     num_ue=5, num_category=1,
    # )

    # trainer.fit(net, train_dataloaders=train_loader)
