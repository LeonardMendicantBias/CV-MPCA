from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Any

import numpy as np
from PIL import Image

from pysolotools.core.models import BoundingBox3DAnnotation as BBox3DAnno
from pysolotools.core.models import Frame, Capture
from scipy.spatial.transform import Rotation as R

from .base import Object
import wireless_env as nr

@dataclass
class UE(Object):
    required_rate: float | List[float]
    los: np.ndarray | List[np.ndarray]
    ca: np.ndarray | List[np.ndarray]
    
    @staticmethod
    def collate_fn(data: List[UE]) -> Dict[str, UE]:
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
