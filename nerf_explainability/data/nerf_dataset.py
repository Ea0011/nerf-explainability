from load_blender import load_blender_data
from load_llff import load_llff_data
import numpy as np
from nerf_explainability.config.nerf_config import load_config, Config
from .blender_dataset import BlenderDataset
from .llff_dataset import LLFFDataset
import torch
import types


class NeRFDataset:
    @staticmethod
    def from_dataset_type(name="blender"):
        return {
            "blender": BlenderDataset,
            "llff": LLFFDataset,
        }[name]


if __name__ == "__main__":
    cfg = load_config("./nerf_explainability/config/fern.ini")
    ds = NeRFDataset.from_dataset_type(cfg.dataset_type)(cfg, num_poses=1, offset=4)

    print(ds.render_poses.shape, ds.H, ds.images.shape)
