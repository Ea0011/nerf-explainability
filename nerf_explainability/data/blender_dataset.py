from load_blender import load_blender_data
import numpy as np
from nerf_explainability.config.nerf_config import load_config, Config
import torch


class BlenderDataset:
    """
    A dataset that loads samples for NeRF

    images - the images taken at different angles (N, H, W, 3)
    poses - the poses of the camera used to take each image (N, 4, 4)
    hwf - the height, width and focal length of the camera (H, W, F)
    i_split - train/val/test split indices
    num_poses - number of camera poses to render for
    """

    def __init__(self, cfg: Config, num_poses: int = 1, offset: int = 0) -> None:
        images, poses, self.render_poses, hwf, i_split = load_blender_data(
            cfg.datadir, cfg.half_res, cfg.testskip
        )

        i_train, i_val, i_test = i_split
        near = 2.0
        far = 6.0

        if cfg.white_bkgd:
            self.images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        else:
            self.images = images[..., :3]

        self.H, self.W, self.focal = hwf
        self.H, self.W = int(self.H), int(self.W)
        self.hwf = [self.H, self.W, self.focal]

        self.K = torch.tensor(
            [[self.focal, 0, 0.5 * self.W], [0, self.focal, 0.5 * self.H], [0, 0, 1]]
        )

        if cfg.render_test:
            self.render_poses = torch.tensor(poses[i_test])[
                offset : (offset + num_poses)
            ]
            self.images = self.images[i_test][offset : (offset + num_poses)]

        self.bds_dict = {
            "near": near,
            "far": far,
        }
