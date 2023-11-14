from load_llff import load_llff_data
import numpy as np
from nerf_explainability.config.nerf_config import load_config, Config
import torch

class LLFFDataset():
    """
    A dataset that loads samples for NeRF

    images - the images taken at different angles (N, H, W, 3)
    poses - the poses of the camera used to take each image (N, 4, 4)
    hwf - the height, width and focal length of the camera (H, W, F)
    i_split - train/val/test split indices
    num_poses - number of camera poses to render for
    """
    def __init__(self, cfg: Config, num_poses: int = 1, offset: int = 0) -> None:
        images, poses, bds, render_poses, i_test = load_llff_data(basedir=cfg.datadir,
                                                                        factor=cfg.factor,
                                                                        recenter=True,
                                                                        bd_factor=.75,
                                                                        spherify=cfg.spherify)

        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        
        print('Loaded llff', images.shape, render_poses.shape, hwf, cfg.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if cfg.llffhold > 0:
            print('Auto LLFF holdout,', cfg.llffhold)
            i_test = np.arange(images.shape[0])[::cfg.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if not cfg.ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

        self.H, self.W, self.focal = hwf
        self.H, self.W = int(self.H), int(self.W)
        self.hwf = [self.H, self.W, self.focal]
        self.K = torch.tensor([
            [self.focal, 0, 0.5*self.W],
            [0, self.focal, 0.5*self.H],
            [0, 0, 1]
        ])

        if cfg.render_test:
            print(torch.tensor(poses[i_test]).shape)
            self.render_poses = torch.tensor(poses[i_test])[offset:(offset+num_poses)]
            self.images = images[i_test][offset:(offset+num_poses)]

        self.bds_dict = {
            "near": near,
            "far": far,
        }


if __name__ == "__main__":
    # TODO: Create a config for the Fern Object
    cfg = load_config("./nerf_explainability/config/fern.ini")
    # ds = BlenderDataset(cfg, num_poses=1, offset=4)
    ds = LLFFDataset(cfg, num_poses=1, offset=4)

    print(ds.render_poses.shape, ds.H, ds.images.shape)
