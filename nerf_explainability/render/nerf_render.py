from nerf_explainability.config.nerf_config import Config
from run_nerf import render_path
from run_nerf_helpers import *
from run_nerf import *
from typing import Callable


class SceneRenderer:
    def set_model_fine(self, model_fine: NeRF):
        self.model_fine = model_fine
        return self

    def set_model(self, model: NeRF):
        self.model = model
        return self

    def set_pos_embedder(self, pos_embedder: Embedder):
        self.pos_embedder = pos_embedder
        self.pos_embed_fn = lambda x: self.pos_embedder.embed(x)

        return self

    def set_dir_embedder(self, dir_embedder: Embedder):
        self.dir_embedder = dir_embedder
        self.dir_embed_fn = lambda x: self.dir_embedder.embed(x)

        return self

    def get_nerf_query_kwargs(self, model_query_fn, cfg: Config) -> dict:
        render_kwargs = {
            "network_query_fn": model_query_fn,
            "perturb": cfg.perturb,
            "N_importance": cfg.n_importance,
            "network_fine": self.model_fine,
            "N_samples": cfg.n_samples,
            "network_fn": self.model,
            "use_viewdirs": cfg.use_viewdirs,
            "white_bkgd": cfg.white_bkgd,
            "raw_noise_std": cfg.raw_noise_std,
            "near": cfg.near,
            "far": cfg.far,
            "ndc": cfg.ndc,
            "lindisp": cfg.lindisp,
        }

        return render_kwargs

    def get_nerf_query_fn(self, cfg: Config) -> Callable:
        return lambda inputs, viewdirs, network_fn: run_network(
            inputs,
            viewdirs,
            network_fn,
            embed_fn=self.pos_embed_fn,
            embeddirs_fn=self.dir_embed_fn,
            netchunk=cfg.netchunk,
        )

    def render(
        self,
        cfg: Config,
        render_poses: torch.Tensor,
        hwf: list[float],
        K: torch.Tensor,
        images: torch.Tensor,
    ) -> tuple:
        """
        Renders the NeRF into images given camera poses

        Inputs to rendering functions:
           render_poses: the camera poses to be rendered
           hwf: Hieght, Width and Focal Length
           K: The intrinsic matrix
           chunk: Num of rays to process simulataneously
           render_kwargs: rendering arguments
           gt_images: the ground truth images
           savedir: directory to save the output images to
           render_factor: reduction to the rendering resulotion (H // f)
        """
        nerf_query_fn = self.get_nerf_query_fn(cfg)
        render_kwargs = self.get_nerf_query_kwargs(nerf_query_fn, cfg)

        rgbs, disp = render_path(
            render_poses,
            hwf,
            K,
            cfg.chunk,
            render_kwargs,
            gt_imgs=images,
            savedir=cfg.output_dir,
            render_factor=cfg.render_factor,
        )

        return rgbs, disp
