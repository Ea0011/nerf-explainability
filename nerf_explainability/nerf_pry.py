import torch
from run_nerf_helpers import *
from run_nerf import *
from nerf_explainability.nerf_config import load_config, Config
from run_nerf import render_path
from nerf_dataset import BlenderDataset
from hooks.hook_registration_resolver import HookRegistratorResolver


# TODO: write hook registrators to gather, modify inputs, activations and weights of the NeRF model
class NeRFExtractor():
    def __init__(self, cfg: Config) -> None:
        self.device = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")

        self.cfg = cfg
        self.load_models(cfg=cfg)
        self.load_embedder(cfg=cfg)
        self.load_poses_dataset(cfg=cfg)
        self.load_layer_names()

    def load_models(self, cfg):
        self.model = NeRF(
            D=cfg.netdepth,
            W=cfg.netwidth,
            input_ch=cfg.input_ch,
            output_ch=cfg.output_ch,
            skips=cfg.skips,
            input_ch_views=cfg.input_ch_views,
            use_viewdirs=cfg.use_viewdirs)
        self.model_fine = None

        if cfg.n_importance > 0:
            self.model_fine = NeRF(
                D=cfg.netdepth_fine,
                W=cfg.netwidth_fine,
                input_ch=cfg.input_ch,
                output_ch=cfg.output_ch,
                skips=cfg.skips,
                input_ch_views=cfg.input_ch_views,
                use_viewdirs=cfg.use_viewdirs)

        ckpt = torch.load(cfg.ckpt, map_location=str(self.device))
        self.model.load_state_dict(ckpt['network_fn_state_dict'])
        self.model.requires_grad_(False)

        if self.model_fine is not None:
            self.model_fine.load_state_dict(ckpt['network_fine_state_dict'])
            self.model_fine.requires_grad_(False)

        self.model_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=self.pos_embed_fn,
                                                                embeddirs_fn=self.dir_embed_fn,
                                                                netchunk=cfg.netchunk)

    def load_layer_names(self):
        self.layer_names = [name for name, _ in list(self.model.named_modules())]

    def load_embedder(self, cfg: Config):
        self.pos_embedder = Embedder(**cfg.pos_embedder)
        self.dir_embedder = Embedder(**cfg.dir_embedder)

        self.pos_embed_fn = lambda x: self.pos_embedder.embed(x)
        self.dir_embed_fn = lambda x: self.dir_embedder.embed(x)

    def load_poses_dataset(self, cfg: Config):
        # TODO: ideally check for dataset type and load proper data
        self.poses_dataset = BlenderDataset(cfg, num_poses=1, offset=4)

    def get_nerf_query_kwargs(self, cfg: Config):
        render_kwargs = {
            'network_query_fn' : self.model_query_fn,
            'perturb' : cfg.perturb,
            'N_importance' : cfg.n_importance,
            'network_fine' : self.model_fine,
            'N_samples' : cfg.n_samples,
            'network_fn' : self.model,
            'use_viewdirs' : cfg.use_viewdirs,
            'white_bkgd' : cfg.white_bkgd,
            'raw_noise_std' : cfg.raw_noise_std,
            'near': cfg.near,
            'far': cfg.far,
            'ndc': cfg.ndc,
            'lindisp': cfg.lindisp,
        }

        return render_kwargs

    def get_layers(self, layer_specifications=[]): # layer_spec defines which layers to output: [{"type": "fine", "name": "pts_linears.0"}]
        if len(layer_specifications) == 0:
            return []

        layers_to_gather, layers_to_gather_fine = [], []
        for spec in layer_specifications:
            model_type = spec["type"]
            if model_type == "fine":
                if spec["name"] not in self.layer_names:
                    print(f"WARNING: layer {spec['name']} is not available")

                layers_to_gather_fine.append(spec["name"])
            else:
                if spec["name"] not in self.layer_names:
                    print(f"WARNING: layer {spec['name']} is not available")

                layers_to_gather.append(spec["name"])

        layers = self._get_corresponding_layers(self.model, model_type="coarse", layers_to_gather=layers_to_gather)
        layers_fine = self._get_corresponding_layers(self.model_fine, model_type="fine", layers_to_gather=layers_to_gather_fine)

        return layers + layers_fine

    def _get_corresponding_layers(self, model, model_type, layers_to_gather):
        if len(layers_to_gather) == 0:
            return []

        layers = []
        if model is not None:
            for name, layer in model.named_modules():
                if name in layers_to_gather:
                    layers.append({
                        "name": f"{model_type}.{name}",
                        "layer": layer,
                    })

        return layers

    def register_hooks(self, hooks): # registers forward hooks based on criteria. [{"type": "fine", "layer_name": "name", "hook_type", "hook", forward_hook_fn}]
        hooks_coarse, hooks_fine = {}, {}
        for hook in hooks:
            model_type, layer_name, hook_type, fn = hook["type"], hook["layer_name"], hook["hook_type"], hook["hook"]

            if layer_name not in self.layer_names:
                raise Exception(f'Error: layer {layer_name} is not available')

            if model_type == "fine":
                hooks_fine[layer_name] = (hook_type, fn)
            elif model_type == "coarse":
                hooks_coarse[layer_name] = (hook_type, fn)

        hook_handles_coarse = self._register_hooks(self.model, model_type="coarse", hooks=hooks_coarse)
        hook_handles_fine = self._register_hooks(self.model_fine, model_type="fine", hooks=hooks_fine)

        hook_handles_coarse.update(hook_handles_fine)

        return hook_handles_coarse

    def _register_hooks(self, model, model_type, hooks):
        hook_handles = {}
        hook_registrator_resolver = HookRegistratorResolver()

        if model is not None:
            for name, layer in model.named_modules():
                if name in hooks:
                    hook_type, hook_fn = hooks[name]
                    hook_registrator = hook_registrator_resolver.resolve(hook_type)
                    handle = hook_registrator(layer, hook_fn)

                    hook_handles[f"{model_type}.{name}"] = handle

        return hook_handles


    def render(self):
        """
         Renders the NeRF into images given camera poses

         Args:
            render_poses -> the camera poses to be rendered
            hwf -> Hieght, Width and Focal Length
            K -> The intrinsic matrix
            chunk -> Num of rays to process simulataneously
            render_kwargs -> rendering arguments
            gt_images -> the ground truth images
            savedir -> directory to save the output images to
            render_factor -> reduction to the rendering resulotion (H // f)
        """
        rgbs, disp = render_path(
            self.poses_dataset.render_poses,
            self.poses_dataset.hwf,
            self.poses_dataset.K,
            self.cfg.chunk,
            self.get_nerf_query_kwargs(self.cfg),
            gt_imgs=self.poses_dataset.images,
            savedir=self.cfg.output_dir,
            render_factor=self.cfg.render_factor,)

        return rgbs, disp


if __name__ == "__main__":
    cfg = load_config("./nerf_explainability/config/lego.ini")
    nerf_extractor = NeRFExtractor(cfg)

    print(f"Rendering with config: {cfg}")
    nerf_extractor.render()


