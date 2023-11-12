import torch
from run_nerf_helpers import *
from nerf_explainability.config.nerf_config import load_config, Config
from nerf_explainability.data.nerf_dataset import BlenderDataset
from nerf_explainability.hooks.hook_registration_resolver import HookRegistratorResolver
from nerf_explainability.render.nerf_render import SceneRenderer


class NeRFExtractor:
    """A class that provides access to layers and inputs of NeRF 

    The class loads NeRF models according to configuration
    and provides API to attach torch hooks and get access to
    NeRF layers

    Attributes:
        model: A regula NeRF model
        model_fine: A fine-grained NeRF model for importance sampling (more in paper)
        cfg: Config of the loaded NeRF
        layer_names: Names of NeRF layers
        dir_embedder: Fourier feature embedder of input [theta, phi]
        pos_embedder: Fourier feature embedder of input [x, y, z]
    """
    def __init__(self, cfg: Config) -> None:
        """Initializes the instance based on NeRF config.

        Args:
          cfg: A Config used to instanciate and render from NeRF
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")

        self.cfg = cfg
        self.load_models(cfg=cfg)
        self.load_embedder(cfg=cfg)
        self.load_layer_names()

    def load_models(self, cfg: Config) -> None:
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

    def load_layer_names(self) -> None:
        self.layer_names = [name for name, _ in list(self.model.named_modules())]

    def load_embedder(self, cfg: Config) -> None:
        self.pos_embedder = Embedder(**cfg.pos_embedder)
        self.dir_embedder = Embedder(**cfg.dir_embedder)

    def get_layers(self, layer_specifications: list[dict]) -> list[dict]:
        """Get layers from NeRF models given their names.

        Args:
            layer_specifications: An array of dicts specifying which
            layers to get by name and model type
            An example dict of layer spec is:
                {"type": "fine", "name": "pts_linears.0"}

        Returns:
            An array of all the desired layers in the model
        """
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

    def _get_corresponding_layers(self,
                                  model: NeRF,
                                  model_type: NeRF,
                                  layers_to_gather: list[str]) -> list[dict]:
        """Get layers from NeRF models given their names.

        Args:
            model: A NeRF model loaded by this class
            model_type: Either coarse or fine
            layers_to_gather: A list of names of the layers to get for the model

        Returns:
            An array of all the desired layers in the model
        """
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

    def register_hooks(self, hooks: list[dict]) -> dict:
        """Registers hooks to modules of NeRF.

        Args:
            hooks: An array of dicts specifying hooks and where to attach them
            an example hook config:

            {"type": "fine", "layer_name": "rgb_linear", "hook_type": "pre_forward", "hook": lambda m, i}

        Returns:
            A dict of hook handles attached to each layer of coarse and fine NeRF models
        """
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

    def _register_hooks(self,
                        model: NeRF,
                        model_type: str,
                        hooks: list[dict]) -> dict:
        """Registers hooks to modules of NeRF.

        Args:
            model: A NeRF model loaded in this class
            model_type: Either fine or coarse
            hooks: specifications of hooks and where to attach them

            {"type": "fine", "layer_name": "rgb_linear", "hook_type": "pre_forward", "hook": lambda m, i}

        Returns:
            A dict of hook handles attached to each layer of coarse and fine NeRF models
        """
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


if __name__ == "__main__":
    cfg = load_config("./nerf_explainability/config/lego.ini")
    nerf_extractor = NeRFExtractor(cfg)

    print(f"Rendering with config: {cfg}")

    extracted_layers = nerf_extractor.get_layers([
        {"type": "fine", "name": "pts_linears.0"},
        {"type": "fine", "name": "rgb_linear"},
        {"type": "coarse", "name": "rgb_linear"},
        {"type": "coarse", "name": "rgb_linaer"},
    ])

    print(extracted_layers)

    # hooks = [
    #     {"type": "fine", "layer_name": "rgb_linear", "hook_type": "pre_forward", "hook": lambda m, i: print(f"here we are in {m}")},
    #     {"type": "coarse", "layer_name": "rgb_linear", "hook_type": "forward", "hook": lambda m, i, o: print(f"here we are in {m}")}
    # ]
    # handles = nerf_extractor.register_hooks(hooks)

    ds = BlenderDataset(cfg, num_poses=1, offset=0)

    renderer = SceneRenderer()
    renderer \
        .set_dir_embedder(nerf_extractor.dir_embedder) \
        .set_pos_embedder(nerf_extractor.pos_embedder) \
        .set_model(nerf_extractor.model) \
        .set_model_fine(nerf_extractor.model_fine)
    
    renderer.render(cfg=cfg, render_poses=ds.render_poses, hwf=ds.hwf, K=ds.K, images=ds.images)