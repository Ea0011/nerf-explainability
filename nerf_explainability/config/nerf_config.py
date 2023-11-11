import os
from dataclasses import dataclass
import configparser
from typing import get_type_hints
from ast import literal_eval
import re
import torch


@dataclass
class Config:
    expname: str
    basedir: str
    datadir: str
    dataset_type: str
    ckpt: str
    netdepth: int
    netwidth: int
    netdepth_fine: int
    netwidth_fine: int
    input_ch: int
    input_ch_views: int
    output_ch: int
    n_rand: int
    lrate: float
    lrate_decay: int
    chunk: int
    netchunk: int
    no_batching: bool
    no_reload: bool
    ft_path: str
    n_samples: int
    n_importance: int
    perturb: float
    use_viewdirs: bool
    i_embed: int
    multires: int
    multires_views: int
    raw_noise_std: float
    render_only: bool
    render_test: bool
    render_factor: int
    skips: list[int]
    white_bkgd: bool
    half_res: bool
    precrop_iters: int
    precrop_frac: float
    pos_embedder: dict
    dir_embedder: dict
    testskip: int
    output_dir: str
    near: float
    far: float
    ndc: bool
    lindisp: bool


def load_config(config_path: str):
    assert config_path is not None

    config_types = get_type_hints(Config)
    config_dict = {}
    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)

    bool_map = {
        "False": False,
        "True": True,
    }

    for section in config_parser.sections():
        section_config = config_parser[section]
        for k, v in section_config.items():
            cast_fn = config_types[k]
            config_dict[k] = cast_fn(v)

            if re.search("^list", str(cast_fn)) is not None:
                config_dict[k] = literal_eval(v)

            if str(cast_fn) == "<class 'bool'>":
                config_dict[k] = bool_map[v]

    pos_mulres, dir_mulres = config_dict["multires"], \
        config_dict["multires_views"]

    config_dict["pos_embedder"] = get_embedder_config(pos_mulres)
    config_dict["dir_embedder"] = get_embedder_config(dir_mulres)

    config_dict["input_ch"] = config_dict["pos_embedder"]["out_dim"]
    config_dict["input_ch_views"] = config_dict["dir_embedder"]["out_dim"]
    config_dict["output_ch"] = 5 if config_dict["n_importance"] > 0 else 4

    config = Config(**config_dict)

    return config


def get_embedder_config(multires: int):
    embed_config_dict = {
        "include_input": True,
        "input_dims": 3,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    num_periodic_embedding_dims = multires * \
        len(embed_config_dict["periodic_fns"]) * \
        embed_config_dict["input_dims"] + 3

    embed_config_dict["out_dim"] = num_periodic_embedding_dims

    return embed_config_dict


if __name__ == "__main__":
    cfg = load_config("./nerf_explainability/config/lego.ini")
    print(cfg)
