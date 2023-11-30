import torch
from nerf_explainability.utils.freq_embeddings import (
    modify_periodicity,
    change_band_to_const,
    change_frequency_band,
    change_dir_frequency_band,
)
from nerf_explainability.utils.camera import get_pixel_coords
from typing import Callable, Optional


def construct_constant_mod_hook(
    band_idx: int, value: float, mask: Optional[torch.Tensor]
) -> Callable[..., tuple[torch.Tensor]]:
    def hook_fn(module: torch.nn.Module, input: tuple[torch.Tensor]):
        band_fn = change_band_to_const(value)
        return change_frequency_band(
            input=input[0], band_idx=band_idx, band_mod_fn=band_fn, mask=mask
        )

    return hook_fn


def construct_periodicity_mod_hook(
    band_idx: int, angle_diff: int, mask: Optional[torch.Tensor]
) -> Callable[..., tuple[torch.Tensor]]:
    def hook_fn(module: torch.nn.Module, input: tuple[torch.Tensor]):
        change_periodicty_fn = modify_periodicity(angle_diff)
        return change_frequency_band(
            input=input[0],
            band_idx=band_idx,
            band_mod_fn=change_periodicty_fn,
            mask=mask,
        )

    return hook_fn


def construct_chain_of_hooks(hooks: list[Callable[..., tuple[torch.Tensor]]]):
    def hook_fn(
        module: torch.nn.Module, input: tuple[torch.Tensor]
    ) -> Callable[..., tuple[torch.Tensor]]:
        x = [*input]
        for f in hooks:
            x[0] = f(module, x)
        return (x[0],)

    return hook_fn


def construct_partial_band_mod_hook(
    band_idx: int,
    K_int: torch.Tensor,
    K_ext: torch.Tensor,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    band_modification_fn: Callable[..., torch.Tensor],
) -> Callable[..., tuple[torch.Tensor]]:
    def hook_fn(module: torch.nn.Module, input: tuple[torch.Tensor]):
        input_3d_point = input[0][:, :3]
        pixel_coords = get_pixel_coords(input_3d_point, K_ext, K_int).T

        # Only modify certain locations in the image at the viewpoing specified by camera matrices
        mask_x = torch.logical_and(
            pixel_coords[:, 0] > min_x, pixel_coords[:, 0] < max_x
        )
        mask_y = torch.logical_and(
            pixel_coords[:, 1] > min_y, pixel_coords[:, 1] < max_y
        )
        mask = torch.logical_and(mask_x, mask_y)

        return change_frequency_band(
            input=input[0],
            band_idx=band_idx,
            band_mod_fn=band_modification_fn,
            mask=mask,
        )

    return hook_fn


def construct_partial_direction_band_mod_hook(
    band_idx: int,
    K_int: torch.Tensor,
    K_ext: torch.Tensor,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    band_modification_fn: Callable[..., torch.Tensor],
):
    def hook_fn(module: torch.nn.Module, input: tuple[torch.Tensor]):
        input_3d_point = input[0][:, :3]
        pixel_coords = get_pixel_coords(input_3d_point, K_ext, K_int).T

        # Only modify certain locations in the image at the viewpoing specified by camera matrices
        mask_x = torch.logical_and(
            pixel_coords[:, 0] > min_x, pixel_coords[:, 0] < max_x
        )
        mask_y = torch.logical_and(
            pixel_coords[:, 1] > min_y, pixel_coords[:, 1] < max_y
        )
        mask = torch.logical_and(mask_x, mask_y)

        return change_dir_frequency_band(
            input=input[0],
            band_idx=band_idx,
            band_mod_fn=band_modification_fn,
            mask=mask,
        )

    return hook_fn
