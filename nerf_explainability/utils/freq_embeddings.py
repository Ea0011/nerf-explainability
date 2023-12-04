import torch
from nerf_explainability.utils.camera import get_pixel_coords
from typing import Callable, Optional


def get_bands(band_idx: int) -> tuple[int]:
    band_length = (
        6 if band_idx > 0 else 3
    )  # sin + cos of 3D points. Total 6 coordinates

    band_start = (band_idx - 1) * (band_length) + 3 if band_idx > 1 else band_idx * 3
    band_end = band_start + band_length

    return (band_start, band_end)


def modify_periodicity(angle_diff: int) -> Callable[..., torch.Tensor]:
    def fn(x: torch.Tensor, band_idx: int) -> torch.Tensor:
        embed_fn_sin = lambda p: torch.sin(2.0 ** (band_idx - 1 + angle_diff) * p)
        embed_fn_cos = lambda p: torch.cos(2.0 ** (band_idx - 1 + angle_diff) * p)

        return torch.cat([embed_fn_sin(x), embed_fn_cos(x)], dim=-1)

    return fn


def change_band_to_const(val: float) -> Callable[..., torch.Tensor]:
    def fn(x: torch.Tensor, _) -> torch.Tensor:
        num_pts = x.shape[0]
        return torch.empty((num_pts, 6)).fill_(val).type_as(x)

    return fn


def change_frequency_band(
    input: torch.Tensor,
    band_idx: int,
    band_mod_fn: Callable[..., Callable],
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    band_start, band_end = get_bands(band_idx)
    embed_in = input[:, 0:3].clone()

    res = input.clone()
    res[:, band_start:band_end] = band_mod_fn(embed_in, band_idx)

    if mask is not None:
        change_mask = mask.to(torch.float).unsqueeze(1).expand_as(input)
        ablated_points = (1 - change_mask) * input + (change_mask * res)
    else:
        ablated_points = res

    # NOTE: in place modification to affect the skip connection as well
    input.copy_(ablated_points)

    return input


def change_dir_frequency_band(
    input: torch.Tensor,
    band_idx: int,
    band_mod_fn: Callable[..., Callable],
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    band_start, band_end = get_bands(band_idx)

    features, input_dirs = torch.split(input, [256, 27], dim=-1)
    res = input_dirs.clone()
    res[:, band_start:band_end] = band_mod_fn(res[:, 0:3], band_idx)

    if mask is not None:
        change_mask = mask.to(torch.float).unsqueeze(1).expand_as(input_dirs)
        ablated_points = (1 - change_mask) * input_dirs + (change_mask * res)
    else:
        ablated_points = res

    input.copy_(torch.cat([features, ablated_points], dim=-1))

    return input
