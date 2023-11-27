import torch
from nerf_explainability.utils.freq_embeddings import (
    modify_periodicity,
    change_band_to_const,
    change_frequency_band,
)
from typing import Callable, Optional


def construct_constant_mod_hook(
    band_idx: int, value: float, mask: Optional[torch.Tensor]
) -> Callable[..., tuple[torch.Tensor]]:
    def hook_fn(module: torch.nn.Module, input: tuple[torch.Tensor]):
        zero_band_fn = change_band_to_const(value)
        return change_frequency_band(
            input=input[0], band_idx=band_idx, band_mod_fn=zero_band_fn, mask=mask
        )

    return hook_fn


def construct_periodicty_mod_hook(
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
