import torch
from typing import Callable


def pre_forward_hook_registrator(module: torch.nn.Module, fn: Callable):
    return module.register_forward_pre_hook(fn)


def forward_hook_registrator(module: torch.nn.Module, fn: Callable):
    return module.register_forward_hook(fn)
