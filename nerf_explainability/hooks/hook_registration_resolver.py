from .hook_registrators import *
from typing import Callable


class HookRegistratorResolver:
    def __init__(self) -> None:
        self.hook_to_registrator_mapping = {
            "pre_forward": pre_forward_hook_registrator,
            "forward": forward_hook_registrator,
        }

    def resolve(self, hook_type: str) -> Callable[..., Callable]:
        return self.hook_to_registrator_mapping[hook_type]
