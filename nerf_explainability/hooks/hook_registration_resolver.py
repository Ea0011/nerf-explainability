from .hook_registrators import *

class HookRegistratorResolver():
    def __init__(self):
        self.hook_to_registrator_mapping = {
            "pre_forward": pre_forward_hook_registrator,
            "forward": forward_hook_registrator,
        }

    def resolve(self, hook_type: str):
        return self.hook_to_registrator_mapping[hook_type]
