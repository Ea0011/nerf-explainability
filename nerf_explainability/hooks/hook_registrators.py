def pre_forward_hook_registrator(module, fn):
    return module.register_forward_pre_hook(fn)

def forward_hook_registrator(module, fn):
    return module.register_forward_hook(fn)