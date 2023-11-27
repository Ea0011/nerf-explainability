import torch


def get_pixel_coords(x, K_ext, K_int):
    camera_coords = (
        torch.linalg.inv(K_ext) @ torch.cat([x, torch.ones(x.shape[0], 1)], dim=-1).T
    )
    proj = K_int.type_as(camera_coords) @ camera_coords[:3, :]
    img = proj / proj[-1, :]

    return img
