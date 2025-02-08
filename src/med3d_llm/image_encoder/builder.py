from .vit import VisionTransformer3dTower


def build_vision_tower(config, **kwargs):
    vision_tower = getattr(config, 'vision_tower', None)
    if 'vit3d' in vision_tower.lower():
        return VisionTransformer3dTower(config, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')