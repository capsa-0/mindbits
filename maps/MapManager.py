import torch
import numpy as np
from perlin_numpy import generate_perlin_noise_2d
import torch.nn.functional as F

class MapManager:
    
    @staticmethod
    def generate_terrain(map_config, offset_x=None, offset_y=None):
        scale = map_config.SCALE
        map_width, map_height = map_config.MAP_WIDTH, map_config.MAP_HEIGHT

        if offset_x is None:
            offset_x = np.random.randint(0, 10000)
        if offset_y is None:
            offset_y = np.random.randint(0, 10000)


        noise = generate_perlin_noise_2d(
            (map_height, map_width),
            (scale, scale)
        )

        terrain = torch.from_numpy((noise >= 0.005).astype(np.float32))


        channels = torch.zeros(
            (map_config.N_CHANNELS, map_height, map_width),
            dtype=torch.float32
        )

        channels[0] = terrain
        return channels
    

    @staticmethod
    def extract_patches(tensor, coords, r):
        C, H, W = tensor.shape
        pad = (r, r, r, r)  
        padded = torch.zeros((C, H + 2*r, W + 2*r), dtype=tensor.dtype, device=tensor.device)
        padded[0] = F.pad(tensor[0], pad, value=1)

        for c in range(1, C):
            padded[c] = F.pad(tensor[c], pad, value=0)

        patches = padded.unfold(1, 2*r+1, 1).unfold(2, 2*r+1, 1)

        result = torch.stack([patches[:, x, y] for (x, y) in coords])

        return result

    