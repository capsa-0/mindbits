import numpy as np
from core.config_loader import Config
from noise import pnoise2

class MapManager:
    
    @staticmethod
    def generate_terrain(offset_x=0, offset_y=0):
        terrain = np.zeros((Config.MAP_HEIGHT, Config.MAP_WIDTH))
        scale = 8
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0

        map_width, map_height = Config.MAP_WIDTH, Config.MAP_HEIGHT

        for y in range(map_height):
            for x in range(map_width):
                noise_val = pnoise2((x + offset_x)/scale, (y + offset_y)/scale, 
                                    octaves=octaves, 
                                    persistence=persistence, 
                                    lacunarity=lacunarity)
                terrain[y, x] = 1 if noise_val >= 0.005 else 0


        padded = MapManager.add_padding(terrain)

        return padded
    
    
    @staticmethod    
    def add_padding(matrix, padding= Config.PAD):

        padded_matrix = np.ones((
            matrix.shape[0] + 2*padding,  
            matrix.shape[1] + 2*padding   
        ), dtype=matrix.dtype)
        

        padded_matrix[padding:-padding, padding:-padding] = matrix
        
        return padded_matrix