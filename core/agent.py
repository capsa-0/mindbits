import torch
import random

from core.config_loader import Config
from models.networks import NETWORKS
NeuralNetwork = NETWORKS[Config.NETWORK_TYPE]

class Individual:

    MOVEMENTS =[(-1,0), (0,-1), (0,0), (0,1), (1,0)]
    
    def __init__(self, color=(0,0,255)):

        self.alive = 1
        self.color=color
        self.nn = NeuralNetwork() 


    def get_vision(self, terrain_map, population_map):
        r = Config.VISION_RADIUS

        y_start, y_end = max(0, self.y - r), min(terrain_map.shape[0], self.y + r + 1)
        x_start, x_end = max(0, self.x - r), min(terrain_map.shape[1], self.x + r + 1)

        terrain_vision = terrain_map[y_start:y_end, x_start:x_end]
        population_vision = population_map[y_start:y_end, x_start:x_end]

        return terrain_vision, population_vision



    def decide_movement(self, terrain_map,population_map):

        terrain_vision, population_vision = self.get_vision(terrain_map, population_map)

        logits = self.nn.forward(terrain_vision, population_vision)

        move_idx = torch.argmax(logits).item()  

        return move_idx

    def move(self, terrain_map, population_map):

        movement = self.decide_movement(terrain_map, population_map)

        dx, dy = 0, 0
        dx, dy = self.MOVEMENTS[movement]

        new_x = self.x + dx
        new_y = self.y + dy

        if (0 <= new_x < Config.MAP_WIDTH and 0 <= new_y < Config.MAP_HEIGHT and 
                terrain_map[new_y, new_x] != 1):

            self.x = new_x
            self.y = new_y

        else:
                self.alive = 0


    def revive(self):
        self.cooldown = 0
        self.alive = 1
        self.nn.clean_memory()

    
    def mutate(self):
         self.nn.mutate()
         self.color = self.mutate_color(self.color)

    def mutate_color(self):

        def clamp(val):
            return max(0, min(255, val)) 

        r, g, b = self.color
        r_new = clamp(r + random.randint(-10, 10))
        g_new = clamp(g + random.randint(-10, 10))
        b_new = clamp(b + random.randint(-10, 10))

        self.color = (r_new, g_new, b_new)
