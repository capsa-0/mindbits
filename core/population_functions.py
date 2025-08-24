import numpy as np
from core.config_loader import Config

def get_population_map(population):

    pop_map = np.zeros((Config.MAP_HEIGHT + Config.PAD * 2, Config.MAP_WIDTH+Config.PAD * 2), dtype=int)
        
    for ind in population:
        if ind.alive:
                pop_map[ind.y, ind.x] += 1
   
    return pop_map

def revive_population(population):
    for agent in population:
        agent.alive = 1