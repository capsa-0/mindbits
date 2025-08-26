import random
import numpy as np
from collections import deque
from core.agent import Individual
from core.map_manager import MapManager
from core.config_loader import Config
from core.population_functions import get_population_map, revive_population

class Config_EggHunt:
    MIN_DISTANCE = Config.MAP_HEIGHT // 2


class EggHuntTest:
    def __init__(self, 
                 max_steps=Config.MAX_STEPS, 
                 min_distance=Config.MAP_HEIGHT//2, 
                 agents_population = [Individual(color=(random.randint(0,255), random.randint(0,255), random.randint(0,255))) for n in range (Config.POP_SIZE)]):
        
        self.max_steps = max_steps
        self.min_distance = min_distance
        self.agents_population = agents_population

        self.start_color = (255,0,0)
        self.egg_color = (255,200,10)
        self.egg = Individual(color=(255,200,10))
        self.max_distance = Config.MAP_HEIGHT+Config.MAP_WIDTH

    def reset(self):

        self.steps = 0
        revive_population(self.agents_population)
        self.generate_initial_conditions()

        self.egg.x, self.egg.y = self.egg_position[0], self.egg_position[1]
        self.egg_map = get_population_map([self.egg])
        
        for agent in self.agents_population:
            agent.x, agent.y = self.start_position[0], self.start_position[1]
    

    def generate_initial_conditions(self):

        def random_position():
            return (
                random.randint(Config.PAD, Config.MAP_WIDTH - 1 - Config.PAD),
                random.randint(Config.PAD, Config.MAP_HEIGHT - 1 - Config.PAD)
            )
        

        x0_egg, y0_egg = random_position()
        x0_egg, y0_egg = (Config.PAD,Config.PAD)

        while True:

            x0_start, y0_start = random_position()
 
            if self.distance((x0_egg, y0_egg), (x0_start, y0_start)) > self.min_distance:

                while True:
                    
                    terrain_map = MapManager.generate_terrain(random.randint(0, 10000), random.randint(0, 10000))

                    if self.is_there_a_path((y0_egg, x0_egg), (y0_start, x0_start), terrain_map):

                        break 

                break

        self.egg_position = (x0_egg, y0_egg)
        self.start_position = (x0_start, y0_start)
        self.terrain_map = terrain_map
        

    def distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    
    def is_there_a_path(self, start, end, grid):

        rows, cols = len(grid), len(grid[0])
        (x_start, y_start), (x_end, y_end) = start, end

        # If start or end are obstacles, no path exists
        if grid[x_start][y_start] == 1 or grid[x_end][y_end] == 1:
            return False

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        visited = [[False]*cols for _ in range(rows)]
        queue = deque([(x_start, y_start)])
        visited[x_start][y_start] = True

        while queue:
            x, y = queue.popleft()

            if (x, y) == (x_end, y_end):
                return True  # Path found

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:  # within bounds
                    if not visited[nx][ny] and grid[nx][ny] == 0:
                        visited[nx][ny] = True
                        queue.append((nx, ny))

        return False  # No path found


    def step(self):

        if self.steps > self.max_steps:
            return False
        
        if all([not agent.alive for agent in self.agents_population]):
            return False
        
        self.steps += 1
        for agent in self.agents_population:
            if agent.alive:
                agent.move(self.terrain_map, self.egg_map)

        return True
    
    def get_fitness_list(self):

        fitness_list = [self.get_fitness(agent) for agent in self.agents_population]

        return fitness_list
    
    def get_fitness(self, agent):

        return -self.distance((agent.x, agent.y), self.egg_position)



#class EggHuntDisplay:
 #   @staticmethod