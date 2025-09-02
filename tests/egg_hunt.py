import random
import numpy as np
from collections import deque
from core.agent_copy import BaseAgent
from core.map_manager import MapManager
from core.config_loader import Config
from core.population_functions import revive_population


class   EggHuntAgent(BaseAgent):

    def __init__(self, color=None):
        super().__init__(color=color)  
        self.found_egg = False        

    def get_nn_inputs(self, **raw_inputs):

        r = Config.VISION_RADIUS

        y_start, y_end = max(0, self.y - r), min(raw_inputs['terrain_map'].shape[0], self.y + r + 1)
        x_start, x_end = max(0, self.x - r), min(raw_inputs['terrain_map'].shape[1], self.x + r + 1)

        terrain_vision = raw_inputs['terrain_map'][y_start:y_end, x_start:x_end]
        found_egg = self.found_egg * 1.0

        return terrain_vision, np.array([[found_egg]])
    
    def revive(self):
        super().revive()
        self.found_egg = False

    def found_the_egg(self, egg_position):
        return (self.y, self.x) == egg_position 
    
    def move(self, **raw_inputs):
        super().move(**raw_inputs)
        self.found_egg = self.found_the_egg(raw_inputs['egg_position'])




class EggHuntTest:
    def __init__(self, 
                 max_steps=Config.MAX_STEPS, 
                 min_distance=Config.MIN_DISTANCE_egg, 
                 agents_population = [EggHuntAgent() for n in range (Config.POP_SIZE)]):
        
        self.max_steps = max_steps
        self.min_distance = min_distance
        self.agents_population = agents_population

        self.start_color = (255,0,0)
        self.egg_color = (255,200,10)
        self.max_distance = Config.MAP_HEIGHT + Config.MAP_WIDTH

        self.terrain_map = MapManager.generate_terrain(random.randint(0, 10000), random.randint(0, 10000))

    def reset(self):

        self.steps = 0
        revive_population(self.agents_population)
        self.generate_initial_conditions()
        
        for agent in self.agents_population:
            agent.x, agent.y = self.start_position[0], self.start_position[1]
    

    def generate_initial_conditions(self):
        def random_position():
            return (
                random.randint(Config.PAD, Config.MAP_WIDTH - 1 - Config.PAD),
                random.randint(Config.PAD, Config.MAP_HEIGHT - 1 - Config.PAD)
            )

        while True:
            # Posición del huevo
            x_egg, y_egg = (Config.PAD, Config.PAD)  # fijo o random_position()
            
            success = False
            for _ in range(10):  # intentos de generar start_position válida
                x_start, y_start = random_position()
                if (self.distance((x_egg, y_egg), (x_start, y_start)) > self.min_distance and
                    self.is_there_a_path((y_egg, x_egg), (y_start, x_start), self.terrain_map)):
                    success = True
                    break  # posición válida encontrada

            if success:
                self.egg_position = (x_egg, y_egg)
                self.start_position = (x_start, y_start)
                break  # terminado, mapa válido
            else:
                # No se encontró posición válida: generar un nuevo mapa
                seed_x = random.randint(0, 10000)
                seed_y = random.randint(0, 10000)
                self.terrain_map = MapManager.generate_terrain(seed_x, seed_y)


        

    def distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    
    def is_there_a_path(self, start, end, grid):
        rows, cols = len(grid), len(grid[0])
        row_start, col_start = start
        row_end, col_end = end

        if grid[row_start][col_start] == 1 or grid[row_end][col_end] == 1:
            return False

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visited = [[False]*cols for _ in range(rows)]
        queue = deque([(row_start, col_start)])
        visited[row_start][col_start] = True

        while queue:
            r, c = queue.popleft()
            if (r, c) == (row_end, col_end):
                return True
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not visited[nr][nc] and grid[nr][nc] == 0:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
        return False



    def step(self):

        if self.steps > self.max_steps:
            return False
        
        if all([not agent.alive for agent in self.agents_population]):
            return False
        
        self.steps += 1
        for agent in self.agents_population:
            if agent.alive:
                agent.move(terrain_map=self.terrain_map, egg_position=self.egg_position)

        return True
    
    def get_fitness_list(self):

        fitness_list = [self.get_fitness(agent) for agent in self.agents_population]

        return fitness_list
    
    def get_fitness(self, agent):
        #delta_d = -(agent.x - self.start_position[0] + agent.y - self.start_position[1])
        found_egg = agent.found_egg * 100
        died = -10 if not agent.alive else 0

        return found_egg + died