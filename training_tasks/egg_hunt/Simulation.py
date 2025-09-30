import random
import numpy as np
from collections import deque
from base_classes.BaseSimulation import BaseSimulation   
from maps.MapManager import *



class EggHuntTest(BaseSimulation):


    def __init__(self, task_config, population):
        super().__init__(task_config, population)
        

# ============================================
#   RESET METHODS
# ============================================


    def reset(self, mode):
        if mode == "generation":
            self.generate_valid_map()
        
        if mode == "run":
            self.generate_initial_positions()
            self.reset_population()
            self.steps = 0

            self.dist_start_egg = self.calculate_distance(self.start_position, self.eggs_positions)


    def generate_initial_positions(self):

        if self.task_config.start_mode == "static":
            self.start_position = np.array(self.task_config.start_position)
        elif self.task_config.start_mode == "dynamic":
            self.start_position = self.get_random_position()

        self.eggs_positions = []

        if self.task_config.eggs_mode == "static":
            self.eggs_positions = [np.array(p) for p in self.task_config.eggs_positions]

        elif self.task_config.eggs_mode == "dynamic":
            num_eggs = self.task_config.eggs_amount
            for _ in range(num_eggs):
                egg_pos = self.get_random_position()

                while self.calculate_distance(self.start_position, egg_pos) < self.task_config.min_distance or not self.is_reachable(egg_pos):
                    egg_pos = self.get_random_position()
                self.eggs_positions.append(egg_pos)


    def generate_valid_map(self):
        if self.task_config.map_config.mode == "static":
            if self.map is None:
                self.map = MapManager.generate_terrain(self.task_config.map_config)
                while not self.is_valid_map():
                    self.map = MapManager.generate_terrain(self.task_config.map_config)

        elif self.task_config.map_config.mode == "dynamic":
            self.map = MapManager.generate_terrain(self.task_config.map_config)
            while not self.is_valid_map():
                self.map = MapManager.generate_terrain(self.task_config.map_config)


    def get_random_position(self):
        return np.array([
            random.randint(0, self.task_config.map_config.MAP_WIDTH - 1),
            random.randint(0, self.task_config.map_config.MAP_HEIGHT - 1)
        ])



    def calculate_distance(self, pos1, pos2, metric="euclidean"):
        # Si pos2 es lista o array de varias posiciones
        if isinstance(pos2, (list, np.ndarray)) and len(pos2) > 0 and isinstance(pos2[0], (list, np.ndarray)):
            distances = [self.calculate_distance(pos1, np.array(p), metric) for p in pos2]
            return min(distances)
        
        # Caso normal: pos2 es un solo punto
        pos2 = np.array(pos2)
        if metric == "euclidean":
            return np.linalg.norm(pos1 - pos2)
        elif metric == "manhattan":
            return np.abs(pos1 - pos2).sum()
        elif metric == "chebyshev":
            return np.abs(pos1 - pos2).max()
        else:
            raise ValueError(f"Metric '{metric}' not supported.")



    def is_valid_map(self):

        h, w = self.map.shape[1], self.map.shape[2]
        grid = self.map[0].numpy()


        walkable = [(x, y) for y in range(h) for x in range(w) if grid[y, x] == 0]
        if len(walkable) < 2:
            return False


        start = walkable[0]
        visited = {start: 0}
        queue = deque([start])


        while queue:
            x, y = queue.popleft()
            dist = visited[(x, y)]

            if dist >= self.task_config.min_distance:
                return True 

            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if grid[ny, nx] == 1 and (nx, ny) not in visited:
                        visited[(nx, ny)] = dist + 1
                        queue.append((nx, ny))

        return False
    

    def is_reachable(self, egg_pos):

        h, w = self.map.shape[1], self.map.shape[2]
        grid = self.map[0].numpy()

        start = (int(self.start_position[0]), int(self.start_position[1]))
        egg = (int(egg_pos[0]), int(egg_pos[1]))

        visited = set()
        queue = deque([start])

        while queue:
            x, y = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))

            # ¿llegamos al huevo?
            if (x, y) == egg:
                return True

            # explorar vecinos
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if grid[ny, nx] == 1 and (nx, ny) not in visited:
                        queue.append((nx, ny))

        return False

    def reset_population(self):
        for agent in self.population:
            agent.reset()


# ============================================
#  STEP METHODS
# ============================================


    def step(self):
        if self.steps >= self.task_config.max_steps:
            return False
        
        if all([not agent.alive for agent in self.population]):
            return False
        
        self.steps += 1
        positions = [agent.position for agent in self.population if agent.alive]
        agents_view = MapManager.extract_patches(self.map, positions, self.task_config.VISION_RANGE)

        for i, agent in enumerate(self.population):
            if agent.alive:
                agent.move({'agent_view':agents_view[i]})

        return True


# ============================================
#  FITNESS METHOD
# ============================================
    

    def compute_fitness(self, agent):
        distance_to_egg = self.calculate_distance(agent.position, self.eggs_positions)
        return (100 * (self.dist_start_egg - distance_to_egg)) / (self.task_config.config_map.MAP_HEIGHT + self.task_config.config_map.MAP_WIDTH)