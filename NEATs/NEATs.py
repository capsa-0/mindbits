from screeninfo import get_monitors
from noise import pnoise2
from Config import Config
from scipy.ndimage import label
import numpy as np
import random
import copy
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import pygame

def codificar_one_hot(vision, memoria):
    """
    Recibe:
    - vision: vector 1D de enteros entre 0 y 5
    - memoria: vector 1D de enteros entre 0 y 8
    Devuelve:
    - vector 1D con la concatenación de las codificaciones one-hot
    """

    # Codificación one-hot
    vision_onehot = np.eye(6, dtype=np.uint8)[vision.astype(int)].flatten()   # (n, 6)
    memoria_onehot = np.eye(9, dtype=np.uint8)[memoria.astype(int)].flatten() # (n, 9)

    # Concatenar por fila → (n, 15)
    combinados = np.concatenate([vision_onehot, memoria_onehot])

    # Aplanar → vector 1D
    return combinados.flatten()


# Funciones de gestión de población
import pop_management

# Clase de configuración con constantes del programa
from Config import Config

# Clase para la red neuronal de cada individuo
import neat
from neat.nn import FeedForwardNetwork

class QuitEvolution(Exception):
    """Se cerró Pygame voluntariamente."""
    pass

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    'neat_config.txt'
)

colors = [
    (255, 0, 0),      # Rojo
    (0, 255, 0),      # Verde
    (0, 0, 255),      # Azul
    (255, 255, 0),    # Amarillo
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 165, 0),    # Naranja
    (128, 0, 128),    # Púrpura
    (0, 128, 128),    # Verde azulado
    (128, 128, 0),    # Oliva
    (192, 192, 192),  # Gris claro
    (128, 128, 128),  # Gris
    (0, 0, 0),        # Negro
    (255, 255, 255)   # Blanco
]


#Clase para cada individuo
class Individual:

    MOVEMENTS =[(-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)]
    
    def __init__(self, genome=0, species_id=0):

        self.genome = genome
        if genome:
            self.neural_network = FeedForwardNetwork.create(genome, config)
        
        self.color = colors[species_id]
        #self.color = self.generate_color()
        self.original_color = self.color
        self.alive = 1
        self.cooldown = 0
        self.memory = deque([0] * Config.MEMORY_LENGTH, maxlen=Config.MEMORY_LENGTH) # Almacena movimientos (índices de MOVEMENTS)
        self.energy = Config.INITIAL_ENERGY
        self.age = 0
        self.fitness = 0



    def get_vision(self, terrain, population_map, food_map):
        integrated_map = terrain + population_map + food_map
        r = Config.VISION_RADIUS
        x_start, x_end = max(0, self.x - r), min(integrated_map.shape[0], self.x + r + 1)
        y_start, y_end = max(0, self.y - r), min(integrated_map.shape[1], self.y + r + 1)
        vision = integrated_map[y_start:y_end, x_start:x_end]
        return vision


    def decide_movement(self, terrain, population_map, food_map):

        current_vision = self.get_vision(terrain, population_map, food_map)

        mem = np.array(self.memory) 
        vision = current_vision.flatten()

        one_hot_input = codificar_one_hot(vision, mem)

        output = self.neural_network.activate(one_hot_input )

        action = np.argmax(output)

        return action

    def move(self, terrain, population_map, food_map):
        dx, dy = 0, 0
        self.age += 1
        move_made = False



        if self.cooldown <= Config.COOLDOWN_TIME:
            if 0 < self.cooldown:
                self.cooldown -= 1

            movement = self.decide_movement(terrain, population_map, food_map)

            dx, dy = self.MOVEMENTS[movement]
            movement_idx = self.MOVEMENTS.index((dx, dy))

            self.color = self.original_color

            new_x = self.x + dx
            new_y = self.y + dy

            if (0 <= new_x < Config.MAP_WIDTH and 
                0 <= new_y < Config.MAP_HEIGHT and 
                terrain[new_y, new_x] != 1):

                self.energy -= 1 if (dx == 0 and dy == 0) else Config.MOVE_STAND_RATIO

                self.x = new_x
                self.y = new_y
                move_made = True
            else:
                self.alive = 0

            # Actualizar la memoria con el índice del movimiento
            self.memory.append(movement_idx)

        else:
            self.cooldown -= 1
            self.energy -= 1

        movement_idx = self.MOVEMENTS.index((dx, dy))  # Guardar índice del movimiento
        self.memory.append(movement_idx)

        return move_made

    # Iniciar periodo de espera post-reproducción
    def start_cooldown(self):
        self.color = (255, 200, 0)
        self.cooldown = Config.MATING_TIME + Config.COOLDOWN_TIME
        self.fitness += 1 

    # Revivir al individuo para nueva generación
    def revive(self):
        self.cooldown = 0
        self.color = self.original_color
        self.alive = 1
        self.memory = deque([0] * Config.MEMORY_LENGTH, maxlen=Config.MEMORY_LENGTH)
        self.energy = Config.INITIAL_ENERGY
        self.age = 0

    # Generar color heredado o aleatorio
    def generate_color(self, parent=0):
        if parent:
            return tuple(np.clip(parent.color + np.random.randint(-20,20,3), 0, 255))
        return (random.randint(0,255), random.randint(0,255), random.randint(0,255))

    # Posicionamiento aleatorio inicial
    def spawn_random(self):
        self.x = random.randint(0+Config.PAD, Config.MAP_WIDTH-1- Config.PAD)
        self.y = random.randint(0+Config.PAD, Config.MAP_HEIGHT-1- Config.PAD)

    def eat(self,food):
        if food[self.y,self.x] == 2:
            self.energy += Config.FOOD_ENERGY
            food[self.y,self.x] = 0

    def check_vitals(self):
        if self.energy <= 0:
            self.alive = 0


# Clase para generación y manejo del mapa
class MapManager:
    
    @staticmethod
    def generate_terrain(offset_x=0, offset_y=0):
        terrain = np.zeros((Config.MAP_HEIGHT, Config.MAP_WIDTH))
        scale = 25
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0

        # Generar terreno inicial con noise de Perlin
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
    def spawn_food(terrain_matrix, k=0, j=4, p=Config.P_FOOD):
        # Crear una matriz vacía para la comida
        food_matrix = np.zeros_like(terrain_matrix)

        # Generar coordenadas aleatorias para los centros de los clusters
        num_clusters = int(p * terrain_matrix.size)  # Número de clusters basado en la probabilidad
        cluster_centers = np.column_stack((
            np.random.randint(0, terrain_matrix.shape[0], size=num_clusters),  # Coordenadas y
            np.random.randint(0, terrain_matrix.shape[1], size=num_clusters)   # Coordenadas x
        ))

        # Generar radios aleatorios para los clusters
        cluster_radii = k + ((j - k + 1) * np.random.rand(num_clusters)**2).astype(int)

        # Crear una cuadrícula de coordenadas para calcular distancias
        y, x = np.ogrid[:terrain_matrix.shape[0], :terrain_matrix.shape[1]]

        for center, radius in zip(cluster_centers, cluster_radii):
            cy, cx = center
            distance = (x - cx)**2 + (y - cy)**2
            mask = (distance <= radius**2) & (terrain_matrix == 0)  # Dentro del círculo y no en obstáculos
            food_matrix[mask] = 2

        return food_matrix
    
    @staticmethod    
    def add_padding(matrix, padding= Config.PAD):
        # Crear una matriz de 1's con el tamaño aumentado
        padded_matrix = np.ones((
            matrix.shape[0] + 2*padding,  # Filas originales + padding en ambos lados
            matrix.shape[1] + 2*padding   # Columnas originales + padding en ambos lados
        ), dtype=matrix.dtype)
        
        # Insertar la matriz original en el centro
        padded_matrix[padding:-padding, padding:-padding] = matrix
        
        return padded_matrix
  
    

# Clase para manejar la pantalla y los bordes
class Display_borders:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.border_size = Config.EDGE_SIZE
        self.inner_rect = pygame.Rect(
            self.border_size, 
            self.border_size,
            screen_width - 2*self.border_size,
            screen_height - 2*self.border_size
        )
    
    def draw_borders(self, surface, color=(0,0,0)):
        # Dibujar los cuatro bordes
        borders = [
            (0, 0, self.screen_width, self.border_size),  # Borde superior
            (0, self.screen_height - self.border_size, 
             self.screen_width, self.border_size),  # Borde inferior
            (0, 0, self.border_size, self.screen_height),  # Borde izquierdo
            (self.screen_width - self.border_size, 0, 
             self.border_size, self.screen_height)  # Borde derecho
        ]
        for border in borders:
            pygame.draw.rect(surface, color, border)

# Clase para la cámara
class Camera:
    def __init__(self, viewport_rect, cell_size):
        self.offset_x = Config.PAD  # Usar floats para movimiento suave
        self.offset_y = Config.PAD
        self.speed = 6  # Velocidad en celdas por milisegundo
        self.viewport_rect = viewport_rect
        self.cell_size = cell_size

         # Límites mínimos (inferiores)
        self.min_x = Config.PAD  # Límite izquierdo (ajusta según necesites)
        self.min_y = Config.PAD  # Límite superior (ajusta según necesites)
        
        # Límites máximos (superiores)
        self.max_x = Config.MAP_WIDTH - (viewport_rect.width // cell_size) - 1
        self.max_y = Config.MAP_HEIGHT - (viewport_rect.height // cell_size) - 1
        
    
    def move(self, dx, dy):
        self.offset_x += dx * self.speed
        self.offset_y += dy * self.speed
        # Aplica los límites mínimos y máximos
        self.offset_x = max(self.min_x, min(self.offset_x, self.max_x))
        self.offset_y = max(self.min_y, min(self.offset_y, self.max_y))
    
    def world_to_screen(self, x, y):
        return (
            self.viewport_rect.x + (x - self.offset_x) * self.cell_size,
            self.viewport_rect.y + (y - self.offset_y) * self.cell_size
        )




class AgentRenderer:
    @staticmethod
    def draw_agents(surface, camera, agents, cell_size):
        start_x = int(camera.offset_x)
        start_y = int(camera.offset_y)
        end_x = start_x + (camera.viewport_rect.width // cell_size) + 1
        end_y = start_y + (camera.viewport_rect.height // cell_size) + 1
        
        agent_surface = pygame.Surface((camera.viewport_rect.width, camera.viewport_rect.height), 
                                      pygame.SRCALPHA)
        
        for agent in agents:
            if agent.alive and (start_x <= agent.x < end_x) and (start_y <= agent.y < end_y):
                screen_x, screen_y = camera.world_to_screen(agent.x, agent.y)
                x_rel = screen_x - camera.viewport_rect.x
                y_rel = screen_y - camera.viewport_rect.y
                
                # Calcular centro del círculo
                center_x = x_rel + cell_size // 2
                center_y = y_rel + cell_size // 2
                radius = (cell_size - 1) // 2  # Radio para dejar espacio al borde
                
                # Círculo principal (relleno)
                pygame.draw.circle(
                    agent_surface,
                    agent.color,
                    (center_x, center_y),
                    radius
                )
                
                # Contorno rojo (borde)
                pygame.draw.circle(
                    agent_surface,
                    (0, 0, 0),  # Color outline
                    (center_x, center_y),
                    radius,
                    1  # Grosor del borde (1px)
                )
        
        surface.blit(agent_surface, camera.viewport_rect.topleft)


# Clase principal de simulación
class Simulation:
    def __init__(self, population = [Individual() for _ in range(Config.POPULATION_SIZE)], n_gen=0, repro_rate=[]):
        self.population = population
        self.running = True
        self.generation_count = int(n_gen)
        # Parámetros monitoreados
        self.repro_rate = repro_rate


    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.FULLSCREEN)
        self.clock = pygame.time.Clock()
        self.running = True
        self.display = Display_borders(Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT)
        self.camera = Camera(self.display.inner_rect, Config.CELL_SIZE)

    # Manejar apareamiento entre individuos
    def process_mating(self):
        couples = defaultdict(list)
        for ind in self.population:
            if ind.alive:
                couples[(ind.x, ind.y)].append(ind)
        
        for pos, inds in couples.items():
            if len(inds) == 2:
                father, mother = inds
                if (father.alive and mother.alive) and (father.cooldown==0 and mother.cooldown==0) and ((father.energy + mother.energy) > Config.SON_ENERGY) and (father.age > Config.ADULT_AGE and mother.age > Config.ADULT_AGE):
                    father.start_cooldown()
                    mother.start_cooldown()

    def check_vitals(self):
        for ind in self.population:
            ind.check_vitals()

    def get_food_map_color(self):  
        color_food = np.zeros((self.food.shape[0],self.food.shape[1], 3), dtype=np.uint8)

        # Asignar colores basados en el terreno
        color_food[self.food == 2] = Config.FOOD_COLOR  # COMIDA

        return color_food

    def draw_terrain(self):

        int_full_map = self.terrain + self.food
        full_map = np.zeros((self.terrain.shape[0],self.terrain.shape[1], 3), dtype=np.uint8)

        # Asignar colores basados en el terreno
        full_map[int_full_map == 2] = Config.FOOD_COLOR  
        full_map[int_full_map == 1] = Config.OBSTACLE_COLOR
        full_map[int_full_map == 0] = Config.FONDO  

        # Dibujado
        #self.screen.fill(Config.FONDO)  # Fondo base

        self.display.draw_borders(self.screen)  # Dibujar bordes
        

        # Calcular área visible
        start_x = int(self.camera.offset_x)
        start_y = int(self.camera.offset_y)
        end_x = start_x + self.display.inner_rect.width // Config.CELL_SIZE + 1
        end_y = start_y + self.display.inner_rect.height // Config.CELL_SIZE + 1
        
        # Dibujar celdas en el área visible
        self.screen.set_clip(self.display.inner_rect)
        for y in range(start_y, min(end_y, Config.MAP_HEIGHT)):
            for x in range(start_x, min(end_x, Config.MAP_WIDTH)):
                screen_x, screen_y = self.camera.world_to_screen(x, y)
                color = full_map[y][x]
                pygame.draw.rect(self.screen, color, 
                            (screen_x, screen_y, Config.CELL_SIZE, Config.CELL_SIZE))
                
    
        self.screen.set_clip(None)

    # Reproducir individuos y crear nueva generación
    def reproduce(self):
        self.process_mating()
        for ind in self.population:
            if ind.cooldown == Config.MATING_TIME + Config.COOLDOWN_TIME:  # Individuos listos para reproducirse
                self.next_gen.append(Individual(ind.genome))

    # Asegurar posiciones válidas para nuevos individuos
    def adjust_positions(self):
        positions = []
        for ind in self.population:   
            while ((ind.x, ind.y) in positions) or self.terrain[ind.y, ind.x] == 1:
                ind.spawn_random()
            positions.append((ind.x, ind.y))
 
    # Reiniciar posiciones y estados de individuos
    def reset_individuals(self):
        for ind in self.population:
            ind.x = random.randint(0+Config.PAD, Config.MAP_WIDTH-1- Config.PAD)
            ind.y = random.randint(0+Config.PAD, Config.MAP_HEIGHT-1- Config.PAD)
            ind.revive()
        self.adjust_positions()

    # Crear mapa de densidad de población
    def get_population_map(self):

        pop_map = np.zeros((Config.MAP_HEIGHT+Config.PAD*2, Config.MAP_WIDTH+Config.PAD*2), dtype=int)
        
        for ind in self.population:
            if ind.alive:
                if pop_map[ind.y, ind.x] == 0:
                    pop_map[ind.y, ind.x] = 3
                elif pop_map[ind.y, ind.x] == 3:
                    pop_map[ind.y, ind.x] = 4
                else:
                    pop_map[ind.y, ind.x] = 5
   
        return pop_map
        

    def pop_eat(self):
        for ind in self.population:
                    if ind.alive:
                        ind.eat(self.food)


    def run_generation(self, n_sims=10):
 
        self.next_gen = []
        self.simulation_count = 1
        self.mating_count = 0
        self.total_steps = 0
        self.terrain = MapManager.generate_terrain(random.randint(0, 10000), random.randint(0, 10000))
        food = MapManager.spawn_food(self.terrain)

        
        while self.simulation_count <= n_sims:
            self.food = copy.deepcopy(food)
            self.step = 0
            if self.running:

                self.reset_individuals()

                while any(ind.alive for ind in self.population) and self.running:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                            return
                        elif event.type == pygame.MOUSEBUTTONDOWN and self.close_button.collidepoint(event.pos):
                            self.running = False
                            pygame.quit()
                            return 1

                        # Movimiento de cámara
                        keys = pygame.key.get_pressed()
                        if keys[pygame.K_LEFT]: self.camera.move(-1, 0)
                        if keys[pygame.K_RIGHT]: self.camera.move(1, 0)
                        if keys[pygame.K_UP]: self.camera.move(0, -1)
                        if keys[pygame.K_DOWN]: self.camera.move(0, 1)


                    self.check_vitals()
                    self.total_steps += 1
                    self.step += 1
                    population_map = self.get_population_map()

                    self.pop_eat()
                    for ind in self.population:
                        if ind.alive:

                            ind.move(self.terrain, population_map, self.food)

                    self.reproduce()
                    self.check_vitals()

                    self.draw_terrain()
                    AgentRenderer.draw_agents(self.screen, self.camera, self.population, Config.CELL_SIZE)

   
                    self.display_info()
                    pygame.display.flip()
                    
                    self.clock.tick(Config.FPS)  # Limitar a 60 FPS




                self.simulation_count += 1

        fit = [ind.fitness for ind in self.population]
        return [x / n_sims for x in fit]
        


    def run_blind_generation(self, n_sims=10):
 
        self.next_gen = []
        self.simulation_count = 1
        self.mating_count = 0
        self.total_steps = 0
        self.terrain = MapManager.generate_terrain(random.randint(0, 10000), random.randint(0, 10000))
        food = MapManager.spawn_food(self.terrain)

        
        while self.simulation_count <= n_sims:
            self.food = copy.deepcopy(food)
            self.step = 0
            if self.running:

                self.reset_individuals()

                while any(ind.alive for ind in self.population) and self.running:

                    self.check_vitals()
                    self.total_steps += 1
                    self.step += 1
                    population_map = self.get_population_map()

                    self.pop_eat()

                    for ind in self.population:
                        if ind.alive:

                            ind.move(self.terrain, population_map, self.food)

                    self.reproduce()
                    self.check_vitals()
                print(f'GEN {self.generation_count} | SIM {self.simulation_count} | COUNT {len(self.next_gen)}/{Config.POPULATION_SIZE}')
                self.simulation_count += 1

        fit = [ind.fitness for ind in self.population]
        return [x / n_sims for x in fit]


 
    # Bucle principal de ejecución
    def run(self,blind=0,n=100):
        
        if blind:
            for n in range(n):

                self.run_blind_generation()
                self.generation_count += 1
                if n%5 == 0:
                    pop_management.guardarPoblacion(self.population, "out/pob", "nn", self.generation_count, self.repro_rate)
                print(f'-------------> GEN {self.generation_count}')

        else:
            self.init_pygame()
            while self.running:
                end = self.run_generation()
                if end:
                    return
                self.generation_count += 1
                if self.generation_count%20 == 0:
                    pop_management.guardarPoblacion(self.population, "out/pob", "nn", self.generation_count, self.repro_rate)
   
            
    # Mostrar información en pantalla
    def display_info(self):

        self.display.draw_borders(self.screen)

        font = pygame.font.Font(Config.FONT, 20)  
        font_info = pygame.font.Font(Config.FONT, 15) 
        
        # Texto de generación actual
        gen_text = font.render(f"GEN {self.generation_count} | sim {self.simulation_count} | cruces {len(self.next_gen)} | paso {self.step}", True, (255, 255, 255)) 
        self.screen.blit(gen_text, (Config.INFO_X, 10))


        # Configuración mostrada
        down_text=f'Tamaño pob - {Config.POPULATION_SIZE} | Radio de visión - {Config.VISION_RADIUS} | Tamaño memoria - {Config.MEMORY_LENGTH}' # | Arquitectura - {[Config.INPUT_SIZE] + Config.NN_ARCHITECTURE + [9]}'
        pop_text = font_info.render(down_text, True, (255, 255, 255))
        self.screen.blit(pop_text, (Config.INFO_X, Config.INFO_Y))        
        
        # Botón de cierre
        inverse_size = 60
        self.close_button = pygame.Rect(
            Config.SCREEN_WIDTH-Config.SCREEN_WIDTH//inverse_size, 
            0, 
            Config.SCREEN_WIDTH//inverse_size, 
            Config.SCREEN_WIDTH//inverse_size
        ) 
        pygame.draw.rect(self.screen, (200, 80, 100), self.close_button)

    def save_gen_data(self, next_n, n_sims):
        out_dir = 'out/'
        self.repro_rate.append(next_n/(n_sims*Config.POPULATION_SIZE))
        plt.style.use('dark_background')
        # Figura 2: Evolución de cruces/número de simulaciones
        plt.figure(figsize=(20, 6))  
        plt.plot(self.repro_rate, 'o', markersize=8, markerfacecolor='#fcba03')

        x = np.arange(len(self.repro_rate))
        y = np.array(self.repro_rate)

        if len(self.repro_rate) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            y_pred = p(x)
        
        # Calcular R²
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
            
            # Calcular coeficiente de correlación de Pearson (r)
            r = np.corrcoef(x, y)[0, 1]
            
            # Texto para el gráfico
            texto = (
                f'y = {z[0]:.2f}x + {z[1]:.2f}\nR² = {r2:.2f}\n'
                f"Correlación: {r:.2f}"
            )
            
            plt.plot(x, p(x), 'r--', linewidth=3, alpha=1)
            plt.text(0.05, 0.95, texto, transform=plt.gca().transAxes,
                    fontsize=20, verticalalignment='top', linespacing=1.5,
                    bbox=dict(facecolor='black', alpha=1, edgecolor='white'))

        # Personalización de ejes y título

        plt.xlabel("GEN", labelpad=10)
        plt.ylabel("tasa de reproducción", labelpad=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # Cuadrícula y mejoras visuales
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim(-1, self.generation_count+1) 

        # Ajustar márgenes y guardar
        plt.tight_layout()
        plt.savefig(out_dir+"Cruces.png", dpi=300, bbox_inches='tight')
        plt.close()



counter = 0
def evaluate_genomes1(genomes, config):
    global counter

    # Crear población de agentes con especie_id
    population = [Individual(genome) for genome_id, genome in genomes]

    # Ejecutar simulación
    sim = Simulation(population, n_gen=counter)


    n_sons = sim.run_blind_generation(n_sims=Config.MIN_ITER)
    counter += 1
    # Asignar fitness
    for fit, (genome_id, genome) in zip(n_sons, genomes):
        genome.fitness = fit

    return n_sons

n_blind=Config.BLIND_GENS

def evaluate_genomes2(genomes, config):
    global n_blind, species
    counter = n_blind
    # Crear población de agentes con especie_id
    population = [Individual(genome,species_id=species[genome_id]) for genome_id, genome in genomes]

    # Ejecutar simulación
    sim = Simulation(population, n_gen=counter)

    sim.init_pygame()
    n_sons = sim.run_generation(n_sims=Config.MIN_ITER)

    if n_sons == 1:
        raise QuitEvolution()
    n_blind += 1

    # Asignar fitness
    for fit, (genome_id, genome) in zip(n_sons, genomes):
        genome.fitness = fit

    return n_sons

def obtener_especies_por_genoma(population):

    especie_por_genoma = {}

    for species_id, species in population.species.species.items():
        for genome_id in species.members:
            especie_por_genoma[genome_id] = species_id

    return especie_por_genoma

species = {}
for i in range(Config.POPULATION_SIZE):
    species[i+1] = 0

def run_neat():
    global species
    # Crear población inicial
    population = neat.Population(config)
    
    # Add reporters para monitorear progreso
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    global n_blind
    # Ejecutar evolución
    for i in range (n_blind):
        population.run(evaluate_genomes1, 1)
        species = obtener_especies_por_genoma(population)
 
    for i in range (1000):
        population.run(evaluate_genomes2, 1)
        species = obtener_especies_por_genoma(population)


run_neat()