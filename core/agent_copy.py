import torch
import random
from abc import ABC, abstractmethod

from core.config_loader import Config
from models.networks import NETWORKS

NeuralNetwork = NETWORKS[Config.NETWORK_TYPE]

class BaseAgent(ABC):
    """
    Clase base abstracta para cualquier agente.
    Define comportamiento genérico: red neuronal, color, vida, movimiento básico.
    """

    MOVEMENTS = [(-1,0), (0,-1), (0,0), (0,1), (1,0)]
    
    def __init__(self, color=None):
        self.alive = 1
        self.color = color or (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        self.nn = NeuralNetwork()
        self.x, self.y = 0, 0  # posición inicial por defecto

    @abstractmethod
    def get_nn_inputs(self, **raw_inputs):
        """
        Método abstracto: cada agente define qué mapas usa para ver el entorno.
        Debe devolver las entradas que consumirá la red neuronal.
        """
        pass

    def decide_movement(self, **raw_inputs):
        """
        Usa la red neuronal para decidir movimiento.
        """
        inputs = self.get_nn_inputs(**raw_inputs)  # se delega a la subclase
        logits = self.nn.forward(*inputs)
        move_idx = torch.argmax(logits).item()
        return move_idx

    def move(self, **raw_inputs):
        """
        Movimiento genérico, restringido al mapa de terreno.
        """
        movement = self.decide_movement(**raw_inputs)
        dx, dy = self.MOVEMENTS[movement]

        new_x = self.x + dx
        new_y = self.y + dy

        terrain_map = raw_inputs.get("terrain_map")  # obligatorio
        if (0 <= new_x < Config.MAP_WIDTH and 
            0 <= new_y < Config.MAP_HEIGHT and 
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
        self.mutate_color()

    def mutate_color(self):
        def clamp(val):
            return max(0, min(255, val)) 

        r, g, b = self.color
        r_new = clamp(r + random.randint(-10, 10))
        g_new = clamp(g + random.randint(-10, 10))
        b_new = clamp(b + random.randint(-10, 10))
        self.color = (r_new, g_new, b_new)
