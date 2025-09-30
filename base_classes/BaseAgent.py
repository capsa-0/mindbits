import torch
import random
from abc import ABC, abstractmethod
from utils.yaml_manager import dump_line
import copy

class BaseAgent(ABC):

    MOVEMENTS = torch.tensor([[-1,0], (0,-1), (0,0), (0,1), (1,0)], dtype=torch.long)
    
    def __init__(self, agent_config, brain):
        self.agent_config = copy.deepcopy(agent_config)
        self.brain = brain

        self.id = agent_config.id
        self.color = agent_config.color
        if self.color == 'random':
            self.color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]


    def prepare_brain_inputs(self, **raw_inputs):
        agent_view = raw_inputs.get('agent_view')
        agent_attributes = self.get_agent_attributes()
        task_info = raw_inputs.get('task_info')
    
        return agent_view, agent_attributes, task_info

    def decide_movement(self, **raw_inputs):
        a, b, c = self.prepare_brain_inputs(**raw_inputs)
        logits = self.brain.forward(agent_view=a, agent_attributes=b, task_info=c)
        move_idx = torch.argmax(logits).item()
        return move_idx

    def move(self, **raw_inputs):
 
        move_idx = self.decide_movement(**raw_inputs)
        movement = self.MOVEMENTS[move_idx].to(self.device)

        new_pos = self.position + movement
        new_x, new_y = new_pos.tolist()

        terrain_map = raw_inputs.get("agent_view")  

        if (0 <= new_x < self.config.map_config.MAP_WIDTH and 
            0 <= new_y < self.config.map_config.MAP_HEIGHT):
            
            if terrain_map[0, new_y, new_x] != 1:
                self.position = new_pos
                return

        self.alive = 0

    def reset(self):
        self.alive = 1
        self.brain.clean_memory()
        self.position = torch.tensor([0, 0], dtype=torch.long, device=self.device)

    def mutate(self):
        self.brain.mutate()
        self.mutate_color()

    def mutate_color(self):
        def clamp(val):
            return max(0, min(255, val)) 

        r, g, b = self.color
        r_new = clamp(r + random.randint(-10, 10))
        g_new = clamp(g + random.randint(-10, 10))
        b_new = clamp(b + random.randint(-10, 10))
        self.color = [r_new, g_new, b_new]

    def save_agent_config(self, filepath):
        self.agent_config.color = self.color

        dump_line(self.agent_config, filepath)
