import torch
import torch.nn as nn
import numpy as np

from base_classes.BaseBrain import BaseBrain
from utils.yaml_manager import load_yaml_config

general_info = load_yaml_config('utils/general_info.yaml')

class BasicBrain(BaseBrain):
    def __init__(self, brain_config):
        super().__init__(brain_config)


# ============================================
#  INITIALIZATION METHODS
# ============================================


    def initialize_network(self):
        input_size = self.calculate_input_size()
        output_size = len(self.brain_config.outputs)
        hidden_layers_sizes = self.brain_config.hidden_layers_sizes
        activation = self.brain_config.activation

        # Map activation strings to torch activation layers
        activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU,
        }

        act_layer = activation_map.get(activation.lower(), nn.ReLU)

        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_layers_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(act_layer())
            prev_size = hidden_size

        # Output layer (no activation here, leave to loss/softmax as needed)
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)


    def calculate_input_size(self):
        inputs = self.brain_config.inputs
        vision_range =self.brain_config.vision_range
        vision_size = (2 * vision_range + 1) ** 2
        
        vision_input_size = vision_size * len(inputs.agent_view)
        agent_attr_input_size = sum([getattr(general_info.inputs.agent_attributes_sizes, key)
                     for key in inputs.agent_attributes])

        return vision_input_size + agent_attr_input_size
    
    
# ============================================
#  FORWARD METHODS
# ============================================


    def forward(self, agent_view, agent_attributes, task_info):
        agent_view_inputs = self.get_vision_input(agent_view)
        agent_attributes_inputs = [agent_attributes.get(attr) for attr in self.brain_config.inputs.get('agent_attributes')]
        task_inputs = task_info.values()

        flat_inputs = torch.cat([
            agent_view_inputs.flatten(), 
            torch.tensor(agent_attributes_inputs, dtype=torch.float32, device=agent_view.device),
            torch.tensor(list(task_inputs), dtype=torch.float32, device=agent_view.device)
        ])

        logits = self.network(flat_inputs)
        return logits


    def get_vision_input(self, agent_view):

        vision_channels = self.brain_config.inputs.get('agent_view')

        R_big = agent_view.shape[1] // 2
        R_small = self.vision_range

        center = R_big

        sub_view = agent_view[
            :, 
            center - R_small:center + R_small + 1,
            center - R_small:center + R_small + 1
        ]

        sub_view = sub_view[vision_channels, :, :]

        return sub_view


# ============================================
#  Base methods
# ============================================


    def mutate(self, mutation_std=0.1, mutation_rate=0.05):

        with torch.no_grad():
            for param in self.network.parameters():
                mask = torch.rand_like(param) < mutation_rate
                noise = torch.randn_like(param) * mutation_std
                param.add_(mask * noise)

    def save(self, filepath):
        torch.save(self.network.state_dict(), filepath)


    def load(self, filepath):
        self.initialize_network()
        self.network.load_state_dict(torch.load(filepath, map_location="cpu"))
        self.network.eval()


    def decide_movement(self, logits):
        movement_probs = torch.softmax(logits, dim=0).cpu().numpy()
        movement = np.random.choice(len(movement_probs), p=movement_probs)
        return self.brain_config.outputs[movement]



