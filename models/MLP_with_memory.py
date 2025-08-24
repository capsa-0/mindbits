import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config_loader import Config

class MLP_with_memory(nn.Module):
    def __init__(self, input_size=None, output_size=5, vision_radius=Config.VISION_RADIUS,
                 architecture=[80, 60, 40, 20], memory_size=Config.MEMORY_SIZE):
        super().__init__()
        self.vision_size = 2 * vision_radius + 1
        self.output_size = output_size
        self.memory_size = memory_size

        if input_size is None:
            input_size = 2 * (self.vision_size ** 2)

        memory_input_size = output_size * memory_size
        input_size = input_size + memory_input_size

        if architecture is None:
            architecture = Config.ARCHITECTURE

        layers = []
        prev_size = input_size

        for hidden_size in architecture:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

        self.clean_memory()

    def clean_memory(self):

        self.memory = torch.zeros(self.memory_size, self.output_size)

    def update_memory(self, logits):
   
        action = torch.argmax(logits, dim=1) 
        one_hot = F.one_hot(action, num_classes=self.output_size).float()  

        
        self.memory = torch.roll(self.memory, shifts=-1, dims=0)
        self.memory[-1] = one_hot.squeeze(0)  

    def forward(self, terrain_vision, population_vision):

        terrain_tensor = torch.from_numpy(terrain_vision).float().view(1, -1)
        population_tensor = torch.from_numpy(population_vision).float().view(1, -1)


        memory_tensor = self.memory.flatten().unsqueeze(0) 


        combined = torch.cat([terrain_tensor, population_tensor, memory_tensor], dim=1)

        logits = self.network(combined) 

        self.update_memory(logits)

        return logits

    def mutate(self, mutation_rate=Config.MUTATION_RATE, mutation_std=Config.MUTATION_STD,
               mutation_clip=Config.MUTATION_CLIP):
        for param in self.parameters():
            noise_mask = (torch.rand_like(param) < mutation_rate).float()
            noise = torch.randn_like(param) * mutation_std
            param.data.add_(noise * noise_mask)
            param.data.clamp_(-mutation_clip, mutation_clip)
