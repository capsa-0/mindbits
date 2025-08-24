import torch
import torch.nn as nn
import torch.nn.init as init

from core.config_loader import Config

class SimpleRNN(nn.Module):

    def __init__(self, input_size=None, hidden_size=Config.MEMORY_SIZE, output_size=5, vision_radius=Config.VISION_RADIUS):
        super().__init__()
        self.vision_size = 2 * vision_radius + 1

        if input_size is None:
            input_size = 2 * (self.vision_size ** 2)

        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, 30),
            nn.ReLU()
        )

        init.kaiming_normal_(self.encoder[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.encoder[0].bias)
        init.kaiming_normal_(self.encoder[2].weight, nonlinearity="relu")
        nn.init.zeros_(self.encoder[2].bias)


        self.rnn = nn.RNN(30, hidden_size, batch_first=True)


        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                init.kaiming_normal_(param, nonlinearity="relu")
            elif "bias" in name:
                nn.init.zeros_(param)


        self.decoder = nn.Linear(hidden_size, output_size)


        init.kaiming_normal_(self.decoder.weight, nonlinearity="linear")
        nn.init.zeros_(self.decoder.bias)


        self.hidden_state = None



    def forward(self, terrain_vision, population_vision):
        terrain_tensor = torch.from_numpy(terrain_vision).float().view(1, -1)    
        population_tensor = torch.from_numpy(population_vision).float().view(1, -1) 


        combined = torch.cat([terrain_tensor, population_tensor], dim=1) 

        encoded = self.encoder(combined) 

        encoded_seq = encoded.unsqueeze(1)

        if self.hidden_state is None:
            rnn_out, self.hidden_state = self.rnn(encoded_seq)
        else:
            rnn_out, self.hidden_state = self.rnn(encoded_seq, self.hidden_state)

        move_logits = self.decoder(rnn_out.squeeze(1))  
        return move_logits


    def clean_memory(self):
        self.hidden_state = None


    def mutate(self, mutation_rate=Config.MUTATION_RATE, mutation_std=Config.MUTATION_STD, mutation_clip=Config.MUTATION_CLIP):
        for param in self.parameters():
            noise_mask = (torch.rand_like(param) < mutation_rate).float()
            noise = torch.randn_like(param) * mutation_std
            param.data.add_(noise * noise_mask)
            param.data.clamp_(-mutation_clip, mutation_clip)