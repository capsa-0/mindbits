import torch
import torch.nn as nn
import random
from config_loader import Config

class VisionOnlyNN(nn.Module):
    def __init__(self, vision_radius=Config.VISION_RADIUS):
        super().__init__()
        self.vision_size = 2 * vision_radius + 1
        
        # Capa convolucional para 2 canales (terreno y población)
        self.conv = nn.Conv2d(2, 16, kernel_size=3, padding=1)  # 2 canales de entrada, 16 filtros
        
        # GRU para memoria temporal
        rnn_input_size = 16 * self.vision_size**2  # 16 canales * tamaño de visión
        self.rnn = nn.GRU(rnn_input_size, 32, batch_first=True)
        
        # Capa de decisión
        self.decoder = nn.Linear(32, 9)  # 9 movimientos posibles
        
        # Estado oculto persistente
        self.hidden_state = None

    def forward(self, terrain_vision, population_vision):
        # Convertir inputs NumPy a tensores PyTorch
        terrain_tensor = torch.from_numpy(terrain_vision).float().unsqueeze(0)  # (1, H, W)
        population_tensor = torch.from_numpy(population_vision).float().unsqueeze(0)  # (1, H, W)
        
        # Combinar en un tensor de 2 canales
        combined = torch.stack([terrain_tensor, population_tensor], dim=1)  # (1, 2, H, W)
        
        # Aplicar convolución
        conv_out = self.conv(combined)  # (1, 16, H, W)
        
        # Aplanar manteniendo la dimensión batch
        conv_flat = conv_out.view(1, 1, -1)  # (1, 1, 16*H*W)
        
        # Procesar con GRU
        if self.hidden_state is None:
            rnn_out, self.hidden_state = self.rnn(conv_flat)
        else:
            rnn_out, self.hidden_state = self.rnn(conv_flat, self.hidden_state)
        
        # Decodificar movimiento
        move_logits = self.decoder(rnn_out.squeeze(1))  # (1, 9)
        return move_logits

    def clean_memory(self):
        self.hidden_state = None

    def mutate(self, mutation_rate=Config.MUTATION_RATE, mutation_std=Config.MUTATION_STD, mutation_clip=Config.MUTATION_CLIP):

        for param in self.parameters():
            if random.random() < mutation_rate:
                noise = torch.randn_like(param) * mutation_std
                param.data.add_(noise)
                if hasattr(Config, "MUTATION_CLIP"):
                    param.data.clamp_(-mutation_clip, mutation_clip)
                    
import torch.nn.init as init

class SimpleRNN(nn.Module):

    def __init__(self, input_size=None, hidden_size=Config.MEMORY_SIZE, output_size=5, vision_radius=Config.VISION_RADIUS):
        super().__init__()
        self.vision_size = 2 * vision_radius + 1

        if input_size is None:
            input_size = 2 * (self.vision_size ** 2)

        self.hidden_size = hidden_size

        # --- Encoder con dos capas lineales ---
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, 30),
            nn.ReLU()
        )

        # Inicialización He en encoder
        init.kaiming_normal_(self.encoder[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.encoder[0].bias)
        init.kaiming_normal_(self.encoder[2].weight, nonlinearity="relu")
        nn.init.zeros_(self.encoder[2].bias)

        # RNN simple
        self.rnn = nn.RNN(30, hidden_size, batch_first=True)

        # Inicialización He en RNN
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                init.kaiming_normal_(param, nonlinearity="relu")
            elif "bias" in name:
                nn.init.zeros_(param)

        # Capa de salida (movimientos)
        self.decoder = nn.Linear(hidden_size, output_size)

        # Inicialización He en decoder (sin activación → linear)
        init.kaiming_normal_(self.decoder.weight, nonlinearity="linear")
        nn.init.zeros_(self.decoder.bias)

        # Estado oculto persistente
        self.hidden_state = None



    def forward(self, terrain_vision, population_vision):
        # Convertir inputs NumPy a tensores PyTorch
        terrain_tensor = torch.from_numpy(terrain_vision).float().view(1, -1)     # (1, H*W)
        population_tensor = torch.from_numpy(population_vision).float().view(1, -1) # (1, H*W)

        # Concatenar terreno y población en un solo vector
        combined = torch.cat([terrain_tensor, population_tensor], dim=1)  # (1, 2*H*W)

        # Pasar por el encoder
        encoded = self.encoder(combined)  # (1, 64)

        # Adaptar para RNN (batch=1, seq_len=1, input_size=64)
        encoded_seq = encoded.unsqueeze(1)

        # RNN
        if self.hidden_state is None:
            rnn_out, self.hidden_state = self.rnn(encoded_seq)
        else:
            rnn_out, self.hidden_state = self.rnn(encoded_seq, self.hidden_state)

        # Decodificador final
        move_logits = self.decoder(rnn_out.squeeze(1))  # (1, output_size)
        return move_logits


    def clean_memory(self):
        self.hidden_state = None


    def mutate(self, mutation_rate=Config.MUTATION_RATE, mutation_std=Config.MUTATION_STD, mutation_clip=Config.MUTATION_CLIP):
        for param in self.parameters():
            noise_mask = (torch.rand_like(param) < mutation_rate).float()
            noise = torch.randn_like(param) * mutation_std
            param.data.add_(noise * noise_mask)
            param.data.clamp_(-mutation_clip, mutation_clip)



class MLP(nn.Module):
    def __init__(self, input_size=None, output_size=5, vision_radius=Config.VISION_RADIUS, architecture=[50, 25, 14]):
        super().__init__()
        self.vision_size = 2 * vision_radius + 1

        # Entrada: terreno y población, dos canales de (H, W)
        if input_size is None:
            input_size = 2 * (self.vision_size ** 2)

        if architecture is None:
            architecture = Config.ARCHITECTURE  # lista de tamaños ocultos

        layers = []
        prev_size = input_size

        # Crear capas ocultas dinámicamente
        for hidden_size in architecture:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        # Capa final
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)


    def forward(self, terrain_vision, population_vision):
        # Convertir inputs NumPy a tensores PyTorch
        terrain_tensor = torch.from_numpy(terrain_vision).float().view(1, -1)       # (1, H*W)
        population_tensor = torch.from_numpy(population_vision).float().view(1, -1) # (1, H*W)

        # Concatenar terreno y población
        combined = torch.cat([terrain_tensor, population_tensor], dim=1)  # (1, 2*H*W)

        # Pasar por la red
        logits = self.network(combined)  # (1, output_size)
        return logits


    def mutate(self, mutation_rate=Config.MUTATION_RATE, mutation_std=Config.MUTATION_STD, mutation_clip=Config.MUTATION_CLIP):
        for param in self.parameters():
            noise_mask = (torch.rand_like(param) < mutation_rate).float()
            noise = torch.randn_like(param) * mutation_std
            param.data.add_(noise * noise_mask)
            param.data.clamp_(-mutation_clip, mutation_clip)


import torch.nn.functional as F

class MLP_with_memory(nn.Module):
    def __init__(self, input_size=None, output_size=5, vision_radius=Config.VISION_RADIUS,
                 architecture=[60, 30, 15], memory_size=Config.MEMORY_SIZE):
        super().__init__()
        self.vision_size = 2 * vision_radius + 1
        self.output_size = output_size
        self.memory_size = memory_size

        # Entrada: terreno + población (2 canales) + memoria
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

        # 🔹 Inicialización Kaiming (He) para todas las capas lineales
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

        # Inicializar memoria (tensor de ceros con tamaño [memory_size, output_size])
        self.clean_memory()

    def clean_memory(self):
        """Resetea la memoria a ceros."""
        self.memory = torch.zeros(self.memory_size, self.output_size)

    def update_memory(self, logits):
        """Actualiza la memoria con la acción tomada (argmax -> one hot)."""
        action = torch.argmax(logits, dim=1)  # (1,)
        one_hot = F.one_hot(action, num_classes=self.output_size).float()  # (1, output_size)

        # Desplazar memoria hacia atrás e insertar nueva acción al frente
        self.memory = torch.roll(self.memory, shifts=-1, dims=0)
        self.memory[-1] = one_hot.squeeze(0)  # insertar última acción

    def forward(self, terrain_vision, population_vision):
        # Convertir entradas a tensores
        terrain_tensor = torch.from_numpy(terrain_vision).float().view(1, -1)
        population_tensor = torch.from_numpy(population_vision).float().view(1, -1)

        # Aplanar memoria en un vector
        memory_tensor = self.memory.flatten().unsqueeze(0)  # (1, memory_size*output_size)

        # Concatenar entrada total
        combined = torch.cat([terrain_tensor, population_tensor, memory_tensor], dim=1)

        # Pasar por la red
        logits = self.network(combined)  # (1, output_size)

        # Actualizar memoria con la acción seleccionada
        self.update_memory(logits)

        return logits

    def mutate(self, mutation_rate=Config.MUTATION_RATE, mutation_std=Config.MUTATION_STD,
               mutation_clip=Config.MUTATION_CLIP):
        for param in self.parameters():
            noise_mask = (torch.rand_like(param) < mutation_rate).float()
            noise = torch.randn_like(param) * mutation_std
            param.data.add_(noise * noise_mask)
            param.data.clamp_(-mutation_clip, mutation_clip)
