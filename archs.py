import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import Config


def to_one_hot(input, encoding_size):
    # Versión pre-asignada de memoria (óptima para visiones grandes)

    result = [0] * (len(input) * encoding_size)
    
    for i, cell in enumerate(input):
        if 0 <= cell < encoding_size:
            result[i * encoding_size + int(cell)] = 1
    return result



class FeedForward(nn.Module):
    def __init__(self, input_size, architecture, ws=None, bs=None):
        super(FeedForward, self).__init__()
        self.layers = nn.ModuleList()
        prev_size = input_size
        # Construir capas
        for layer_size in architecture + [9]:
            layer = nn.Linear(prev_size, layer_size)
            
            # Inicialización de He (Kaiming) para pesos
            nn.init.kaiming_normal_(layer.weight, 
                                    mode='fan_in',  # Opciones: 'fan_in', 'fan_out'
                                    nonlinearity='relu')  # Ajustar según tu función de activación
            
            # Inicializar sesgos a cero (opcional, pero común)
            nn.init.zeros_(layer.bias)
            
            self.layers.append(layer)
            prev_size = layer_size

            # Cargar pesos si se proporcionan
            if ws is not None and bs is not None:
                self.load_weights(ws, bs)

    def load_weights(self, ws, bs):
        """Carga pesos desde NumPy arrays, ajustando dimensiones para PyTorch."""
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                # PyTorch espera pesos de shape (out_features, in_features)
                layer.weight.data = torch.tensor(ws[i].T, dtype=torch.float32)
                layer.bias.data = torch.tensor(bs[i], dtype=torch.float32)

    def mutate(
        self,
        mutation_rate=Config.MUTATION_RATE,
        mutation_scale=Config.MUTATION_SCALE,
        zero_rate=Config.DROP_RATE 
    ):

        with torch.no_grad():
            for layer in self.layers:
                # Mutar pesos
                mask = (torch.rand_like(layer.weight) < mutation_rate).float()
                noise = torch.randn_like(layer.weight) * mutation_scale
                layer.weight += mask * noise

                # Anular pesos a cero
                zero_mask = (torch.rand_like(layer.weight) < zero_rate)
                layer.weight[zero_mask] = 0.0

                # Mutar biases
                mask = (torch.rand_like(layer.bias) < mutation_rate).float()
                noise = torch.randn_like(layer.bias) * mutation_scale
                layer.bias += mask * noise

                # Anular biases a cero
                zero_mask = (torch.rand_like(layer.bias) < zero_rate)
                layer.bias[zero_mask] = 0.0



    def forward(self, input):

        if Config.ENCODING == 'one_hot':
            input = input

        #elif Config.ENCODING == 'dense': 
         #   input = list(x_env.flatten()) + list(x_memory)# + [energy/100]
        
        x = torch.as_tensor(input, dtype=torch.float32).view(1, -1)  # Forma (1, 1)
        # Forward a través de las capas
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        
        output = self.layers[-1](x)

        return np.argmax(output.detach().numpy())



import math
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class NeuralNetworkNEAT:
    def __init__(self, input_size = Config.INPUT_SIZE, arch = None):
        self.input_size = input_size
        self.output_size = 9
        self.nodes = (
            [{"id": i, "type": "input"} for i in range(input_size)]
            + [{"id": input_size + j, "type": "output"} for j in range(self.output_size)]
        )
        self.connections = []
        self.innovation_counter = 0
        self.node_counter = input_size + self.output_size
        self.topological_order = []

        # ==== Conexiones iniciales (10 aleatorias entrada->salida) ====
        input_nodes = range(self.input_size)
        output_nodes = range(self.input_size, self.input_size + self.output_size)
        
        # Generar todas combinaciones posibles de conexiones entrada-salida
        all_possible_connections = [
            (in_node, out_node)
            for in_node in input_nodes
            for out_node in output_nodes
        ]
        
        # Seleccionar 10 conexiones únicas (sin repetición)
        random.shuffle(all_possible_connections)
        selected_connections = all_possible_connections[:10]  # Toma las primeras 10
        
        for in_node, out_node in selected_connections:
            self.connections.append({
                "in_node": in_node,
                "out_node": out_node,
                "weight": random.uniform(-1, 1),
                "enabled": True,
                "innovation": self.innovation_counter
            })
            self.innovation_counter += 1

        # Probabilidades de mutación (ajustables)
        self.p_add_conn = 0.3
        self.p_add_node = 0.1
        self.p_adjust_weight = 0.7
        self.p_toggle_enabled = 0.1
        self.p_remove_conn = 0.05
        self.weight_perturb_prob = 0.9  # Probabilidad de perturbar vs reemplazar peso

        self.update_topological_order()
        self.plot_network()
    

    def forward(self, x_env, x_memory, energy):
        inputs = x_env.flatten().tolist() + list(x_memory) + [energy / 100]

        
        activations = {node["id"]: 0.0 for node in self.nodes}
        
        # Asignar entradas
        for i in range(self.input_size):
            activations[i] = inputs[i]
    
        
        # Procesar en orden topológico

        for node_id in self.topological_order:
            node = next(n for n in self.nodes if n["id"] == node_id)
            if node["type"] == "input":
                continue
            
            # Calcular suma ponderada
            total = 0.0

            for conn in self.connections:
                if conn["out_node"] == node_id and conn["enabled"]:
                    total += activations[conn["in_node"]] * conn["weight"]

            activations[node_id] = self.sigmoid(total)

        
        # Extraer salidas
        output_ids = [n["id"] for n in self.nodes if n["type"] == "output"]
        output = [activations[id] for id in output_ids]
        action = np.argmax(output)

        return action

    def mutate(self):
        if random.random() < self.p_add_conn:
            self._mutate_add_connection()
        if random.random() < self.p_add_node:
            self._mutate_add_node()
        if random.random() < self.p_adjust_weight:
            self._mutate_adjust_weights()
        if random.random() < self.p_toggle_enabled:
            self._mutate_toggle_connection()
        if random.random() < self.p_remove_conn:
            self._mutate_remove_connection()
        self.update_topological_order()

    def _mutate_add_connection(self):
        attempts = 100
        for _ in range(attempts):
            a = random.choice([n["id"] for n in self.nodes if n["type"] in ["input", "hidden"]])
            b = random.choice([n["id"] for n in self.nodes if n["type"] in ["hidden", "output"]])
            if a == b or not self._can_add_connection(a, b):
                continue
            self.connections.append({
                "in_node": a,
                "out_node": b,
                "weight": random.uniform(-1, 1),
                "enabled": True,
                "innovation": self.innovation_counter
            })
            self.innovation_counter += 1
            return

    def _can_add_connection(self, a, b):
        # Verificar si existe conexión previa
        for conn in self.connections:
            if conn["in_node"] == a and conn["out_node"] == b:
                return False
        # Detección de ciclos (BFS)
        visited = set()
        queue = deque([b])
        while queue:
            current = queue.popleft()
            if current == a:
                return False  # Hay ciclo
            visited.add(current)
            for conn in self.connections:
                if conn["out_node"] == current and conn["in_node"] not in visited:
                    queue.append(conn["in_node"])
        return True

    def _mutate_add_node(self):
        enabled_conns = [c for c in self.connections if c["enabled"]]
        if not enabled_conns:
            return
        conn = random.choice(enabled_conns)
        conn["enabled"] = False
        new_node_id = self.node_counter
        self.node_counter += 1
        self.nodes.append({"id": new_node_id, "type": "hidden"})
        # Añadir nuevas conexiones
        self.connections.append({
            "in_node": conn["in_node"],
            "out_node": new_node_id,
            "weight": 1.0,
            "enabled": True,
            "innovation": self.innovation_counter
        })
        self.innovation_counter += 1
        self.connections.append({
            "in_node": new_node_id,
            "out_node": conn["out_node"],
            "weight": conn["weight"],
            "enabled": True,
            "innovation": self.innovation_counter
        })
        self.innovation_counter += 1

    def _mutate_adjust_weights(self):
        for conn in self.connections:
            if random.random() < 0.8:  # Probabilidad de ajustar este peso
                if random.random() < self.weight_perturb_prob:
                    conn["weight"] += random.uniform(-0.5, 0.5)
                else:
                    conn["weight"] = random.uniform(-1, 1)
                conn["weight"] = max(min(conn["weight"], 1.0), -1.0)

    def _mutate_toggle_connection(self):
        if self.connections:
            conn = random.choice(self.connections)
            conn["enabled"] = not conn["enabled"]

    def _mutate_remove_connection(self):
        if self.connections:
            conn = random.choice(self.connections)
            self.connections.remove(conn)

    def update_topological_order(self):
        adj = {n["id"]: [] for n in self.nodes}
        in_degree = {n["id"]: 0 for n in self.nodes}
        for conn in self.connections:
            if conn["enabled"]:
                adj[conn["in_node"]].append(conn["out_node"])
                in_degree[conn["out_node"]] += 1
        queue = deque([node_id for node_id in in_degree if in_degree[node_id] == 0])
        top_order = []
        while queue:
            node = queue.popleft()
            top_order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        self.topological_order = top_order

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    @property
    def size(self):
        return len(self.connections)
    
    def plot_network(self, show_weights=True, innovation_numbers=False):
        plt.figure(figsize=(12, 8))
        G = nx.DiGraph()
        
        # Agregar nodos con atributos
        node_colors = []
        for node in self.nodes:
            G.add_node(node["id"])
            if node["type"] == "input":
                node_colors.append("lightgreen")
            elif node["type"] == "output":
                node_colors.append("salmon")
            else:
                node_colors.append("skyblue")
        
        # Agregar conexiones
        edge_colors = []
        edge_widths = []
        labels = {}
        for conn in self.connections:
            if conn["enabled"]:
                G.add_edge(conn["in_node"], conn["out_node"])
                # Color por signo del peso
                edge_colors.append("red" if conn["weight"] < 0 else "blue")
                # Ancho proporcional al peso absoluto
                edge_widths.append(abs(conn["weight"]) * 2 + 0.5)
                
                # Etiquetas
                if innovation_numbers:
                    labels[(conn["in_node"], conn["out_node"])] = f"{conn['innovation']}"
                elif show_weights:
                    labels[(conn["in_node"], conn["out_node"])] = f"{conn['weight']:.2f}"

        # Posicionamiento en capas
        pos = {}
        input_nodes = [n["id"] for n in self.nodes if n["type"] == "input"]
        hidden_nodes = [n["id"] for n in self.nodes if n["type"] == "hidden"]
        output_nodes = [n["id"] for n in self.nodes if n["type"] == "output"]
        
        # Inputs a la izquierda
        for i, node_id in enumerate(input_nodes):
            pos[node_id] = (0, i/len(input_nodes))
            
        # Outputs a la derecha
        for i, node_id in enumerate(output_nodes):
            pos[node_id] = (2, i/len(output_nodes))
            
        # Hidden en el medio (organizados verticalmente)
        for i, node_id in enumerate(hidden_nodes):
            pos[node_id] = (1, (i+1)/(len(hidden_nodes)+1))

        # Dibujar
        nx.draw(
            G, pos,
            node_color=node_colors,
            node_size=800,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=15,
            with_labels=True,
            font_weight='bold'
        )
        
        # Dibujar etiquetas de conexiones
        if labels:
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=labels,
                font_color='black',
                label_pos=0.75
            )
            
        # Leyenda
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Input', markerfacecolor='lightgreen', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Hidden', markerfacecolor='skyblue', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Output', markerfacecolor='salmon', markersize=10),
            plt.Line2D([0], [0], color='blue', lw=2, label='Peso positivo'),
            plt.Line2D([0], [0], color='red', lw=2, label='Peso negativo')
        ]
        plt.legend(handles=legend_elements, loc='best')
        
        plt.title(f"Red NEAT (Tamaño: {self.size})")
        plt.show()