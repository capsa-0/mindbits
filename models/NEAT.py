import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config_loader import Config

class NEATWithMemory(nn.Module):

    def __init__(
        self,
        input_size=None,
        output_size=5,
        vision_radius=Config.VISION_RADIUS,
        memory_size=Config.MEMORY_SIZE,
        init_connect_prob: float = 0.2,
        include_bias: bool = True,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.vision_size = 2 * vision_radius + 1
        self.output_size = output_size
        self.memory_size = memory_size
        self.include_bias = include_bias
        self.device = device
        self.dtype = dtype

        if input_size is None:
            input_size = 2 * (self.vision_size ** 2)
        memory_input_size = output_size * memory_size
        self.sensor_input_size = input_size
        self.total_input_size = input_size + memory_input_size 

        self.num_input_nodes = self.total_input_size
        if include_bias:
            self.num_input_nodes += 1

        self.node_types: Dict[int, str] = {}
        for i in range(self.num_input_nodes):
            self.node_types[i] = "input"
        next_id = self.num_input_nodes

        self.output_ids = list(range(next_id, next_id + output_size))
        for oid in self.output_ids:
            self.node_types[oid] = "output"
        self.next_node_id = self.output_ids[-1] + 1

        self.connections: List[Dict] = []
        self._minimal_init(init_connect_prob)


        self.clean_memory()

    def clean_memory(self):
        self.memory = torch.zeros(self.memory_size, self.output_size,
                                  device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def update_memory(self, logits: torch.Tensor):
        action = torch.argmax(logits, dim=1)
        one_hot = F.one_hot(action, num_classes=self.output_size).to(self.dtype)
        self.memory = torch.roll(self.memory, shifts=-1, dims=0)
        self.memory[-1] = one_hot.squeeze(0)

    def forward(self, terrain_vision, population_vision):
        terrain_tensor = torch.from_numpy(terrain_vision).to(self.device, self.dtype).view(1, -1)
        population_tensor = torch.from_numpy(population_vision).to(self.device, self.dtype).view(1, -1)
        memory_tensor = self.memory.flatten().unsqueeze(0)

        inputs = torch.cat([terrain_tensor, population_tensor, memory_tensor], dim=1)
        if self.include_bias:
            bias = torch.ones((1, 1), device=self.device, dtype=self.dtype)
            inputs = torch.cat([inputs, bias], dim=1)

        assert inputs.shape[1] == self.num_input_nodes, \
            f"Desajuste: inputs={inputs.shape[1]}, num_input_nodes={self.num_input_nodes}"

        activations: Dict[int, torch.Tensor] = {}
        for i in range(self.num_input_nodes):
            activations[i] = inputs[:, i]

        order = self._topological_order()

        def act_hidden(x): return torch.relu(x)

        incoming: Dict[int, List[Tuple[int, float]]] = {}
        for c in self.connections:
            if c["enabled"]:
                incoming.setdefault(c["out"], []).append((c["in"], c["w"]))

        for nid in order:
            ntype = self.node_types[nid]
            if ntype == "input":
                continue
            s = 0.0
            for src, w in incoming.get(nid, []):
                s += activations[src] * w
            if ntype == "hidden":
                activations[nid] = act_hidden(s)
            elif ntype == "output":
                activations[nid] = s  

        out = torch.stack([activations.get(oid, torch.zeros(1, device=self.device, dtype=self.dtype))
                           for oid in self.output_ids], dim=1)

        self.update_memory(out)
        return out

    @torch.no_grad()
    def mutate(self, p_perturb: float = 0.8, p_add_conn: float = 0.3, p_add_node: float = 0.05):
        if random.random() < p_perturb and self.connections:
            for c in self.connections:
                if c["enabled"]:
                    c["w"] += random.gauss(0, 0.5)
        if random.random() < p_add_conn:
            self._mut_add_connection(1.0)
        if random.random() < p_add_node:
            self._mut_add_node()


    def _minimal_init(self, init_connect_prob: float):
        made_any = False
        for o in self.output_ids:
            for i in range(self.num_input_nodes):
                if random.random() < init_connect_prob:
                    self.connections.append({"in": i, "out": o, "w": random.gauss(0, 1), "enabled": True})
                    made_any = True
        if not made_any:
            i = random.randrange(self.num_input_nodes)
            o = random.choice(self.output_ids)
            self.connections.append({"in": i, "out": o, "w": random.gauss(0, 1), "enabled": True})
        self._enforce_acyclicity()


    def _edges_enabled(self):
        return [(c["in"], c["out"]) for c in self.connections if c["enabled"]]

    def _build_graph(self):
        edges = self._edges_enabled()
        adj_out, indeg = {}, {nid: 0 for nid in self.node_types}
        for u, v in edges:
            adj_out.setdefault(u, []).append(v)
            indeg[v] = indeg.get(v, 0) + 1
        return adj_out, indeg

    def _topological_order(self):
        adj_out, indeg = self._build_graph()
        order, seen = [], set()
        from collections import deque
        q = deque([nid for nid, t in self.node_types.items() if t == "input"])
        while q:
            u = q.popleft()
            if u in seen:
                continue
            seen.add(u)
            order.append(u)
            for v in adj_out.get(u, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        for nid in self.node_types:
            if nid not in seen:
                order.append(nid)
        return order

    def _is_acyclic(self):
        return True 

    def _would_create_cycle(self, src, dst):
        return False

    def _enforce_acyclicity(self):
        pass

    def _mut_add_connection(self, new_weight_std: float):
        i = random.randrange(self.num_input_nodes)
        o = random.choice(self.output_ids)
        self.connections.append({"in": i, "out": o, "w": random.gauss(0, new_weight_std), "enabled": True})

    def _mut_add_node(self):
        enabled_edges = [c for c in self.connections if c["enabled"]]
        if not enabled_edges:
            return
        conn = random.choice(enabled_edges)
        conn["enabled"] = False
        new_id = self.next_node_id
        self.next_node_id += 1
        self.node_types[new_id] = "hidden"
        self.connections.append({"in": conn["in"], "out": new_id, "w": 1.0, "enabled": True})
        self.connections.append({"in": new_id, "out": conn["out"], "w": conn["w"], "enabled": True})

    def summary(self):
        num_hidden = sum(1 for t in self.node_types.values() if t == "hidden")
        num_enabled = sum(1 for c in self.connections if c["enabled"])
        return f"Inputs={self.num_input_nodes}, Hidden={num_hidden}, Outputs={len(self.output_ids)}, Conns={len(self.connections)} (enabled={num_enabled})"
