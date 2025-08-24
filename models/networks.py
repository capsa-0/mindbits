from models.SimpleRNN import SimpleRNN
from models.MLP_with_memory import MLP_with_memory
from models.NEAT import NEATWithMemory

NETWORKS = {
    "SimpleRNN": SimpleRNN,
    "MLP_with_memory": MLP_with_memory,
    "NEATWithMemory": NEATWithMemory
}
