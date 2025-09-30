from abc import ABC, abstractmethod


class BaseBrain(ABC):
    def __init__(self, brain_config):
        self.brain_config = brain_config

    @abstractmethod
    def initialize_network(self):
        """"Initialize the weights and/or connections of the neural network.
        """
        pass

    @abstractmethod
    def forward(self, *inputs):
        """Forward pass of the neural network.
        """
        pass

    @abstractmethod
    def mutate(self):
        """Mutate the neural network.
        """
        pass

    @abstractmethod
    def save(self, filepath):
        """Save the neural network to a file.
        """
        pass

    @abstractmethod
    def load(self, filepath):
        """Load the neural network from a file.
        """
        pass


class BrainWithMemory(BaseBrain):
    def __init__(self, brain_config):
        super().__init__(brain_config)

    @abstractmethod
    def clean_memory(self):
        """"Clean the memory of the neural network.
        """
        pass

    @abstractmethod
    def update_memory(self):
        """Update the memory of the neural network.
        """
        pass