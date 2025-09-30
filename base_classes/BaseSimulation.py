class BaseSimulation:
    def __init__(self, task_config, population):
        self.task_config = task_config
        self.population = population
        self.map = None


    @abstractmethod
    def reset(self, mode="generation"):
        """
        Reinicia el estado de la simulación.
        mode="generation" -> nueva generación: cambia mapa, reinicia agentes.
        mode="run"        -> nuevo run dentro de la misma generación: 
                             mantiene mapa, cambia posiciones iniciales.
        """
        pass

    @abstractmethod
    def step(self):
        """
        Ejecuta un paso de la simulación:
        - actualiza el estado del entorno,
        - obtiene las observaciones de los agentes,
        - aplica las acciones de los agentes.
        Devuelve False si el episodio/run ha terminado, True en caso contrario.
        """
        pass

    @abstractmethod
    def compute_fitness(self, agent):
        """
        Devuelve el fitness de un agente individual.
        """
        pass

    def compute_population_fitness(self):
        """
        Fitness para toda la población (usando compute_fitness de cada agente).
        """
        return [self.compute_fitness(agent) for agent in self.agents]

    def is_done(self):
        """
        Condición de finalización del episodio/run:
        - todos los agentes murieron, o
        - se alcanzó el máximo de pasos.
        """
        return self.step_count >= self.config.max_steps or all(not a.alive for a in self.agents)
