import copy
import random
from core.config_loader import Config
from tests.egg_hunt import EggHuntAgent

class Selection:
    @staticmethod
    def reproduce(
        agents,
        fitness_list,
        num_offspring=None,
        elitism=Config.ELITISM,
        p_survivors=Config.SURVIVAL_RATE,
        p_new=0.15
    ):
        if num_offspring is None:
            num_offspring = len(agents)

        # --- 1) Ordenar por fitness (mejores primero)
        sorted_agents = [
            agent for _, agent in sorted(
                zip(fitness_list, agents), key=lambda x: x[0], reverse=True
            )
        ]

        offspring = []

        # --- 2) Elitismo
        n_elite = max(1, int(elitism * num_offspring))
        offspring.extend(copy.deepcopy(ind) for ind in sorted_agents[:n_elite])

        # --- 3) Selección de padres (supervivientes)
        n_parents = max(1, int(p_survivors * num_offspring))
        parents = sorted_agents[:n_parents]

        # --- 4) Generar hijos mutados
        new = 0
        while len(offspring) < num_offspring:
            if random.random() < p_new:
                # --- 5) Inyección de nuevo individuo aleatorio
                offspring.append(EggHuntAgent())
                new += 1
            else:
                parent = random.choice(parents)
                child = copy.deepcopy(parent)
                child.nn.mutate(
                    Config.MUTATION_RATE,
                    Config.MUTATION_STD,
                    Config.MUTATION_CLIP
                )
                child.mutate_color()
                offspring.append(child)

        # --- Ajuste final por si nos pasamos
        return offspring[:num_offspring]
