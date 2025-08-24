import copy

from core.config_loader import Config

class Selection:
    @staticmethod


    def reproduce(agents, fitness_list, num_offspring=None, elitism=Config.ELITISM, p_survivors=Config.SURVIVAL_RATE):
        if num_offspring is None:
            num_offspring = len(agents)

        sorted_agents = [agent for _, agent in sorted(zip(fitness_list, agents), key=lambda x: x[0], reverse=True)]


        n_elite = max(1, int(len(sorted_agents) * elitism))
        elite = [copy.deepcopy(ind) for ind in sorted_agents[:n_elite]]

        n_survivors = int(p_survivors * num_offspring)
        repetitions = max(1, int(1 / p_survivors))

        parents = sorted_agents[:n_survivors] * repetitions

        offspring = elite  

        for parent in parents:
            if len(offspring) >= num_offspring:
                break

            child = copy.deepcopy(parent)
            child.nn.mutate(
                Config.MUTATION_RATE,
                Config.MUTATION_STD,
                Config.MUTATION_CLIP
            )
            child.mutate_color()
            offspring.append(child)

        return offspring


