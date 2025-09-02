import pygame
from tests.egg_hunt import EggHuntTest
from graphics.display import DisplayManager
from core.selection import Selection
import torch
from core.config_loader import Config

torch.set_grad_enabled(False)


def run_generations(initial_population, n_generations):
    """
    Corre la simulación durante n_generaciones, partiendo de initial_population.
    Devuelve la población final y lista de fitness promedio por generación.
    """
    display = DisplayManager()
    population = initial_population
    fitness_history = []

    for gen in range(n_generations):
        fitness_list = [0] * len(population)

        for n_sim in range(1, Config.N_SIMULATIONS + 1):
            simulation = EggHuntTest()
            simulation.agents_population = population.copy()
            simulation.reset()

            sim_running = True
            while sim_running:
                for event in pygame.event.get():
                    if display.close_button.is_clicked(event):
                        pygame.quit()
                        return population, fitness_history

                display.handle_input()
                display.draw_frame(
                    simulation.terrain_map,
                    simulation.agents_population,
                    [
                        (simulation.egg_position, simulation.egg_color),
                        (simulation.start_position, simulation.start_color)
                    ],
                    gen, n_sim
                )

                pygame.display.flip()
                display.clock.tick(Config.MAX_FPS)

                sim_running = simulation.step()

            fitness_list = [x + y for x, y in zip(fitness_list, simulation.get_fitness_list())]

        fitness_list_average = [x / Config.N_SIMULATIONS for x in fitness_list]
        avg_fitness = sum(fitness_list_average) / len(fitness_list_average)
        fitness_history.append(avg_fitness)
        display.hud.push_gen_fitness(avg_fitness)

        population = Selection.reproduce(population, fitness_list_average)

    pygame.quit()
    return population, fitness_history



def run_generations_headless(initial_population, n_generations):
    """
    Corre la simulación durante n_generations sin visualización.
    Devuelve la población final y lista de fitness promedio por generación.
    """
    population = initial_population
    fitness_history = []

    for gen in range(n_generations):
        print(f"\n=== Generation {gen+1} ===")
        fitness_list = [0] * len(population)

        for n_sim in range(1, Config.N_SIMULATIONS + 1):
            simulation = EggHuntTest()
            simulation.agents_population = population.copy()
            simulation.reset()

            sim_running = True
            step_count = 0
            while sim_running:
                sim_running = simulation.step()
                step_count += 1

            print(f"Simulation {n_sim}/{Config.N_SIMULATIONS}")
            fitness_list = [x + y for x, y in zip(fitness_list, simulation.get_fitness_list())]

        # Promedio de fitness por agente
        fitness_list_average = [x / Config.N_SIMULATIONS for x in fitness_list]
        avg_fitness = sum(fitness_list_average) / len(fitness_list_average)
        fitness_history.append(avg_fitness)

        print(f"Mean fitness {gen+1}: {avg_fitness:.4f}")

        # Reproduce para la siguiente generación
        population = Selection.reproduce(population, fitness_list_average)

    return population, fitness_history




def main():
    # Inicializa población
    simulation = EggHuntTest()
    population = simulation.agents_population

    population, fitness_history = run_generations_headless(population, n_generations=20) 
    population, fitness_history = run_generations(population, n_generations=5000) 



if __name__ == "__main__":
    main()
