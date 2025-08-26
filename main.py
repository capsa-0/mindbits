import pygame
from tests.egg_hunt import EggHuntTest
from graphics.display import DisplayManager
from core.selection import Selection
import torch

from core.config_loader import Config


torch.set_grad_enabled(False)

def main():
    display = DisplayManager()


    n_gen = 0
    running = True
    simulation = EggHuntTest()

    while running:
        fitness_list = [0] * Config.POP_SIZE

        for n_sim in range(1, Config.N_SIMULATIONS + 1):
            simulation.reset()
            sim_running = True

            while sim_running:
                for event in pygame.event.get():
                    if display.close_button.is_clicked(event):
                        pygame.quit()
                        return

                display.handle_input()   
                display.draw_frame(
                    simulation.terrain_map,
                    simulation.agents_population,
                    [
                        (simulation.egg_position, simulation.egg_color),
                        (simulation.start_position, simulation.start_color)
                    ],
                    n_gen, n_sim
                )


                pygame.display.flip()
                display.clock.tick(Config.MAX_FPS)

                sim_running = simulation.step()

            fitness_list = [x + y for x, y in zip(fitness_list, simulation.get_fitness_list())]

        fitness_list_average = [x / Config.N_SIMULATIONS for x in fitness_list]

        avg_fitness = sum(fitness_list_average) / len(fitness_list_average)
        display.hud.push_gen_fitness(avg_fitness)

        simulation.agents_population = Selection.reproduce(
            simulation.agents_population,
            fitness_list_average
        )

        n_gen += 1

    pygame.quit()


if __name__ == "__main__":
    main()
