import pygame
import os
from screeninfo import get_monitors

from graphics.render import DisplayBorders, AgentRenderer, TerrainRenderer, UIButton, NotInteractionRenderer, SimulationHUD
from graphics.camera import Camera

from core.config_loader import Config

####################################################################
monitores = get_monitors()
pantalla_objetivo = monitores[1] # Primary screen
x = pantalla_objetivo.x
y = pantalla_objetivo.y
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
####################################################################

class DisplayManager:
    def __init__(self):

        pygame.init()
        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.FULLSCREEN)
        self.clock = pygame.time.Clock()
        self.display = DisplayBorders(Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT)
        self.camera = Camera(self.display.inner_rect, Config.CELL_SIZE)
        self.close_button = UIButton()
        self.hud = SimulationHUD()

    def draw_frame(self, terrain, agents, not_interacting_objects, n_gen, n_sim):
        self.display.draw_borders(self.screen)
        TerrainRenderer.draw_terrain(self.screen, self.display, self.camera, terrain)
        NotInteractionRenderer.draw_squares(self.screen, self.camera, Config.CELL_SIZE, not_interacting_objects)
        AgentRenderer.draw_agents(self.screen, self.camera, agents, Config.CELL_SIZE)
        self.hud.draw(n_gen, n_sim)
        self.close_button.draw(self.screen)

    def update(self):
        pygame.display.flip()
        self.clock.tick(Config.FPS)

    def handle_input(self):
        keys = pygame.key.get_pressed()

        dx, dy = 0, 0
        if keys[pygame.K_LEFT]:
            dx = -1
        if keys[pygame.K_RIGHT]:
            dx = 1
        if keys[pygame.K_UP]:
            dy = -1
        if keys[pygame.K_DOWN]:
            dy = 1

        if dx or dy:
            self.camera.move(dx, dy)

