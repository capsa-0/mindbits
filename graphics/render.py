import pygame
import numpy as np
from core.config_loader import Config

class DisplayBorders:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.border_size = Config.EDGE_SIZE
        self.inner_rect = pygame.Rect(
            self.border_size,
            self.border_size,
            screen_width - 2*self.border_size,
            screen_height - 2*self.border_size
        )

    def draw_borders(self, surface, color=(0, 0, 0)):
        borders = [
            (0, 0, self.screen_width, self.border_size),
            (0, self.screen_height - self.border_size,
             self.screen_width, self.border_size),
            (0, 0, self.border_size, self.screen_height),
            (self.screen_width - self.border_size, 0,
             self.border_size, self.screen_height)
        ]
        for border in borders:
            pygame.draw.rect(surface, color, border)



class AgentRenderer:
    @staticmethod
    def draw_agents(surface, camera, agents, cell_size):
        start_x = int(camera.offset_x)
        start_y = int(camera.offset_y)
        end_x = start_x + (camera.viewport_rect.width // cell_size) + 1
        end_y = start_y + (camera.viewport_rect.height // cell_size) + 1

        agent_surface = pygame.Surface(
            (camera.viewport_rect.width, camera.viewport_rect.height),
            pygame.SRCALPHA
        )

        for agent in agents:
            if agent.alive and (start_x <= agent.x < end_x) and (start_y <= agent.y < end_y):
                screen_x, screen_y = camera.world_to_screen(agent.x, agent.y)
                x_rel = screen_x - camera.viewport_rect.x
                y_rel = screen_y - camera.viewport_rect.y

                center_x = x_rel + cell_size // 2
                center_y = y_rel + cell_size // 2
                radius = (cell_size - 1) // 2

                pygame.draw.circle(agent_surface, agent.color, (center_x, center_y), radius)
                pygame.draw.circle(agent_surface, (0, 0, 0), (center_x, center_y), radius, 1)

        surface.blit(agent_surface, camera.viewport_rect.topleft)
    


class NotInteractionRenderer:
    @staticmethod
    def draw_squares(surface, camera, cell_size, squares):

        start_x = int(camera.offset_x)
        start_y = int(camera.offset_y)
        end_x = start_x + (camera.viewport_rect.width // cell_size) + 1
        end_y = start_y + (camera.viewport_rect.height // cell_size) + 1


        square_surface = pygame.Surface(
            (camera.viewport_rect.width, camera.viewport_rect.height),
            pygame.SRCALPHA
        )

        for position, color in squares:
            x, y = position

            if start_x <= x < end_x and start_y <= y < end_y:
                screen_x, screen_y = camera.world_to_screen(x, y)
                x_rel = screen_x - camera.viewport_rect.x
                y_rel = screen_y - camera.viewport_rect.y

                rect = pygame.Rect(x_rel, y_rel, cell_size, cell_size)
                pygame.draw.rect(square_surface, color, rect)
                pygame.draw.rect(square_surface, (0, 0, 0), rect, 1)


        surface.blit(square_surface, camera.viewport_rect.topleft)



class TerrainRenderer:
    @staticmethod
    def draw_terrain(screen, display, camera, terrain):
        background_color = Config.FONDO
        start_x = int(camera.offset_x)
        start_y = int(camera.offset_y)
        end_x = start_x + display.inner_rect.width // Config.CELL_SIZE + 1
        end_y = start_y + display.inner_rect.height // Config.CELL_SIZE + 1

        screen.set_clip(display.inner_rect)
        for y in range(start_y, min(end_y, Config.MAP_HEIGHT)):
            for x in range(start_x, min(end_x, Config.MAP_WIDTH)):
                screen_x, screen_y = camera.world_to_screen(x, y)

                if terrain[y, x] == 1:
                    pygame.draw.rect(screen, Config.OBSTACLE_COLOR,
                                     (screen_x, screen_y, Config.CELL_SIZE, Config.CELL_SIZE))
                if terrain[y, x] == 0:
                    pygame.draw.rect(screen, background_color,
                                     (screen_x, screen_y, Config.CELL_SIZE, Config.CELL_SIZE))
        screen.set_clip(None)



class UIButton:
    def __init__(self, rect=(Config.SCREEN_WIDTH-Config.PAD-15, Config.PAD, 15, 15), color=(200,50,50)):
        self.rect = pygame.Rect(rect)
        self.color = color

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        return False



class SimulationHUD:
    def __init__(self, pos=(800, 43), size=(500, 400), font_name="consolas", font_size=16, padding=10, alpha=190):
        self.x, self.y = pos
        self.w, self.h = size
        self.padding = padding
        self.bg_color = (18, 18, 20)
        self.border_color = (80, 80, 90)
        self.text_color = (230, 230, 235)
        self.accent = (120, 180, 255)
        self.grid = (60, 60, 70)

        self.chart_h = 180
        info_h = self.h - self.chart_h - 2 * self.padding  

        self.info_panel = pygame.Surface((self.w, info_h))
        self.info_panel.set_alpha(alpha)

        self.chart_panel = pygame.Surface((self.w - 2*self.padding, self.chart_h))
        self.chart_panel.set_alpha(alpha)
        self.chart_rect = self.chart_panel.get_rect(topleft=(self.x + self.padding,
                                                             self.y + info_h + self.padding))

        try:
            self.font = pygame.font.SysFont(font_name, font_size)
            self.font_bold = pygame.font.SysFont(font_name, font_size, bold=True)
            self.font_small = pygame.font.SysFont(font_name, max(12, font_size - 2))
        except Exception:
            self.font = pygame.font.Font(None, font_size)
            self.font_bold = pygame.font.Font(None, font_size)
            self.font_small = pygame.font.Font(None, max(12, font_size - 2))

        self.fitness_history = []

    def push_gen_fitness(self, avg_fitness: float):
        if isinstance(avg_fitness, (int, float)):
            self.fitness_history.append(float(avg_fitness))

    def draw(self, n_gen: int, n_sim: int | None = None):
        surface = pygame.display.get_surface()
        if surface is None:
            return

        self.info_panel.fill(self.bg_color)
        self._blit_text("EGG HUNT (STATIC)", (self.padding, self.padding), bold=True, panel=self.info_panel)
        self._blit_text(f"GEN: {n_gen}", (self.padding, self.padding + 24), panel=self.info_panel)
        self._blit_text(f"SIM: {n_sim}", (self.padding + 160, self.padding + 24), panel=self.info_panel)

        self._draw_config_block((self.padding, 60), panel=self.info_panel)
        surface.blit(self.info_panel, (self.x, self.y))

        chart_y = self.y + self.info_panel.get_height() + self.padding
        self.chart_rect.topleft = (self.x + self.padding, chart_y)
        self._draw_chart(surface, self.chart_rect, self.fitness_history)

        pygame.draw.rect(surface, self.border_color, (self.x, self.y, self.w, self.h), 1)


    def _blit_text(self, text, pos, bold=False, small=False, color=None, panel=None):
        color = color or self.text_color
        font = self.font_bold if bold else (self.font_small if small else self.font)
        surf = font.render(str(text), True, color)
        target = panel if panel else self.info_panel
        target.blit(surf, pos)


    def _draw_chart(self, surface: pygame.Surface, rect: pygame.Rect, data: list[float]):
        self.chart_panel.fill((28, 28, 32))

        top_pad, bottom_pad = 30, 20
        chart_h = rect.h - top_pad - bottom_pad

        pygame.draw.line(self.chart_panel, self.text_color, (40, top_pad), (40, rect.h - bottom_pad), 2) 
        pygame.draw.line(self.chart_panel, self.text_color, (40, rect.h - bottom_pad), (rect.w, rect.h - bottom_pad), 2)  

        self._blit_text("Fitness", (rect.w // 2 - 50, 5), bold=True, panel=self.chart_panel)

        if len(data) < 2:
            self._blit_text("Fitness plot (min. 2 generations)", (50, 40),
                            small=True, color=(180, 180, 190), panel=self.chart_panel)
            surface.blit(self.chart_panel, rect.topleft)
            return

        vmin, vmax = min(data), max(data)
        rng = vmax - vmin if vmax != vmin else 1.0

        last_val = data[-1]
        y_last = top_pad + int((vmax - last_val) / rng * chart_h)
        pygame.draw.line(self.chart_panel, (200, 200, 200), (35, y_last), (40, y_last)) 
        self._blit_text(f"{last_val:.2f}", (0, y_last - 8), small=True, panel=self.chart_panel)

        step_x = (rect.w - 50) / max(1, len(data) - 1)
        points = []
        for i, v in enumerate(data):
            x = 40 + int(i * step_x)
            y = top_pad + int((vmax - v) / rng * chart_h)
            points.append((x, y))
        pygame.draw.lines(self.chart_panel, self.accent, False, points, 2)


        max_ticks = 5
        n_points = len(data)
        tick_positions = np.linspace(0, n_points - 1, min(max_ticks, n_points), dtype=int)

        for i in tick_positions:
            x = 40 + int(i * step_x)
            pygame.draw.line(self.chart_panel, (80, 80, 90), (x, rect.h - bottom_pad), (x, rect.h - bottom_pad + 5))
            self._blit_text(f"{i}", (x - 8, rect.h - bottom_pad + 8), small=True, panel=self.chart_panel)

        surface.blit(self.chart_panel, rect.topleft)


    def _draw_config_block(self, origin: tuple[int, int], panel=None):
        panel = panel or self.info_panel
        ox, oy = origin
        self._blit_text("Hiper-parameters (Config)", (ox, oy), bold=True, panel=panel)

        keys = ["NETWORK_TYPE",
                "VISION_RADIUS", "MEMORY_SIZE",
                "POP_SIZE", "N_SIMULATIONS",
                "MAX_STEPS",
                "ELITISM", "SURVIVAL_RATE",
                "MUTATION_RATE", "MUTATION_STD",
                "MAP_WIDTH", "MAP_HEIGHT"
        ]

        y = oy + 22
        col2_x = ox + 260
        for idx, k in enumerate(keys):
            exists = hasattr(Config, k)
            name_text = f"{k}:"
            val_text = getattr(Config, k) if exists else "—"

            x = ox if idx < (len(keys) // 2 + len(keys) % 2) else col2_x
            if idx == (len(keys) // 2 + len(keys) % 2):
                y = oy + 22

            name_surf = self.font_small.render(name_text, True, (190, 190, 200))
            panel.blit(name_surf, (x, y))

            val_x = x + name_surf.get_width() + 10
            val_surf = self.font_small.render(str(val_text), True, self.text_color)
            panel.blit(val_surf, (val_x, y))

            y += 18