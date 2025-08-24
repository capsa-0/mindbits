from core.config_loader import Config

class Camera:
    def __init__(self, viewport_rect, cell_size):
        self.offset_x = Config.PAD
        self.offset_y = Config.PAD
        self.speed = 6
        self.viewport_rect = viewport_rect
        self.cell_size = cell_size

        self.min_x = Config.PAD
        self.min_y = Config.PAD
        self.max_x = Config.MAP_WIDTH - (viewport_rect.width // cell_size) - 1
        self.max_y = Config.MAP_HEIGHT - (viewport_rect.height // cell_size) - 1

    def move(self, dx, dy):
        self.offset_x += dx * self.speed
        self.offset_y += dy * self.speed
        self.offset_x = max(self.min_x, min(self.offset_x, self.max_x))
        self.offset_y = max(self.min_y, min(self.offset_y, self.max_y))

    def world_to_screen(self, x, y):
        return (
            self.viewport_rect.x + (x - self.offset_x) * self.cell_size,
            self.viewport_rect.y + (y - self.offset_y) * self.cell_size
        )
