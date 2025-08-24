import json
from dataclasses import dataclass
from typing import Tuple, Union
from screeninfo import get_monitors


@dataclass
class _Config:

    CELL_SIZE: int = 10
    P_FOOD: float = 0.05
    FOOD_ENERGY: int = 40

    VISION_RADIUS: int = 1
    MEMORY_SIZE: int = 10
    INITIAL_ENERGY: int = 200
    MOVE_STAND_RATIO: int = 3
    NETWORK_TYPE: Union[int, str] = "auto"

    MUTATION_RATE: float = 0.1       
    MUTATION_STD: float = 0.05        
    MUTATION_CLIP: float = 1.0  

    ELITISM: float = 0.05,
    SURVIVAL_RATE: float = 0.3,
    POP_SIZE: int = 100
    N_SIMULATIONS: int = 10
    MAX_STEPS: int = 100

    SCREEN_WIDTH: Union[int, str] = "auto"
    SCREEN_HEIGHT: Union[int, str] = "auto"
    EDGE_SIZE: int = 0
    INFO_Y: int = 0
    INFO_X: int = 50
    FONDO: Tuple[int, int, int] = (150, 150, 150)
    OBSTACLE_COLOR: Tuple[int, int, int] = (50, 50, 50)
    FOOD_COLOR: Tuple[int, int, int] = (40, 130, 40)
    FONT: str = "Aileron-Regular.otf"
    MAX_FPS: int = 9999
    PAD: int = 3

    total_cells: int = 0
    MAP_WIDTH: int = 0
    MAP_HEIGHT: int = 0


    def finalize(self) -> "_Config":
        if isinstance(self.FONDO, list): 
            self.FONDO = tuple(self.FONDO)
        if isinstance(self.OBSTACLE_COLOR, list): 
            self.OBSTACLE_COLOR = tuple(self.OBSTACLE_COLOR)
        if isinstance(self.FOOD_COLOR, list): 
            self.FOOD_COLOR = tuple(self.FOOD_COLOR)

        if self.SCREEN_WIDTH == "auto" or self.SCREEN_HEIGHT == "auto":
            mon = get_monitors()[0]
            self.SCREEN_WIDTH = mon.width
            self.SCREEN_HEIGHT = mon.height

        self.EDGE_SIZE = self.SCREEN_WIDTH // 30
        self.PAD = self.VISION_RADIUS
        self.INFO_Y = self.SCREEN_HEIGHT - 31

        return self


def load_config(path: str = "config.json") -> _Config:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _Config(**data).finalize()


Config = load_config()
