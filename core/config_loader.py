import yaml
from dataclasses import dataclass, field
from typing import Tuple, Union, List, Optional
from screeninfo import get_monitors


@dataclass
class _Config:
    # ==============================
    # SIMULATION / ENVIRONMENT
    # ==============================
    MAP_WIDTH: int = 40
    MAP_HEIGHT: int = 40
    MAX_STEPS: int = 200

    POP_SIZE: int = 50
    RUNS_PER_GENERATION: int = 10

    # ==============================
    # EVOLUTIONARY ALGORITHM
    # ==============================
    MUTATION_RATE: float = 0.04
    MUTATION_STD: float = 3.0
    MUTATION_CLIP: float = 10.0
    ELITISM: float = 0.1
    SURVIVAL_RATE: float = 0.6

    # ==============================
    # VISUALIZATION
    # ==============================
    CELL_SIZE: int = 18
    SCREEN_WIDTH: Union[int, str] = "auto"
    SCREEN_HEIGHT: Union[int, str] = "auto"

    FONDO: Tuple[int, int, int] = (150, 150, 150)
    OBSTACLE_COLOR: Tuple[int, int, int] = (50, 50, 50)
    FOOD_COLOR: Tuple[int, int, int] = (40, 130, 40)
    FONT: str = "Aileron-Regular.otf"
    MAX_FPS: int = 9999

    # ==============================
    # EXPERIMENT / TEST
    # ==============================
    TEST: str = "egg_hunt"
    NETWORK_TYPE: str = "MLP_with_memory"

    # Specific to "egg_hunt"
    MIN_DISTANCE_egg: Optional[int] = None

    # ==============================
    # AGENT BRAINS
    # ==============================

    # --- MLP with Memory ---
    MEMORY_SIZE: int = 10
    HIDDEN_ARCHITECTURE: List[int] = field(default_factory=list)
    VISION_RADIUS: int = 1

    # --- NEAT ---
    # (placeholder)

    # --- Simple RNN ---
    HIDDEN_STATE_SIZE: Optional[int] = None

    # ==============================
    # DERIVED ATTRIBUTES
    # ==============================
    total_cells: int = 0
    EDGE_SIZE: int = 0
    INFO_Y: int = 0
    INFO_X: int = 50
    PAD: int = 3

    def finalize(self) -> "_Config":
        # Convert colors from list -> tuple if needed
        if isinstance(self.FONDO, list):
            self.FONDO = tuple(self.FONDO)
        if isinstance(self.OBSTACLE_COLOR, list):
            self.OBSTACLE_COLOR = tuple(self.OBSTACLE_COLOR)
        if isinstance(self.FOOD_COLOR, list):
            self.FOOD_COLOR = tuple(self.FOOD_COLOR)

        # Auto screen size from monitor
        if self.SCREEN_WIDTH == "auto" or self.SCREEN_HEIGHT == "auto":
            mon = get_monitors()[0]
            self.SCREEN_WIDTH = mon.width
            self.SCREEN_HEIGHT = mon.height

        # Derived params
        self.EDGE_SIZE = self.SCREEN_WIDTH // 30
        self.PAD = self.VISION_RADIUS
        self.INFO_Y = self.SCREEN_HEIGHT - 31
        self.total_cells = self.MAP_WIDTH * self.MAP_HEIGHT

        return self


def load_config(path: str = "config.yaml") -> _Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)  # carga YAML en dict de Python
    return _Config(**data).finalize()


Config = load_config()
