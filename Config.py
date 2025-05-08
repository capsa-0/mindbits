import sys
from screeninfo import get_monitors

# Clase de configuración con constantes del programa
class Config:

    # Parámetros de la simulación
    CELL_SIZE = 10 # Tamaño de cada celda en píxeles, controla el tamaño del mapa
    POPULATION_SIZE = 500  # Número de individuos en la población
    NEAT_POPULATION_SIZE = 30  # Número de individuos en la población de NEAT
    POP_DENSITY = 0.01  # Densidad de población, controla el tamaño del mapa
    total_cells = int(POPULATION_SIZE / POP_DENSITY)  # Número total de celdas en el mapa
    MAP_WIDTH = int(total_cells**(1/2))  # Ancho del mapa en celdas
    MAP_HEIGHT = MAP_WIDTH  # Alto del mapa en celdas
    P_FOOD = 0.03  # Probabilidad de generación de comida en el mapa
    MIN_ITER = 30  # Número de iteraciones por simulación
    FOOD_ENERGY = 20  # Energía ganada al consumir una unidad de comida
    MUTATION_RATE = 0.05  # Probabilidad de que un parámetro de la red mute
    MUTATION_RATE_DECAY = 0.997
    MUTATION_SCALE = 2 # Escala de la mutación (desvío estándar de la distribución normal utilizada para mutar los parámetros de la red)
    DROP_RATE = 0.002  # Probabilidad de que un parámetro de la red se elimine (dropout)
    ENCODING = 'one_hot'  # Codificación de la red neuronal (opciones: 'one_hot', 'binary', 'real')
    ELITISM = 0.1  # Proporción de individuos que sobreviven a la selección natural (el 20% de los mejores)
    P_SURVIVORS = 0.5  # Proporción de individuos que sobreviven a la selección natural 

    # Parámetros de los individuos
    VISION_RADIUS = 2  # Radio de visión de los individuos (en celdas)
    MEMORY_LENGTH = 10  # Longitud de la memoria de los individuos (número de movimientos recordados)
    MOVE_STAND_RATIO = 2  # Relación de energía gastada al moverse frente a quedarse quieto
    INITIAL_ENERGY = 200  # Energía inicial de cada individuo
    ADULT_AGE = 20  # Edad mínima para que un individuo sea considerado adulto y pueda reproducirse
    SON_ENERGY = int(INITIAL_ENERGY*2)  # Energía necesaria para reproducirse (suma de la energía de ambos padres)
    MATING_TIME = 30
    COOLDOWN_TIME = 10  # Tiempo de espera para que un individuo pueda reproducirse nuevamente después de aparearse

    # Opciones de visualización
    SCREEN_WIDTH, SCREEN_HEIGHT = get_monitors()[0].width, get_monitors()[0].height  # Dimensiones de la pantalla
    EDGE_SIZE =  SCREEN_WIDTH // 30
    INFO_Y = SCREEN_HEIGHT - 31  # Posición Y para mostrar información en pantalla
    INFO_X = 50  # Posición X para mostrar información en pantalla
    FONDO = (150, 150, 150)  # Color de fondo del mapa (RGB)
    OBSTACLE_COLOR = (50, 50, 50)  # Color de los obstáculos (RG
    FOOD_COLOR = (40, 130, 40)  # Color de la comida (RGB)
    FONT = 'Aileron-Regular.otf'  # Fuente utilizada para mostrar texto en pantalla
    FPS = 5000  # Máximo de cuadros por segundo (controla la velocidad de la simulación)
    PAD = VISION_RADIUS

    # Cálculo del tamaño del input y la arquitectura de red
    vision_size = (VISION_RADIUS * 2 + 1) ** 2  # Tamaño del área de visión 
    INPUT_SIZE = vision_size + MEMORY_LENGTH #+ 1 # Tamaño total del input para la red neuronal 
    if ENCODING == 'one_hot':
        INPUT_SIZE = vision_size*6 + MEMORY_LENGTH*9 #+ 1# Tamaño total del input para la red neuronal 

    @staticmethod
    def get_architecture(n_input, n_output=9):
        """
        Calcula la arquitectura de la red neuronal.
        Reduce el número de neuronas en cada capa a la mitad hasta que sea menor o igual al número de salidas.
        :param n_input: Número de neuronas en la capa de entrada
        :param n_output: Número de neuronas en la capa de salida (por defecto 9, para los movimientos posibles)
        :return: Lista con el número de neuronas en cada capa oculta
        """
        arch = []
        n = n_input
        while n // 2 > n_output:  # Mientras el número de neuronas sea mayor que las salidas
            arch.append(int(n // 2))  # Agregar la mitad del número de neuronas a la arquitectura
            n = n // 2  # Reducir el número de neuronas a la mitad
        return arch

    NN_ARCHITECTURE = get_architecture(INPUT_SIZE)  # Arquitectura de la red neuronal basada en el tamaño del input
    NN_ARCHITECTURE = [50,20]

    def get_n_parameters(architecture):
        n_parameters = 0
        for i in range(len(architecture)-1):
            n_parameters += architecture[i] * architecture[i+1] + architecture[i+1]

        return n_parameters
    
    
    N_PARAMETERS = get_n_parameters([INPUT_SIZE] + NN_ARCHITECTURE)

    # Número de generaciones a ciegas
    BLIND_GENS = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # Número de generaciones sin visualización, tomado de los argumentos de la línea de comandos