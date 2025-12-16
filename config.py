from param import MapType


class SimulationConfig:
    CITY_WIDTH = 70
    CITY_HEIGHT = 70
    MAP_TYPE = MapType.GRID  # MapType.RING

    N_HOSPITALS = 4
    N_AMBULANCE_STATIONS = 8
    N_RESIDENTIAL_AREAS = 15
    N_ROADS = 15

    SIMULATION_TIME_MINUTES = 8 * 60  # 8小时

    RANDOM_SEED = 42

    SAVE_PLOTS = True
    SAVE_DATA = True
    OUTPUT_DIR = "simulation_results"