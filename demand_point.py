class DemandPoint:
    def __init__(self, x, y, weight, emergency_rate):
        self.position = (x, y)
        self.weight = weight
        self.emergency_rate = emergency_rate

class LocationOptimizer:
    def __init__(self, simulation, max_stations, max_ambulances):
        self.simulation = simulation
        self.max_stations = max_stations
        self.max_ambulances = max_ambulances
        self.demand_points = self._extract_demand_points()


    def _extract_demand_points(self):
        demand_points = []
        for area in self.simulation.residential_areas:
            demand_points.extend(self._discretize_area(area))
        return demand_points