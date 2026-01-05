from enum import Enum
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


class EmergencyPriority(Enum):
    PRIORITY_1 = 1  # 心脏
    PRIORITY_2 = 2  # 创伤、呼吸
    PRIORITY_3 = 3  # other

class HospitalType(Enum):
    GENERAL = 'general_hospital'
    COMMUNITY = 'community_hospital'

@dataclass
class Hospital:
    id: str
    name: str
    position: Tuple[float, float]
    capacity: int
    hospital_type: HospitalType
    emergency_beds: int
    current_patients: int = 0
    strategy: str = 'cooperative'

@dataclass
class AmbulanceStation:
    id: str
    position: Tuple[float, float]
    ambulance_count: int
    coverage_radius: float
    available_ambulances: int = 0

@dataclass
class ResidentialArea:
    id: str
    name: str
    position: Tuple[float, float]
    population: int
    density: float
    area: float
    area_type: str

@dataclass
class Emergency:
    id: str
    position: Tuple[float, float]
    emergency_type: str
    priority: EmergencyPriority
    timestamp: float
    residential_area_id: Optional[str] = None
    status: str = 'pending'
    assigned_ambulance: Optional[str] = None
    assigned_hospital: Optional[str] = None
    response_time: Optional[float] = None


class MapType(Enum):
    GRID = "grid"  # 方型
    RING = "ring"  # 环形


class ResidentialArea:
    def __init__(self, id, name, position, population, density, area, area_type):
        self.id = id
        self.name = name
        self.position = position
        self.population = population
        self.density = density
        self.area = area
        self.area_type = area_type

        self.width = None
        self.height = None
        self.vertices = None

    def set_rectangle_dimensions(self, width, height):
        self.width = width
        self.height = height
        self._calculate_vertices()

    def _calculate_vertices(self):
        if self.width is not None and self.height is not None:
            x, y = self.position
            half_width = self.width / 2
            half_height = self.height / 2

            self.vertices = [
                (x - half_width, y - half_height),  # 左下角
                (x + half_width, y - half_height),  # 右下角
                (x + half_width, y + half_height),  # 右上角
                (x - half_width, y + half_height),  # 左上角
            ]