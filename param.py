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