from typing import Tuple, Optional, List

import numpy as np

from create_map import AmbulanceSimulation
from param import Emergency, Hospital, EmergencyPriority, AmbulanceStation


class GameTheory:
    def __init__(self, simulation: AmbulanceSimulation):
        self.simulation = simulation
        self.decision_history = []

    def make_dispatch_decision(self, emergency: Emergency) -> Tuple[Optional[str], Optional[str]]:
        available_stations = self._get_available_ambulance_stations()
        available_hospitals = self._get_available_hospitals()
        if not available_stations or not available_hospitals:
            return None, None

        decisions_with_payoff = []
        for station_id in available_stations:
            station = next(s for s in self.simulation.ambulance_stations if s.id == station_id)
            for hospital_id in available_hospitals:
                hospital = next(h for h in self.simulation.hospitals if h.id == hospital_id)
                payoff = self._calculate_decision_payoff(emergency, station, hospital)
                decisions_with_payoff.append((station_id, hospital_id, payoff))

        if not decisions_with_payoff:
            return None, None

        best_decision = max(decisions_with_payoff, key=lambda x: x[2])
        best_station, best_hospital, best_payoff = best_decision

        self.decision_history.append({
            'emergency_id': emergency.id,
            'station_id': best_station,
            'hospital_id': best_hospital,
            'payoff': best_payoff,
            'priority': emergency.priority.value,
            'hospital_strategy': next(h.strategy for h in self.simulation.hospitals if h.id == best_hospital)
        })

        return best_station, best_hospital

    def _calculate_decision_payoff(self, emergency: Emergency, station: AmbulanceStation, hospital: Hospital) -> float:
        response_time = self._calculate_total_response_time(station.position, emergency.position, hospital.position)
        base_payoff = 100 - response_time

        strategy_adjustment = self._get_strategy_adjustment(hospital.strategy, hospital, emergency.priority)
        distance_penalty = self._calculate_distance_penalty(station.position, emergency.position)

        utilization_penalty = self._calculate_utilization_penalty(station)
        final_payoff = (base_payoff + strategy_adjustment - distance_penalty - utilization_penalty)
        return max(0.0, final_payoff)

    def _calculate_total_response_time(self, station_pos: Tuple[float, float], emergency_pos: Tuple[float, float],
                                       hospital_pos: Tuple[float, float]) -> float:
        to_scene_time = self._calculate_travel_time(station_pos, emergency_pos)

        if emergency_pos == EmergencyPriority.PRIORITY_1:
            on_scene_time = 5
        elif emergency_pos == EmergencyPriority.PRIORITY_2:
            on_scene_time = 8
        else:
            on_scene_time = 10

        to_hospital_time = self._calculate_travel_time(emergency_pos, hospital_pos)
        return to_scene_time + on_scene_time + to_hospital_time

    def _calculate_travel_time(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        distance = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
        base_speed = 40
        travel_time = (distance / base_speed) * 60
        return max(1, travel_time)

    def _get_strategy_adjustment(self, strategy: str, hospital: Hospital, priority: EmergencyPriority) -> float:
        capacity_utilization = hospital.current_patients / hospital.emergency_beds

        # cooperative
        if strategy == "cooperative":
            priority_bonus = 10 if priority == EmergencyPriority.PRIORITY_1 else 0
            load_penalty = capacity_utilization * 15
            return priority_bonus - load_penalty

        # competitive
        elif strategy == "competitive":
            load_bonus = 20 * (1 - capacity_utilization)
            return load_bonus

        # balanced
        else:
            priority_bonus = 5 if priority == EmergencyPriority.PRIORITY_1 else 0
            load_penalty = 10 * capacity_utilization
            return priority_bonus - load_penalty

    def _calculate_distance_penalty(self, station_pos: Tuple[float, float],
                                    emergency_pos: Tuple[float, float]) -> float:
        distance = np.sqrt((station_pos[0] - emergency_pos[0]) ** 2 + (station_pos[1] - emergency_pos[1]) ** 2)
        max_reasonable_distance = self.simulation.city_width * 0.3
        if distance > max_reasonable_distance:
            excess_ratio = (distance - max_reasonable_distance) / max_reasonable_distance
            return 50 * excess_ratio
        return 0

    def _calculate_utilization_penalty(self, station: AmbulanceStation) -> float:
        utilization = 1 - (station.available_ambulances / station.ambulance_count)
        if utilization > 0.8:
            excess_utilization = (utilization - 0.8) / 0.2
            return 30 * excess_utilization
        return 0

    def _get_available_ambulance_stations(self) -> List[str]:
        return [station.id for station in self.simulation.ambulance_stations if station.available_ambulances > 0]

    def _get_available_hospitals(self) -> List[str]:
        return [hospital.id for hospital in self.simulation.hospitals
                if hospital.current_patients < hospital.emergency_beds]
