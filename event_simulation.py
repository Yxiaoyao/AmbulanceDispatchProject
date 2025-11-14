import numpy as np
import random
from param import Emergency, HospitalType, EmergencyPriority
from create_map import AmbulanceSimulation
from typing import Tuple, Optional
import heapq
from collections import defaultdict

class EventSimulator:

    def __init__(self, simulation: AmbulanceSimulation):
        self.simulation = simulation
        self.event_queue = []
        self.current_time = 0
        self.clock = 0
        self.performance_stats = {
            'total_emergencies': 0,
            'served_emergencies': 0,
            'response_times': [],
            'ambulance_utilization': [],
            'hospital_utilization': [],
            'strategy_performance': defaultdict(list),
            'waiting_times': [],
            'service_times': []
        }
        self.ambulance_locations = {}
        self.emergency_records = {}

    def schedule(self, event_time: float, event_type: str, **kwargs):
        heapq.heappush(self.event_queue, (event_time, event_type, kwargs))

    def run_simulation(self, simulation_time: float = 24 * 60):
        self._generate_initial_emergencies()
        while self.event_queue and self.current_time < simulation_time:
            event_time, event_type, event_data = heapq.heappop(self.event_queue)
            self.current_time = event_time
            if event_type == 'emergency_occur':
                self._handle_emergency_occur(event_data)
            elif event_type == 'ambulance_dispatch':
                self._handle_ambulance_dispatch(event_data)
            elif event_type == 'ambulance_arrival':
                self._handle_ambulance_arrival(event_data)
            elif event_type == 'hospital_arrival':
                self._handle_hospital_arrival(event_data)
            elif event_type == 'ambulance_return':
                self._handle_ambulance_return(event_data)
        self._calculate_final_statistics()
        return self.performance_stats

    def _get_time_factor(self, time: float) -> float:
        hour = time / 60
        if 7 <= hour < 9 or 17 <= hour < 19: # 早晚高峰
            return 1.8
        elif 9 <= hour < 17:
            return 1.2
        else:
            return 0.6

    def _handle_emergency_occur(self, event_data):
        emergency = self._create_emergency(event_data['emergency_id'])
        self.simulation.emergencies.append(emergency)
        self.schedule(event_time=self.current_time, event_type='ambulance_dispatch', emergency_id=emergency.id)
        print(f"Time {self.current_time:.1f}: emergency event {emergency.id} occur")

    def _select_emergency_type(self) -> Tuple[str, EmergencyPriority]:
        types = {
            'cardiac': (EmergencyPriority.PRIORITY_1, 0.25),
            'trauma': (EmergencyPriority.PRIORITY_2, 0.30),
            'respiratory': (EmergencyPriority.PRIORITY_2, 0.20),
            'other': (EmergencyPriority.PRIORITY_3, 0.25)
        }
        type_names = list(types.keys())
        weights = [types[t][1] for t in type_names]
        chosen_type = random.choices(type_names, weights=weights)[0]
        priority = types[chosen_type][0]
        return chosen_type, priority

    def _handle_ambulance_dispatch(self, event_data):
        emergency_id = event_data['emergency_id']
        if emergency_id not in self.emergency_records:
            return
        emergency_record = self.emergency_records[emergency_id]
        emergency = emergency_record['emergency']

        # game_theoretic
        station_id, hospital_id = self.simulation.game_theoretic_dispatcher.make_dispatch_decision(emergency)
        if station_id and hospital_id:
            ambulance_id = self._find_available_ambulance(station_id)
            if ambulance_id:
                # update status
                self.ambulance_locations[ambulance_id]['status'] = 'dispatched'
                self.ambulance_locations[ambulance_id]['current_emergency'] = emergency_id

                # update event
                emergency_record['dispatch_time'] = self.current_time
                emergency_record['assigned_ambulance'] = ambulance_id
                emergency_record['assigned_hospital'] = hospital_id
                emergency_record['assigned_station'] = station_id

                # cal time
                station = next(s for s in self.simulation.ambulance_stations if s.id == station_id)
                travel_time = self._calculate_travel_time(station.position, emergency.position)
                arrival_time = self.current_time + travel_time
                self.schedule(arrival_time, "ambulance_arrival",
                              emergency_id=emergency_id, ambulance_id=ambulance_id)

                # update acailable
                station.available_ambulances -= 1

                print(
                    f"Time {self.current_time:.1f}: dispatch {ambulance_id} to {emergency.id}, after {travel_time:.1f}min ")

            else:
                retry_delay = np.random.exponential(5)
                self.schedule(self.current_time + retry_delay, "ambulance_dispatch",
                              emergency_id=emergency_id)
                print(f"Time {self.current_time:.1f}: no available，{retry_delay:.1f}min retry {emergency.id}")
        else:
            print(f"Time {self.current_time:.1f}: can not find resource for {emergency.id}")

    def _handle_ambulance_arrival(self, event_data):
        emergency_id = event_data['emergency_id']
        ambulance_id = event_data['ambulance_id']

        if emergency_id not in self.emergency_records:
            return

        emergency_record = self.emergency_records[emergency_id]
        emergency = emergency_record['emergency']

        # update status
        self.ambulance_locations[ambulance_id]['status'] = 'on_scene'
        emergency_record['ambulance_arrival_time'] = self.current_time

        # cal time
        response_time = self.current_time - emergency_record['start_time']
        self.performance_stats['response_times'].append(response_time)

        if emergency.priority == EmergencyPriority.PRIORITY_1:
            on_scene_time = np.random.triangular(3, 5, 8)
        elif emergency.priority == EmergencyPriority.PRIORITY_2:
            on_scene_time = np.random.triangular(5, 8, 12)
        else:
            on_scene_time = np.random.triangular(8, 12, 15)

        hospital_id = emergency_record['assigned_hospital']
        hospital = next(h for h in self.simulation.hospitals if h.id == hospital_id)
        travel_time_to_hospital = self._calculate_travel_time(emergency.position, hospital.position)

        hospital_arrival_time = self.current_time + on_scene_time + travel_time_to_hospital
        self.schedule(hospital_arrival_time, "hospital_arrival",
                      emergency_id=emergency_id, ambulance_id=ambulance_id)
        print(
            f"Time {self.current_time:.1f}: {ambulance_id} arrival {emergency.id}, response time: {response_time:.1f}min")

    def _handle_hospital_arrival(self, event_data):
        emergency_id = event_data['emergency_id']
        ambulance_id = event_data['ambulance_id']

        if emergency_id not in self.emergency_records:
            return

        emergency_record = self.emergency_records[emergency_id]
        emergency = emergency_record['emergency']

        # update status
        self.ambulance_locations[ambulance_id]['status'] = 'transporting'
        emergency_record['hospital_arrival_time'] = self.current_time
        service_time = self.current_time - emergency_record['start_time']
        self.performance_stats['service_times'].append(service_time)

        # update hospital status
        hospital_id = emergency_record['assigned_hospital']
        hospital = next(h for h in self.simulation.hospitals if h.id == hospital_id)
        hospital.current_patients += 1
        strategy = hospital.strategy
        response_time = emergency_record['ambulance_arrival_time'] - emergency_record['start_time']
        self.performance_stats['strategy_performance'][strategy].append(response_time)

        if hospital.hospital_type == HospitalType.GENERAL:
            if emergency.priority == EmergencyPriority.PRIORITY_1:
                hospital_processing_time = np.random.triangular(10, 15, 25)
            else:
                hospital_processing_time = np.random.triangular(15, 25, 40)
        else:
            if emergency.priority == EmergencyPriority.PRIORITY_1:
                hospital_processing_time = np.random.triangular(15, 25, 35)
            else:
                hospital_processing_time = np.random.triangular(20, 35, 50)

        return_time = self.current_time + hospital_processing_time
        self.schedule(return_time, 'ambulance_return', ambulance_id=ambulance_id, hospital_id=hospital_id)
        self.performance_stats['served_emergencies'] += 1
        print(
            f"Time {self.current_time:.1f}: {ambulance_id} deliver {emergency.id} to {hospital_id}, "
            f"service_times: {service_time:.1f}min")

    def _handle_ambulance_return(self, event_data):
        ambulance_id = event_data['ambulance_id']
        hospital_id = event_data['hospital_id']

        # update hos status
        hospital = next(h for h in self.simulation.hospitals if h.id == hospital_id)
        hospital.current_patients = max(0, hospital.current_patients - 1)

        # update amb status
        ambulance_info = self.ambulance_locations[ambulance_id]
        ambulance_info['status'] = 'available'
        ambulance_info['current_emergency'] = None

        # return station
        station_id = ambulance_info['station_id']
        station = next(s for s in self.simulation.ambulance_stations if s.id == station_id)
        station_position = station.position

        # cal return time
        travel_time_back = self._calculate_travel_time(
            hospital.position, station_position
        )
        actual_return_time = self.current_time + travel_time_back

        # update staion
        self.ambulance_locations[ambulance_id]['location'] = station_position
        station.available_ambulances += 1
        utilization_rate = 1 - (station.available_ambulances / station.ambulance_count)
        self.performance_stats['ambulance_utilization'].append(utilization_rate)

        print(
            f"Time {self.current_time:.1f}: {ambulance_id} return {station_id}, after {travel_time_back:.1f}min")

    def _handle_performance_monitor(self):
        available_ambulances = sum(
            1 for amb in self.ambulance_locations.values()
            if amb['status'] == 'available'
        )
        total_ambulances = len(self.ambulance_locations)
        utilization = 1 - (available_ambulances / total_ambulances)
        self.performance_stats['ambulance_utilization'].append(utilization)

        next_monitor_time = self.current_time + 30
        if next_monitor_time < 24 * 60:  # 24小时内
            self.schedule(next_monitor_time, "performance_monitor")

    def _calculate_final_statistics(self):
        stats = self.performance_stats

        if stats['response_times']:
            stats['avg_response_time'] = np.mean(stats['response_times'])
            stats['median_response_time'] = np.median(stats['response_times'])
            stats['response_time_std'] = np.std(stats['response_times'])
            stats['coverage_8min'] = len([t for t in stats['response_times'] if t <= 8]) / len(
                stats['response_times']) * 100
        else:
            stats['avg_response_time'] = 0
            stats['median_response_time'] = 0
            stats['response_time_std'] = 0
            stats['coverage_8min'] = 0

        if stats['ambulance_utilization']:
            stats['avg_utilization'] = np.mean(stats['ambulance_utilization'])
        else:
            stats['avg_utilization'] = 0

        if stats['service_times']:
            stats['avg_service_time'] = np.mean(stats['service_times'])
        else:
            stats['avg_service_time'] = 0

        stats['service_rate'] = stats['served_emergencies'] / stats['total_emergencies'] * 100 \
            if stats['total_emergencies'] > 0 else 0

        print("\n=== Event simulation result ===")
        print(f"total events: {stats['total_emergencies']}")
        print(f"service events: {stats['served_emergencies']}")
        print(f"service rate: {stats['service_rate']:.1f}%")
        print(f"avg response time: {stats['avg_response_time']:.1f}min")
        print(f"coverage 8min: {stats['coverage_8min']:.1f}%")
        print(f"avg utilization: {stats['avg_utilization']:.1f}%")

    def _find_available_ambulance(self, station_id: str) -> Optional[str]:
        for ambulance_id, ambulance_info in self.ambulance_locations.items():
            if ambulance_info['station_id'] == station_id and ambulance_info['status'] == 'available':
                return ambulance_id
        return None

    def _calculate_travel_time(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        distance = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
        base_speed = 50  # km/h

        hour = self.current_time / 60
        if 7 <= hour < 9 or 17 <= hour < 19:  # 早晚高峰
            traffic_factor = 0.5
        elif 9 <= hour < 17:  # 日间
            traffic_factor = 0.7
        else:  # 夜间
            traffic_factor = 0.9

        if self._is_on_arterial_road(start, end):
            road_factor = 1.2
        else:
            road_factor = 0.8

        effective_speed = base_speed * traffic_factor * road_factor
        travel_time = (distance / effective_speed) * 60

        return max(1, travel_time)

    def _is_on_arterial_road(self, start: Tuple[float, float], end: Tuple[float, float]) -> bool:
        city_center = (self.simulation.city_width * 0.5, self.simulation.city_height * 0.5)
        start_dist = np.sqrt((start[0] - city_center[0]) ** 2 + (start[1] - city_center[1]) ** 2)
        end_dist = np.sqrt((end[0] - city_center[0]) ** 2 + (end[1] - city_center[1]) ** 2)
        max_city_radius = self.simulation.city_width * 0.3
        return start_dist <= max_city_radius or end_dist <= max_city_radius

    def _generate_initial_emergencies(self):
        base_rate = 0.15

        time = 0
        emergency_id = 1
        while time < 24 * 60:
            time_factor = self._get_time_factor(time)
            current_rate = base_rate * time_factor

            inter_arrival_time = np.random.exponential(1.0 / current_rate)
            time += inter_arrival_time

            if time < 24 * 60:
                self.schedule(time, "emergency_occur",
                              emergency_id=f"E{emergency_id}")
                emergency_id += 1
                self.performance_stats['total_emergencies'] += 1

        self.schedule(30, "performance_monitor")

    def _create_emergency(self, emergency_id: str) -> Emergency:
        if self.simulation.residential_areas:
            weights = [area.population for area in self.simulation.residential_areas]
            chosen_area = random.choices(self.simulation.residential_areas, weights=weights)[0]
            base_x, base_y = chosen_area.position
            x = np.random.normal(base_x, self.simulation.city_width * 0.05)
            y = np.random.normal(base_y, self.simulation.city_height * 0.05)
            residential_area_id = chosen_area.id
        else:
            x = np.random.uniform(0, self.simulation.city_width)
            y = np.random.uniform(0, self.simulation.city_height)
            residential_area_id = None

        x = np.clip(x, 0, self.simulation.city_width)
        y = np.clip(y, 0, self.simulation.city_height)

        emergency_type, priority = self._select_emergency_type()

        return Emergency(
            id=emergency_id,
            position=(x, y),
            emergency_type=emergency_type,
            priority=priority,
            timestamp=self.current_time,
            residential_area_id=residential_area_id
        )
