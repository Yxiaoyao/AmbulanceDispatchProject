import json
import os

import numpy as np
import random

from game_theory import GameTheory
from param import Emergency, HospitalType, EmergencyPriority
from create_map import AmbulanceSimulation
from typing import Tuple, Optional
import heapq
from collections import defaultdict

class EventSimulator:

    def __init__(self, simulation: AmbulanceSimulation):
        self.simulation = simulation
        self.game_theory = GameTheory(self.simulation)
        self.hospitals = simulation.hospitals
        self.ambulance_stations = simulation.ambulance_stations
        self.residential_areas = simulation.residential_areas
        self.city_width = simulation.city_width
        self.city_height = simulation.city_height
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
            'service_times': [],
            'failed_emergencies': 0,
            'fallback_decisions': 0
        }
        self.ambulance_infos = {}
        self.emergency_records = {}

    def schedule(self, event_time: float, event_type: str, **kwargs):
        heapq.heappush(self.event_queue, (event_time, event_type, kwargs))

    def run_simulation(self, simulation_time: float = 24 * 60):
        print(f"run simulation: {simulation_time} min")
        self._initialize_ambulance_infos()
        self._generate_initial_emergencies()

        events_processed = 0
        max_events = 10000
        while self.event_queue and self.current_time < simulation_time and events_processed < max_events:
            event_time, event_type, event_data = heapq.heappop(self.event_queue)
            self.current_time = event_time
            events_processed += 1
            try:
                if event_type == 'emergency_occur':
                    print(" ********* emergency_occur *********")
                    self._handle_emergency_occur(event_data)
                elif event_type == 'ambulance_dispatch':
                    print(" ********* ambulance_dispatch *********")
                    self._handle_ambulance_dispatch(event_data)
                elif event_type == 'ambulance_arrival':
                    print(" ********* ambulance_arrival *********")
                    self._handle_ambulance_arrival(event_data)
                elif event_type == 'hospital_arrival':
                    print(" ********* hospital_arrival *********")
                    self._handle_hospital_arrival(event_data)
                elif event_type == 'ambulance_return':
                    print(" ********* ambulance_return *********")
                    self._handle_ambulance_return(event_data)
            except Exception as e:
                print(f"{event_type} error: {e}")
                continue
        print(f"simulation finished: {events_processed} events")
        self._calculate_final_statistics()
        return self.performance_stats

    def _initialize_ambulance_infos(self):
        print("initializing ambulance locations")
        for station in self.ambulance_stations:
            for i in range(station.ambulance_count):
                ambulance_id = f"{station.id}_amb{i + 1}"
                self.ambulance_infos[ambulance_id] = {
                    'location': station.position,
                    'status': 'available',
                    'station_id': station.id,
                    'current_emergency': None
                }
            station.available_ambulances = station.ambulance_count
            print(f"ambulance station {station.id}: {station.ambulance_count} cars")

    def _get_time_factor(self, time: float) -> float:
        hour = time / 60
        if 7 <= hour < 9 or 17 <= hour < 19:  # 早晚高峰
            return 1.8
        elif 9 <= hour < 17:
            return 1.2
        else:
            return 0.6

    def _handle_emergency_occur(self, event_data):
        try:
            emergency = self._create_emergency(event_data['emergency_id'])
            self.simulation.emergencies.append(emergency)
            self.emergency_records[emergency.id] = {
                'emergency': emergency,
                'start_time': self.current_time,
                'dispatch_time': None,
                'ambulance_arrival_time': None,
                'hospital_arrival_time': None,
                'completion_time': None,
                'retry_count': 0
            }
            self.schedule(event_time=self.current_time, event_type='ambulance_dispatch', emergency_id=emergency.id)
            print(f"Time {self.current_time:.1f}: emergency event {emergency.id} occur")
        except Exception as e:
            print(f"emergency occur error occur: {e}")

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
        try:
            emergency_id = event_data['emergency_id']
            if emergency_id not in self.emergency_records:
                print(f"Error: {emergency_id} not in emergency records")
                return
            emergency_record = self.emergency_records[emergency_id]
            emergency = emergency_record['emergency']

            if emergency_record['retry_count'] >= 5:
                print(f"Time {self.current_time:.1f}: emergency {emergency.id} fail")
                self.performance_stats['failed_emergencies'] += 1
                emergency.status = 'failed'
                return
            emergency_record['retry_count'] += 1

            # game_theoretic
            station_id, hospital_id = self.game_theory.make_dispatch_decision(emergency)
            # print(f"make_dispatch_decision called with emergency: {emergency}")
            if station_id and hospital_id:
                ambulance_id = self._find_available_ambulance(station_id)
                if ambulance_id:
                    # update status
                    self.ambulance_infos[ambulance_id]['status'] = 'dispatched'
                    self.ambulance_infos[ambulance_id]['current_emergency'] = emergency_id

                    # update event
                    emergency_record['dispatch_time'] = self.current_time
                    emergency_record['assigned_ambulance'] = ambulance_id
                    emergency_record['assigned_hospital'] = hospital_id
                    emergency_record['assigned_station'] = station_id
                    emergency.status = 'dispatched'
                    emergency_record['retry_count'] = 0

                    # cal time
                    station = next(s for s in self.ambulance_stations if s.id == station_id)
                    travel_time = self._calculate_travel_time(station.position, emergency.position)
                    arrival_time = self.current_time + travel_time
                    self.schedule(arrival_time, "ambulance_arrival",
                                  emergency_id=emergency_id, ambulance_id=ambulance_id)

                    # update available
                    station.available_ambulances -= 1
                    print(f"Time {self.current_time:.1f}: "
                          f"dispatch {ambulance_id} to {emergency.id}, after {travel_time:.1f}min ")

                else:
                    retry_delay = np.random.exponential(2.0)
                    next_dispatch_time = self.current_time + retry_delay
                    if next_dispatch_time < 24 * 60:
                        self.schedule(self.current_time + retry_delay, "ambulance_dispatch",
                                      emergency_id=emergency_id)
                        print(f"Time {self.current_time:.1f}: no available，{retry_delay:.1f}min retry {emergency.id}")
            else:
                self.performance_stats['fallback_decisions'] += 1
                retry_delay = np.random.exponential(3.0)
                next_dispatch_time = self.current_time + retry_delay
                if next_dispatch_time < 24 * 60:
                    self.schedule(next_dispatch_time, "ambulance_dispatch", emergency_id=emergency_id)
                    print(f"Time {self.current_time:.1f}: can not find resource for {emergency.id}")
        except Exception as e:
            print(f"ambulance dispatch error occur: {e}")

    def _handle_ambulance_arrival(self, event_data):
        try:
            emergency_id = event_data['emergency_id']
            ambulance_id = event_data['ambulance_id']

            if emergency_id not in self.emergency_records:
                print(f"Error: {emergency_id} not in emergency records")
                return

            emergency_record = self.emergency_records[emergency_id]
            emergency = emergency_record['emergency']

            # update status
            self.ambulance_infos[ambulance_id]['status'] = 'on_scene'
            emergency_record['ambulance_arrival_time'] = self.current_time
            emergency.status = 'on_scene'

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
            hospital = next(h for h in self.hospitals if h.id == hospital_id)
            travel_time_to_hospital = self._calculate_travel_time(emergency.position, hospital.position)

            hospital_arrival_time = self.current_time + on_scene_time + travel_time_to_hospital
            self.schedule(hospital_arrival_time, "hospital_arrival",
                          emergency_id=emergency_id, ambulance_id=ambulance_id)
            print(
                f"Time {self.current_time:.1f}: {ambulance_id} arrival {emergency.id}, response time: {response_time:.1f}min")

        except Exception as e:
            print(f"ambulance arrival error occur: {e}")

    def _handle_hospital_arrival(self, event_data):
        try:
            emergency_id = event_data['emergency_id']
            ambulance_id = event_data['ambulance_id']

            if emergency_id not in self.emergency_records:
                print(f"Error: {emergency_id} not in emergency records")
                return

            emergency_record = self.emergency_records[emergency_id]
            emergency = emergency_record['emergency']

            # update status
            self.ambulance_infos[ambulance_id]['status'] = 'transporting'
            emergency_record['hospital_arrival_time'] = self.current_time
            emergency.status = 'at_hospital'
            service_time = self.current_time - emergency_record['start_time']
            self.performance_stats['service_times'].append(service_time)

            # update hospital status
            hospital_id = emergency_record['assigned_hospital']
            hospital = next(h for h in self.hospitals if h.id == hospital_id)
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
            emergency.status = 'completed'
            print(
                f"Time {self.current_time:.1f}: {ambulance_id} deliver {emergency.id} to {hospital_id}, "
                f"service_times: {service_time:.1f}min")
        except Exception as e:
            print(f"hospital arrival error occur: {e}")

    def _handle_ambulance_return(self, event_data):
        try:
            ambulance_id = event_data['ambulance_id']
            hospital_id = event_data['hospital_id']

            # update hos status
            hospital = next(h for h in self.hospitals if h.id == hospital_id)
            hospital.current_patients = max(0, hospital.current_patients - 1)

            # update amb status
            self.ambulance_infos[ambulance_id]['status'] = 'available'
            self.ambulance_infos[ambulance_id]['current_emergency'] = None

            # return station
            station_id = self.ambulance_infos[ambulance_id]['station_id']
            station = next(s for s in self.ambulance_stations if s.id == station_id)
            station_position = station.position
            self.ambulance_infos[ambulance_id]['location'] = station_position

            # cal return time
            travel_time_back = self._calculate_travel_time(
                hospital.position, station_position
            )

            # update staion
            station.available_ambulances += 1
            utilization_rate = 1 - (station.available_ambulances / station.ambulance_count)
            self.performance_stats['ambulance_utilization'].append(utilization_rate)

            print(
                f"Time {self.current_time:.1f}: {ambulance_id} return {station_id}, after {travel_time_back:.1f}min")
        except Exception as e:
            print(f"ambulance return error occur: {e}")

    # def _handle_performance_monitor(self):
    #     available_ambulances = sum(
    #         1 for amb in .values()
    #         if amb['status'] == 'available'
    #     )
    #     total_ambulances = len(self.ambulance_locations)
    #     utilization = 1 - (available_ambulances / total_ambulances)
    #     self.performance_stats['ambulance_utilization'].append(utilization)
    #
    #     next_monitor_time = self.current_time + 30
    #     if next_monitor_time < 24 * 60:  # 24小时内
    #         self.schedule(next_monitor_time, "performance_monitor")

    def _calculate_final_statistics(self):
        stats = self.performance_stats
        total_emergencies = stats['total_emergencies']
        served_emergencies = stats['served_emergencies']
        print(f"=== events simulation result ===")
        print(f"total_emergencies: {total_emergencies}")
        print(f"served_emergencies: {served_emergencies}")
        if total_emergencies > 0:
            service_rate = served_emergencies / total_emergencies * 100.0
            print(f"Service rate: {service_rate:.1f}%")
        else:
            service_rate = 0.0
            print(f"Service rate: 0.0%")

        if stats['response_times']:
            stats['avg_response_time'] = np.mean(stats['response_times'])
            stats['median_response_time'] = np.median(stats['response_times'])
            stats['response_time_std'] = np.std(stats['response_times'])
            stats['coverage_8min'] = len([t for t in stats['response_times'] if t <= 8]) / len(
                stats['response_times']) * 100
            print(f"Average response time: {stats['avg_response_time']:.1f}min")
            print(f"Coverage 8min: {stats['coverage_8min']:.1f}%")
        else:
            stats['avg_response_time'] = 0.0
            stats['median_response_time'] = 0.0
            stats['response_time_std'] = 0.0
            stats['coverage_8min'] = 0.0
            print(f"Average response time: 0.0min")
            print(f"Coverage 8min: 0.0%")

        if stats['ambulance_utilization']:
            stats['avg_utilization'] = np.mean(stats['ambulance_utilization'])
            print(f"Average utilization: {stats['avg_utilization']:.1f}%")
        else:
            stats['avg_utilization'] = 0.0
            print(f"Average utilization: 0.0%")

        if stats['service_times']:
            stats['avg_service_time'] = float(np.mean(stats['service_times']))
            print(f"Average service time: {stats['avg_service_time']:.1f}min")
        else:
            stats['avg_service_time'] = 0.0

        stats['service_rate'] = stats['served_emergencies'] / stats['total_emergencies'] * 100 \
            if stats['total_emergencies'] > 0 else 0

        print("\n=== End event simulation ===")

    def _find_available_ambulance(self, station_id: str) -> Optional[str]:
        for ambulance_id, ambulance_info in self.ambulance_infos.items():
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

        return max(1.0, travel_time)

    def _is_on_arterial_road(self, start: Tuple[float, float], end: Tuple[float, float]) -> bool:
        city_center = (self.city_width * 0.5, self.city_height * 0.5)
        start_dist = np.sqrt((start[0] - city_center[0]) ** 2 + (start[1] - city_center[1]) ** 2)
        end_dist = np.sqrt((end[0] - city_center[0]) ** 2 + (end[1] - city_center[1]) ** 2)
        max_city_radius = self.city_width * 0.3
        return start_dist <= max_city_radius or end_dist <= max_city_radius

    def _generate_initial_emergencies(self):
        base_rate = 0.15

        time = 0.0
        emergency_id = 1
        max_emergencies = 200  # 限制最大事件数

        while time < 24 * 60 and emergency_id <= max_emergencies:
            time_factor = self._get_time_factor(time)
            current_rate = base_rate * time_factor

            inter_arrival_time = np.random.exponential(1.0 / max(0.01, current_rate))
            time += inter_arrival_time

            if time < 24 * 60:
                self.schedule(time, "emergency_occur",
                              emergency_id=f"E{emergency_id}")
                emergency_id += 1
                self.performance_stats['total_emergencies'] += 1

        self.schedule(30, "performance_monitor")

    def _create_emergency(self, emergency_id: str) -> Emergency:
        if self.residential_areas:
            weights = [area.population for area in self.residential_areas]
            chosen_area = random.choices(self.residential_areas, weights=weights)[0]
            base_x, base_y = chosen_area.position
            x = np.random.normal(base_x, self.city_width * 0.05)
            y = np.random.normal(base_y, self.city_height * 0.05)
            residential_area_id = chosen_area.id
        else:
            x = np.random.uniform(0, self.city_width)
            y = np.random.uniform(0, self.city_height)
            residential_area_id = None

        x = np.clip(x, 0, self.city_width)
        y = np.clip(y, 0, self.city_height)

        emergency_type, priority = self._select_emergency_type()

        return Emergency(
            id=emergency_id,
            position=(x, y),
            emergency_type=emergency_type,
            priority=priority,
            timestamp=self.current_time,
            residential_area_id=residential_area_id
        )

    def _print_initial_status(self):
        print("====== Initial information ======")
        total_ambulances = sum(station.ambulance_count for station in self.ambulance_stations)
        available_ambulances = sum(station.available_ambulances for station in self.ambulance_stations)

        total_beds = sum(hospital.emergency_beds for hospital in self.hospitals)
        available_beds = sum(hospital.emergency_beds - hospital.current_patients for hospital in self.hospitals)

        print(f"ambulances: {available_ambulances}/{total_ambulances}")
        print(f"beds: {available_beds}/{total_beds} ")

        print("\n--- station ---")
        for station in self.ambulance_stations:
            print(f"  {station.id}: {station.available_ambulances}/{station.ambulance_count} available ambulances")

        print("\n--- hospital ---")
        for hospital in self.hospitals:
            print(f"  {hospital.id}: {hospital.current_patients}/{hospital.emergency_beds} beds occupied")

    def _print_system_status(self):
        available_ambulances = sum(station.available_ambulances for station in self.ambulance_stations)
        total_ambulances = sum(station.ambulance_count for station in self.ambulance_stations)

        available_beds = sum(max(0, hospital.emergency_beds - hospital.current_patients)
                             for hospital in self.hospitals)
        total_beds = sum(hospital.emergency_beds for hospital in self.hospitals)

        print(f"Time {self.current_time:.1f}: available ambulances {available_ambulances}/{total_ambulances}, "
              f"available beds {available_beds}/{total_beds}")
