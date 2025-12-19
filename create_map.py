import json
import math
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
# from scipy.spatial import Voronoi, voronoi_plot_2d
import random
from typing import List, Dict, Tuple
from param import Hospital, AmbulanceStation, ResidentialArea, Emergency, HospitalType, MapType


class AmbulanceSimulation:
    """
    救护车仿真地图
    """

    def __init__(self, city_width: float = 100, city_height: float = 100, random_seed: int = 42,
                 map_type: MapType = MapType.GRID, ring_density_factor: float = 1.5):
        """
        :param city_width: 城市宽度
        :param city_height: 城市长度
        """
        self.city_width = city_width
        self.city_height = city_height
        self.ring_density_factor = ring_density_factor
        self.hospitals: List[Hospital] = []  # 医院
        self.ambulance_stations: List[AmbulanceStation] = []  # 救护站
        self.residential_areas: List[ResidentialArea] = []  # 居民区
        self.roads: List[dict] = []
        self.emergencies: List[Emergency] = []
        self.event_queue = []
        self.current_time = 0
        self.performance_metrics = {}
        self.formatted_time = datetime.now().strftime("%Y%m%d%H%M%S")

        self.map_type = map_type
        self.city_center = (city_width / 2, city_height / 2)

        np.random.seed(random_seed)

    def gen_city_layout(self,
                        n_hospitals: int = 4,
                        n_stations: int = 8,
                        n_residential_areas: int = 15,
                        n_major_roads: int = 15):
        """生成城市布局"""
        # 1. hospitals
        print("gen hospitals")
        self.hospitals = self._gen_hospitals(n_hospitals)

        # 2. stations
        print("gen ambulance stations")
        self.ambulance_stations = self._gen_ambulance_stations(n_stations)

        # 3. residential areas
        print("gen residential areas")
        self.residential_areas = self._gen_residential_areas(n_residential_areas)

        # 4. roads
        print(f"gen roads for {self.map_type.value}")
        if self.map_type == MapType.GRID:
            self.roads = self._gen_grid_road_network(n_major_roads)
        elif self.map_type == MapType.RING:
            self.roads = self._gen_ring_road_network(n_major_roads)
        else:
            raise ValueError(f"Unsupported map type: {self.map_type}")
        # self.roads = self._gen_road_network(n_major_roads)

        for station in self.ambulance_stations:
            station.available_ambulances = station.ambulance_count

        print("=== gen city layout done. ===")
        self.save_city_layout()

    def _gen_hospitals(self, n: int) -> List[Hospital]:
        """gen hospitals location"""
        hospitals = []

        strategies = ["cooperative", "competitive", "balance"]
        strategy_weights = [0.4, 0.3, 0.3]

        for i in range(n):
            if i == 0:
                if self.map_type == MapType.RING:
                    pop_center = self._calculate_population_center()
                    x, y = pop_center
                    x += np.random.normal(0, self.city_width * 0.05)
                    y += np.random.normal(0, self.city_height * 0.05)
                else:
                    x, y = self.city_width * 0.5, self.city_height * 0.5
                hospital_type = HospitalType.GENERAL
                capacity = np.random.randint(200, 350)
                emergency_beds = capacity // 3
                strategy = "cooperative"
            else:
                # 基于区域
                if self.map_type == MapType.RING:
                    if i < n // 2:
                        distance = np.random.uniform(0.1, 0.4) * min(self.city_width, self.city_height)
                    else:
                        distance = np.random.uniform(0.4, 0.7) * min(self.city_width, self.city_height)
                    angle = 2 * math.pi * i / n
                    distance = np.random.uniform(0.2, 0.6) * min(self.city_width, self.city_height)
                    x = self.city_center[0] + distance * math.cos(angle)
                    y = self.city_center[1] + distance * math.sin(angle)
                else:
                    region = self._get_region_for_facility()
                    x, y = self._get_position_in_region(region)

                if i < 2 or random.random() < 0.4:
                    hospital_type = HospitalType.GENERAL
                    capacity = np.random.randint(100, 250)
                else:
                    hospital_type = HospitalType.COMMUNITY
                    capacity = np.random.randint(50, 150)

                emergency_beds = max(10, capacity // 4)
                strategy = random.choices(strategies, weights=strategy_weights)[0]

            x = np.clip(x, self.city_width * 0.02, self.city_width * 0.98)
            y = np.clip(y, self.city_height * 0.02, self.city_height * 0.98)

            hospital = Hospital(
                id=f'H{i+1}',
                name=f'{hospital_type.value}_{i+1}',
                position=(x, y),
                capacity=capacity,
                hospital_type=hospital_type,
                emergency_beds=emergency_beds,
                strategy=strategy
            )
            hospitals.append(hospital)


        return hospitals

    def _is_in_downtown(self, position: Tuple[float, float]) -> bool:
        """判断位置是否在市中心区域"""
        x, y = position
        downtown_width = self.city_width * 0.4  # 市中心宽度为城市宽度的40%
        downtown_height = self.city_height * 0.4  # 市中心高度为城市高度的40%

        downtown_left = (self.city_width - downtown_width) / 2
        downtown_right = downtown_left + downtown_width
        downtown_bottom = (self.city_height - downtown_height) / 2
        downtown_top = downtown_bottom + downtown_height

        return (downtown_left <= x <= downtown_right and
                downtown_bottom <= y <= downtown_top)

    def _gen_ambulance_stations(self, n: int) -> List[AmbulanceStation]:
        """gen ambulance stations location"""
        stations = []

        if self.residential_areas:
            population_weights = [area.population for area in self.residential_areas]
            positions = [area.position for area in self.residential_areas]

            candidate_indices = np.random.choice(
                len(self.residential_areas),
                size=min(n, len(self.residential_areas)),
                p=np.array(population_weights)/sum(population_weights),
                replace=False
            )
            for i, idx in enumerate(candidate_indices):
                if i >= n:
                    break

                base_x, base_y = positions[idx]
                # 在居民区附近但不在正中心
                x = np.random.normal(base_x, self.city_width * 0.08)
                y = np.random.normal(base_y, self.city_height * 0.08)
                x = np.clip(x, self.city_width * 0.05, self.city_width * 0.95)
                y = np.clip(y, self.city_height * 0.05, self.city_height * 0.95)

                station = AmbulanceStation(
                    id=f'AS{i + 1}',
                    position=(x, y),
                    ambulance_count=np.random.randint(2, 8),
                    coverage_radius=np.random.uniform(8, 15)
                )
                stations.append(station)

        while len(stations) < n:
            region = self._get_region_for_facility()
            x, y = self._get_position_in_region(region)

            station = AmbulanceStation(
                id=f'AS{len(stations) + 1}',
                position=(x, y),
                ambulance_count=np.random.randint(2, 6),
                coverage_radius=np.random.uniform(8, 15)
            )
            stations.append(station)


        return stations

    def _gen_residential_areas(self, n: int) -> List[ResidentialArea]:
        """gen residential areas location"""
        areas = []
        area_types = [
            {'type': 'high_density', 'weight': 0.3, 'density_range': (8, 15), 'size_range': (0.5, 2.0)},
            {'type': 'medium_density', 'weight': 0.5, 'density_range': (4, 8), 'size_range': (1.0, 3.0)},
            {'type': 'low_density', 'weight': 0.2, 'density_range': (1, 4), 'size_range': (2.0, 5.0)}
        ]

        type_counts = self._distribute_by_weights(n, [t['weight'] for t in area_types])

        area_id = 0
        for type_idx, area_type in enumerate(area_types):
            count = type_counts[type_idx]

            for _ in range(count):
                if self.map_type == MapType.RING:
                    if area_type['type'] == 'high_density':
                        distance = np.random.beta(2, 5) * min(self.city_width, self.city_height) * 0.4
                    elif area_type['type'] == 'medium_density':
                        distance = np.random.uniform(0.3, 0.6) * min(self.city_width, self.city_height)
                    else:
                        distance = np.random.uniform(0.6, 0.9) * min(self.city_width, self.city_height)
                    angle = np.random.uniform(0, 2 * np.pi)
                    x = self.city_center[0] + distance * math.cos(angle)
                    y = self.city_center[1] + distance * math.sin(angle)

                else:
                    region = self._get_region_for_residential(area_type['type'])
                    x, y = self._get_position_in_region(region)

                density = np.random.uniform(area_type['density_range'][0],
                                            area_type['density_range'][1])
                area_size = np.random.uniform(area_type['size_range'][0],
                                              area_type['size_range'][1])
                population = int(density * area_size * 1000)

                residential_area = ResidentialArea(
                    id=f'R{area_id + 1}',
                    name=f'{area_type["type"]}_area{area_id + 1}',
                    position=(x, y),
                    population=population,
                    density=density,
                    area=area_size,
                    area_type=area_type['type']
                )
                areas.append(residential_area)
                area_id += 1

        return areas


    def _gen_grid_road_network(self, n: int) -> List[Dict]:
        """gen road network"""
        print("Generating grid road network...")
        roads = []

        important_nodes = [hosp.position for hosp in self.hospitals]
        if self.residential_areas:
            high_density_areas = [area for area in self.residential_areas if area.area_type == 'high_density']
            important_nodes.extend([area.position for area in high_density_areas[:3]])

        arterial_roads_count = min(n // 2, len(important_nodes) - 1)
        for i in range(arterial_roads_count):
            start_idx = i
            end_idx = (i + 1) % len(important_nodes)

            roads.append({
                'id': f'R_Arterial{i + 1}',
                'type': 'arterial',
                'points': [important_nodes[start_idx], important_nodes[end_idx]],
                'speed_limit': 60,
                'capacity': 1000
            })

        n_grid = n - len(roads)
        horizontal_roads = n_grid // 2
        vertical_roads = n_grid - horizontal_roads

        for i in range(horizontal_roads):
            y = self.city_height * (i + 1) / (horizontal_roads + 1)
            roads.append({
                'id': f'R_H{i + 1}',
                'type': 'local',
                'points': [(0, y), (self.city_width, y)],
                'speed_limit': 40,
                'capacity': 500
            })

        for i in range(vertical_roads):
            x = self.city_width * (i + 1) / (vertical_roads + 1)
            roads.append({
                'id': f'R_V{i + 1}',
                'type': 'local',
                'points': [(x, 0), (x, self.city_height)],
                'speed_limit': 40,
                'capacity': 500
            })

        return roads

    def _gen_ring_road_network(self, n: int) -> List[Dict]:
        print("Generating ring road network...")
        roads = []

        population_center = self._calculate_population_center()
        center_x, center_y = population_center
        distances_to_boundary = [
            center_x,
            self.city_width - center_x,
            center_y,
            self.city_height - center_y
        ]
        max_radius = min(distances_to_boundary) * 0.85
        n_rings = min(7, max(4, n // 3))
        ring_radii = []

        for i in range(n_rings):
            alpha = self.ring_density_factor

            if n_rings > 1:
                normalized_position = i / (n_rings - 1)
                radius = max_radius * (normalized_position ** alpha)
                if i >= n_rings - 2:
                    adjustment = 1.0 + (0.1 * (i - (n_rings - 2)))
                    radius = min(radius * adjustment, max_radius * 0.95)
            else:
                radius = max_radius * 0.5

            min_radius = max_radius * 0.1
            radius = max(min_radius, radius)
            ring_radii.append(radius)

        ring_radii.sort()

        if ring_radii:
            ring_radii[-1] = max(ring_radii[-1], max_radius * 0.9)

        print(f"Generated {len(ring_radii)} ring roads, max radius: {max(ring_radii):.1f} km")
        print(f"Edge rings (last 2): {ring_radii[-2]:.1f} km, {ring_radii[-1]:.1f} km")

        for i, radius in enumerate(ring_radii):
            is_edge_ring = i >= len(ring_radii) - 2
            if radius < max_radius * 0.3:
                road_type = 'arterial'
                speed_limit = 60
                capacity = 1200
                line_width = 2.5
                color = '#e74c3c'
                alpha = 0.8
                label = 'Core Ring Road'
            elif radius < max_radius * 0.6:
                road_type = 'collector'
                speed_limit = 50
                capacity = 800
                line_width = 2.0
                color = '#f39c12'
                alpha = 0.7
                label = 'Middle Ring Road'
            else:
                road_type = 'local'
                speed_limit = 40
                capacity = 600 if is_edge_ring else 500
                line_width = 1.8 if is_edge_ring else 1.5
                color = '#3498db' if is_edge_ring else '#7f8c8d'
                alpha = 0.7 if is_edge_ring else 0.6
                label = 'Edge Ring Road' if is_edge_ring else 'Outer Ring Road'

            circle_points = []
            n_segments = max(24, int(32 * (1 - radius / max_radius) + 16))

            for j in range(n_segments + 1):
                angle = 2 * math.pi * j / n_segments
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                # 确保点在城市边界内
                x = max(0, min(self.city_width, x))
                y = max(0, min(self.city_height, y))
                circle_points.append((x, y))

            is_truncated = False
            boundary_points = 0
            for point in circle_points:
                x, y = point
                if x == 0 or x == self.city_width or y == 0 or y == self.city_height:
                    boundary_points += 1

            if boundary_points > n_segments * 0.1:
                is_truncated = True

            roads.append({
                'id': f'R_Ring{i + 1}',
                'type': road_type,
                'points': circle_points,
                'speed_limit': speed_limit,
                'capacity': capacity,
                'is_circular': True,
                'radius': radius,
                'is_edge_ring': is_edge_ring,
                'is_truncated': is_truncated,
                'visual_params': {
                    'color': color,
                    'linewidth': line_width,
                    'alpha': alpha,
                    'label': label
                }
            })

        n_radials = n - len(roads)
        radial_angles = []

        for i in range(n_radials):
            base_angle = 2 * math.pi * i / n_radials

            if i < n_radials * 0.3:
                angle_variation = np.random.uniform(-0.05, 0.05) * 2 * math.pi
            else:
                angle_variation = np.random.uniform(-0.15, 0.15) * 2 * math.pi

            angle = base_angle + angle_variation
            radial_angles.append(angle % (2 * math.pi))
        radial_angles.sort()

        for i, angle in enumerate(radial_angles):
            end_x, end_y = self._find_boundary_intersection(center_x, center_y, angle)
            intersects_core = False
            for ring_radius in ring_radii:
                if ring_radius < max_radius * 0.4:
                    if i < n_radials * 0.4:
                        intersects_core = True
                        break

            is_edge_radial = False
            if i > n_radials * 0.7:
                is_edge_radial = True

            if intersects_core:
                road_type = 'arterial'
                speed_limit = 60
                capacity = 1000
                line_width = 2.0
                color = '#e74c3c'
                alpha = 0.7
                label = 'Core Radial Road'
            elif is_edge_radial:
                road_type = 'local'
                speed_limit = 45
                capacity = 600
                line_width = 1.5
                color = '#3498db'
                alpha = 0.6
                label = 'Edge Radial Road'
            else:
                road_type = 'local'
                speed_limit = 50
                capacity = 600
                line_width = 1.5
                color = '#7f8c8d'
                alpha = 0.6
                label = 'Radial Road'

            roads.append({
                'id': f'R_Radial{i + 1}',
                'type': road_type,
                'points': [(center_x, center_y), (end_x, end_y)],
                'speed_limit': speed_limit,
                'capacity': capacity,
                'is_radial': True,
                'angle': angle,
                'is_edge_radial': is_edge_radial,
                'visual_params': {
                    'color': color,
                    'linewidth': line_width,
                    'alpha': alpha,
                    'label': label
                }
            })

        n_connectors = min(6, max(0, n - len(roads)))

        if n_connectors > 0 and len(ring_radii) >= 2:
            for i in range(n_connectors):
                if i < n_connectors // 2:
                    ring_idx1 = len(ring_radii) - 2
                    ring_idx2 = len(ring_radii) - 1
                else:
                    # 随机选择两个环形道路
                    ring_idx1 = np.random.randint(0, len(ring_radii) - 1)
                    ring_idx2 = ring_idx1 + 1

                radius1 = ring_radii[ring_idx1]
                radius2 = ring_radii[ring_idx2]
                angle = np.random.uniform(0, 2 * math.pi)

                point1 = (
                    center_x + radius1 * math.cos(angle),
                    center_y + radius1 * math.sin(angle)
                )
                point2 = (
                    center_x + radius2 * math.cos(angle),
                    center_y + radius2 * math.sin(angle)
                )
                point1 = (max(0, min(self.city_width, point1[0])),
                          max(0, min(self.city_height, point1[1])))
                point2 = (max(0, min(self.city_width, point2[0])),
                          max(0, min(self.city_height, point2[1])))
                is_edge_connector = i < n_connectors // 2

                roads.append({
                    'id': f'R_Conn{i + 1}',
                    'type': 'local',
                    'points': [point1, point2],
                    'speed_limit': 40,
                    'capacity': 400,
                    'is_connector': True,
                    'is_edge_connector': is_edge_connector,
                    'visual_params': {
                        'color': '#3498db' if is_edge_connector else '#95a5a6',
                        'linewidth': 1.0,
                        'alpha': 0.5,
                        'label': 'Edge Connector' if is_edge_connector else 'Connector Road'
                    }
                })

        print(f"Generated {len(roads)} roads: {len(ring_radii)} rings "
              f"(including {len([r for r in roads if r.get('is_edge_ring', False)])} edge rings), "
              f"{n_radials} radials, {n_connectors} connectors")
        self._analyze_road_density(roads, center_x, center_y, max_radius)
        self._analyze_edge_rings(roads, center_x, center_y, min(distances_to_boundary))

        return roads

    def _gen_truncated_circle(self, center_x: float, center_y: float,
                              radius: float, width: float, height: float) -> List[Tuple[float, float]]:
        n_segments = 64
        full_circle_points = []

        for i in range(n_segments + 1):
            angle = 2 * math.pi * i / n_segments
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            full_circle_points.append((x, y))

        clipped_points = []
        for point in full_circle_points:
            x, y = point
            if 0 <= x <= width and 0 <= y <= height:
                clipped_points.append(point)

        if len(clipped_points) < 2:
            return []

        angles = []
        for point in clipped_points:
            x, y = point
            angle = math.atan2(y - center_y, x - center_x)
            if angle < 0:
                angle += 2 * math.pi
            angles.append(angle)

        sorted_indices = np.argsort(angles)
        sorted_points = [clipped_points[i] for i in sorted_indices]
        result_points = []

        for i in range(len(sorted_points)):
            result_points.append(sorted_points[i])

            if i < len(sorted_points) - 1:
                angle_diff = angles[sorted_indices[i + 1]] - angles[sorted_indices[i]]
            else:
                angle_diff = (2 * math.pi + angles[sorted_indices[0]]) - angles[sorted_indices[-1]]

            if angle_diff > math.pi / 2:
                pass

        if len(result_points) < 4:
            return []

        return result_points

    def _analyze_road_density(self, roads: List[Dict], center_x: float, center_y: float, max_radius: float):
        print("=== analyze road density ===")
        n_zones = 5
        zone_radii = [max_radius * (i+1) / n_zones for i in range(n_zones)]
        zone_lengths = [0.0] * n_zones
        zone_counts = [0] * n_zones

        for road in roads:
            points = road['points']
            avg_radius = 0.0
            for point in points:
                distance = math.sqrt((point[0] - center_x) ** 2 + (point[1] - center_y) ** 2)
                avg_radius += distance
            avg_radius /= len(points)

            road_length = 0.0
            for i in range(len(points) - 1):
                dx = points[i + 1][0] - points[i][0]
                dy = points[i + 1][1] - points[i][1]
                road_length += math.sqrt(dx * dx + dy * dy)

            for zone_idx in range(n_zones):
                if avg_radius <= zone_radii[zone_idx]:
                    zone_lengths[zone_idx] += road_length
                    zone_counts[zone_idx] += 1
                    break

        total_city_area = self.city_width * self.city_height
        print(f"total: {total_city_area:.1f} km²")
        print(f"max r: {max_radius:.1f} km")

        prev_radius = 0.0
        for zone_idx in range(n_zones):
            zone_radius = zone_radii[zone_idx]
            zone_area = math.pi * (zone_radius ** 2 - prev_radius ** 2)

            if zone_area > 0:
                road_density = zone_lengths[zone_idx] / zone_area  # 单位面积道路长度
            else:
                road_density = 0.0

            print(f"zone {zone_idx + 1} (radius {prev_radius:.1f}-{zone_radius:.1f} km): "
                  f"{zone_area:.1f} km², len={zone_lengths[zone_idx]:.1f} km, "
                  f"density={road_density:.4f} km/km², counts={zone_counts[zone_idx]}")

            prev_radius = zone_radius

    def _get_region_for_facility(self) -> str:
        regions = ["city_center", "suburban", "rural"]
        weights = [0.4, 0.4, 0.2]
        return random.choices(regions, weights=weights)[0]

    def _get_region_for_residential(self, area_type: str) -> str:
        if area_type == 'high_density':
            return 'city_center'
        elif area_type == 'medium_density':
            return random.choices(["city_center", "suburban"], weights=[0.3, 0.7])[0]
        else:
            return 'rural'

    def _get_position_in_region(self, region: str) -> Tuple[float, float]:
        if region == 'city_center':
            x = np.random.normal(self.city_width * 0.5, self.city_width * 0.1)
            y = np.random.normal(self.city_height * 0.5, self.city_height * 0.1)
        elif region == 'suburban':
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0.2, 0.4)
            x = 0.5 + distance * np.cos(angle) * 0.5
            y = 0.5 + distance * np.sin(angle) * 0.5
            x *= self.city_width
            y *= self.city_height
        else:
            corner = random.choice([(0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)])
            x = np.random.normal(corner[0], 0.1) * self.city_width
            y = np.random.normal(corner[1], 0.1) * self.city_height

        x = np.clip(x, self.city_width * 0.02, self.city_width * 0.98)
        y = np.clip(y, self.city_height * 0.02, self.city_height * 0.98)
        return x, y

    def _distribute_by_weights(self, total: int, weights: List[float]) -> List[int]:
        counts = [int(total * weight) for weight in weights]
        remaining = total - sum(counts)
        if remaining > 0:
            max_idx = weights.index(max(weights))
            counts[max_idx] += remaining
        return counts

    def save_city_layout(self):
        output_dir = "simulation_results"
        os.makedirs(output_dir, exist_ok=True)
        layout_data = {
            'simulation_id': self.formatted_time,
            'city_dimensions': {
                'width': self.city_width,
                'height': self.city_height
            },
            'map_type': self.map_type.value,
            'ring_density_factor': self.ring_density_factor if self.map_type == MapType.RING else None,
            'hospitals': [
                {
                    'id': hosp.id,
                    'name': hosp.name,
                    'position': hosp.position,
                    'capacity': hosp.capacity,
                    'type': hosp.hospital_type.value,
                    'emergency_beds': hosp.emergency_beds,
                    'strategy': hosp.strategy
                } for hosp in self.hospitals
            ],
            'ambulance_stations': [
                {
                    'id': station.id,
                    'position': station.position,
                    'ambulance_count': station.ambulance_count,
                    'coverage_radius': station.coverage_radius
                } for station in self.ambulance_stations
            ],
            'residential_areas': [
                {
                    'id': area.id,
                    'name': area.name,
                    'position': area.position,
                    'population': area.population,
                    'density': area.density,
                    'area': area.area,
                    'type': area.area_type
                } for area in self.residential_areas
            ],
            'roads': self.roads,
            'timestamp': datetime.now().isoformat()
        }
        filename = os.path.join(output_dir,  f"{self.formatted_time}_city_layout.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, ensure_ascii=False, indent=2)

        print(f"city layout saved to {filename}")

    def visualize_city_layout(self, show_voronoi: bool = True, show_emergencies: bool = True, save_plot: bool = True):
        fig, ax = plt.subplots(1, 1, figsize=(15, 12))
        ax.set_facecolor('#f0f0f0')
        ax.set_xlim(0, self.city_width)
        ax.set_ylim(0, self.city_height)
        ax.set_xlabel('EW Distance (km)')
        ax.set_ylabel('SN Distance (km)')
        title = f'City Layout - {self.map_type.value.upper()} Planning'
        if self.map_type == MapType.RING:
            title += f' (Density Factor: {self.ring_density_factor})'
            has_edge_rings = any(road.get('is_edge_ring', False) for road in self.roads)
            if has_edge_rings:
                title += ' - With Edge Ring Roads'
        title += ' - Ambulance Dispatch Simulation'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        # roads
        road_legend_added = {'highway': False, 'arterial': False, 'local': False}
        for road in self.roads:
            points = np.array(road['points'])

            if self.map_type == MapType.RING:
                legend_added = {
                    'core_ring': False, 'middle_ring': False, 'edge_ring': False,
                    'core_radial': False, 'radial': False, 'edge_radial': False,
                    'connector': False, 'edge_connector': False
                }
                for road in self.roads:
                    points = np.array(road['points'])
                    visual_params = road.get('visual_params', {})
                    if len(points) < 2:
                        continue

                    if road.get('is_circular', False):
                        if road.get('is_edge_ring', False):
                            label = 'Edge Ring Road' if not legend_added['edge_ring'] else ""
                            legend_added['edge_ring'] = True
                        elif visual_params.get('label', '').startswith('Core'):
                            label = 'Center Ring Road' if not legend_added['core_ring'] else ""
                            legend_added['core_ring'] = True
                        elif visual_params.get('label', '').startswith('Middle'):
                            label = 'Middle Ring Road' if not legend_added['middle_ring'] else ""
                            legend_added['middle_ring'] = True
                        else:
                            label = 'Side Ring Road' if not legend_added.get('outer_ring', False) else ""
                            legend_added['outer_ring'] = True

                        line_style = {
                            'color': visual_params.get('color', '#e74c3c'),
                            'linewidth': visual_params.get('linewidth', 2.5),
                            'alpha': visual_params.get('alpha', 0.7),
                            'linestyle': '-',
                            'label': label
                        }
                        ax.plot(points[:, 0], points[:, 1], **line_style)
                        if road.get('is_edge_ring', False) and road.get('is_truncated', False) and len(points) > 2:
                            ax.scatter(points[0, 0], points[0, 1], s=40, c='#e74c3c', alpha=0.8, zorder=5, marker='s')
                            ax.scatter(points[-1, 0], points[-1, 1], s=40, c='#e74c3c', alpha=0.8, zorder=5, marker='s')

                    elif road.get('is_radial', False):
                        if road.get('is_edge_radial', False):
                            label = 'Edge Radial Road' if not legend_added['edge_radial'] else ""
                            legend_added['edge_radial'] = True
                        elif visual_params.get('label', '').startswith('Core'):
                            label = 'Core Radial Road' if not legend_added['core_radial'] else ""
                            legend_added['core_radial'] = True
                        else:
                            label = 'Radial Road' if not legend_added['radial'] else ""
                            legend_added['radial'] = True

                        line_style = {
                            'color': visual_params.get('color', '#7f8c8d'),
                            'linewidth': visual_params.get('linewidth', 1.5),
                            'alpha': visual_params.get('alpha', 0.6),
                            'linestyle': '--',
                            'label': label
                        }
                        ax.plot(points[:, 0], points[:, 1], **line_style)

                    elif road.get('is_connector', False):
                        if road.get('is_edge_connector', False):
                            label = 'Edge Connector' if not legend_added['edge_connector'] else ""
                            legend_added['edge_connector'] = True
                        else:
                            label = 'Connector Road' if not legend_added['connector'] else ""
                            legend_added['connector'] = True

                        line_style = {
                            'color': visual_params.get('color', '#95a5a6'),
                            'linewidth': visual_params.get('linewidth', 1.0),
                            'alpha': visual_params.get('alpha', 0.5),
                            'linestyle': ':',
                            'label': label
                        }
                        ax.plot(points[:, 0], points[:, 1], **line_style)

                    else:
                        line_style = {
                            'color': '#7f8c8d',
                            'linewidth': 1.5,
                            'alpha': 0.6,
                            'linestyle': '-',
                            'label': 'Local Road' if not legend_added.get('other', False) else ""
                        }
                        legend_added['other'] = True
                        ax.plot(points[:, 0], points[:, 1], **line_style)

                # if road.get('is_circular', False):
                #     center_x = sum(p[0] for p in points) / len(points)
                #     center_y = sum(p[1] for p in points) / len(points)
                #     radius = np.sqrt((points[0][0] - center_x) ** 2 + (points[0][1] - center_y) ** 2)
                #
                #     circle = patches.Circle((center_x, center_y), radius,
                #                             fill=False, linestyle='-', linewidth=2.5,
                #                             alpha=0.7, color='#e74c3c',
                #                             label='Ring Road' if not road_legend_added['arterial'] else "")
                #     ax.add_patch(circle)
                #     road_legend_added['arterial'] = True
                # elif road.get('is_radial', False):
                #     line_style = {'color': '#7f8c8d', 'linewidth': 1.5, 'alpha': 0.6,
                #                   'linestyle': '--', 'label': 'Radial Road' if not road_legend_added['local'] else ""}
                #     road_legend_added['local'] = True
                #     ax.plot(points[:, 0], points[:, 1], **line_style)
                # else:
                #     line_style = {'color': '#7f8c8d', 'linewidth': 1.5, 'alpha': 0.6,
                #                   'linestyle': '-', 'label': 'Local Road' if not road_legend_added['local'] else ""}
                #     road_legend_added['local'] = True
                #     ax.plot(points[:, 0], points[:, 1], **line_style)
                edge_buffer = patches.Rectangle((0, 0), self.city_width, self.city_height,
                                                fill=False, linestyle='--', linewidth=2,
                                                edgecolor='#2c3e50', alpha=0.5, label='City Boundary')
                ax.add_patch(edge_buffer)
                center_x, center_y = self.city_center
                distances_to_boundary = [
                    center_x,
                    self.city_width - center_x,
                    center_y,
                    self.city_height - center_y
                ]
                min_distance_to_boundary = min(distances_to_boundary)
                edge_rings = [road for road in self.roads if road.get('is_edge_ring', False)]

                if edge_rings:
                    max_edge_radius = max(road.get('radius', 0) for road in edge_rings)

                    edge_circle = patches.Circle((center_x, center_y), max_edge_radius,
                                                 fill=False, linestyle=':', linewidth=1.5,
                                                 edgecolor='#3498db', alpha=0.7)
                    ax.add_patch(edge_circle)
                    ax.text(center_x, center_y + max_edge_radius + 2,
                            f'Edge Ring Road Zone',
                            fontsize=9, color='#3498db', alpha=0.8,
                            horizontalalignment='center',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#3498db", alpha=0.8))

            else:
                if road['type'] == 'highway':
                    line_style = {'color': '#f39c12', 'linewidth': 4, 'alpha': 0.8,
                                  'linestyle': '-', 'label': 'Highway' if not road_legend_added['highway'] else ""}
                    road_legend_added['highway'] = True
                elif road['type'] == 'arterial':
                    line_style = {'color': '#e74c3c', 'linewidth': 2.5, 'alpha': 0.7,
                                  'linestyle': '-', 'label': 'Arterial Road' if not road_legend_added['arterial'] else ""}
                    road_legend_added['arterial'] = True
                else:  # local
                    line_style = {'color': '#7f8c8d', 'linewidth': 1.5, 'alpha': 0.6,
                                  'linestyle': '--', 'label': 'Local Road' if not road_legend_added['local'] else ""}
                    road_legend_added['local'] = True
                ax.plot(points[:, 0], points[:, 1], **line_style)

        # residential areas
        if self.residential_areas:
            for area in self.residential_areas:
                x, y = area.position

                if 'high' in area.area_type:
                    color = '#e74c3c'
                    size = 100 + (area.population / 5000) * 200
                    label = 'High Density'
                elif 'medium' in area.area_type:
                    color = '#f39c12'
                    size = 80 + (area.population / 5000) * 150
                    label = 'Medium Density'
                else:
                    color = '#27ae60'
                    size = 60 + (area.population / 5000) * 100
                    label = 'Low Density'

                ax.scatter(x, y, s=size, c=color, alpha=0.7,
                           edgecolors='darkblue', linewidth=1)

        # hospitals
        for hospital in self.hospitals:
            x, y = hospital.position

            if hospital.hospital_type == HospitalType.GENERAL:
                marker, size, color = '^', 400, '#c0392b'
                label = 'General Hospital'
            else:
                marker, size, color = 's', 300, '#e74c3c'
                label = 'Community Hospital'

            ax.scatter(x, y, s=size, marker=marker, c=color,
                       edgecolors='darkred', linewidth=3, label=label)

            ax.annotate(f"{hospital.name}",
                        (x, y), xytext=(5, 15), textcoords='offset points',
                        fontsize=9, fontweight='bold', color=color,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.9))

        # ambulance stations
        for station in self.ambulance_stations:
            x, y = station.position
            ax.scatter(x, y, s=200, marker='s', c='#27ae60', edgecolors='#229954',
                       linewidth=2, label='Ambulance Station')

            if show_voronoi:
                circle = patches.Circle((x, y), station.coverage_radius,
                                        fill=True, linestyle='--', alpha=0.15, color='#27ae60')
                ax.add_patch(circle)

        if self.map_type == MapType.RING:
            legend_elements = [
                Line2D([0], [0], marker='^', color='w', markerfacecolor='#c0392b',
                       markersize=12, label='General Hospital', markeredgecolor='darkred'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='#e74c3c',
                       markersize=10, label='Community Hospital', markeredgecolor='darkred'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='#27ae60',
                       markersize=10, label='Ambulance Station', markeredgecolor='#229954'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
                       markersize=10, label='High Density Area', alpha=0.7),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12',
                       markersize=8, label='Medium Density Area', alpha=0.7),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60',
                       markersize=6, label='Low Density Area', alpha=0.7),
                Line2D([0], [0], color='#e74c3c', linewidth=3, label='Core Ring Road'),
                Line2D([0], [0], color='#f39c12', linewidth=2.5, label='Middle Ring Road'),
                Line2D([0], [0], color='#3498db', linewidth=2, label='Edge Ring Road'),
                Line2D([0], [0], color='#e74c3c', linewidth=2, linestyle='--', label='Core Radial Road'),
                Line2D([0], [0], color='#7f8c8d', linewidth=1.5, linestyle='--', label='Radial Road'),
                Line2D([0], [0], color='#3498db', linewidth=1.5, linestyle='--', label='Edge Radial Road'),
                Line2D([0], [0], color='#95a5a6', linewidth=1, linestyle=':', label='Connector Road'),
                Line2D([0], [0], color='#3498db', linewidth=1, linestyle=':', label='Edge Connector'),
                Line2D([0], [0], color='#2c3e50', linewidth=2, linestyle='--', label='City Boundary'),
            ]
        else:
            legend_elements = [
                Line2D([0], [0], marker='^', color='w', markerfacecolor='#c0392b',
                       markersize=12, label='General Hospital', markeredgecolor='darkred'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='#e74c3c',
                       markersize=10, label='Community Hospital', markeredgecolor='darkred'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='#27ae60',
                       markersize=10, label='Ambulance Station', markeredgecolor='#229954'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
                       markersize=10, label='High Density Area', alpha=0.7),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12',
                       markersize=8, label='Medium Density Area', alpha=0.7),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60',
                       markersize=6, label='Low Density Area', alpha=0.7),
                Line2D([0], [0], color='#e74c3c', linewidth=3, label='Arterial Road'),
                Line2D([0], [0], color='#7f8c8d', linewidth=2, linestyle='--', label='Local Road'),
            ]

        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))

        # statistics
        if self.map_type == MapType.RING:
            ring_count = sum(1 for road in self.roads if road.get('is_circular', False))
            edge_ring_count = sum(1 for road in self.roads if road.get('is_edge_ring', False))
            truncated_count = sum(1 for road in self.roads if road.get('is_truncated', False))
            max_ring_radius = 0

            for road in self.roads:
                if road.get('is_circular', False) and 'radius' in road:
                    max_ring_radius = max(max_ring_radius, road['radius'])

            center_x, center_y = self.city_center
            distances_to_boundary = [
                center_x,
                self.city_width - center_x,
                center_y,
                self.city_height - center_y
            ]
            min_distance_to_boundary = min(distances_to_boundary)

            if max_ring_radius > 0:
                distance_to_boundary = min_distance_to_boundary - max_ring_radius

                ring_info = f"• Total Ring Roads: {ring_count}\n"
                ring_info += f"• Edge Ring Roads: {edge_ring_count}\n"
                ring_info += f"• Truncated Rings: {truncated_count}\n"
                ring_info += f"• Max Ring Radius: {max_ring_radius:.1f} km\n"
                ring_info += f"• Distance to Boundary: {distance_to_boundary:.1f} km\n"
                ring_info += f"• Edge zones have ring roads ✓"
            else:
                ring_info = "• No ring roads generated"
        else:
            ring_info = "• Grid Layout: Complete Road Network"

        stats_text = f"""
                    City Statistics:
                    • Layout Type: {self.map_type.value.upper()}
                    • Area: {self.city_width * self.city_height:.0f} km²
                    • Hospitals: {len(self.hospitals)}
                    • Ambulance Stations: {len(self.ambulance_stations)}
                    • Residential Areas: {len(self.residential_areas)}
                    • Total Population: {sum(area.population for area in self.residential_areas):,}
                    • Total Ambulances: {sum(station.ambulance_count for station in self.ambulance_stations)}
                    • Total Roads: {len(self.roads)}
                    {ring_info}
                    """

        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8))

        ax.grid(True, alpha=0.2)
        plt.tight_layout()

        if save_plot:
            output_dir = "simulation_results"
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{self.formatted_time}_city_layout.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"city layout saved to: {filename}")

        return fig, ax

    def _calculate_population_center(self) -> Tuple[float, float]:
        if not self.residential_areas:
            return self.city_center

        total_population = sum(area.population for area in self.residential_areas)
        if total_population == 0:
            return self.city_center

        weighted_x = sum(area.position[0] * area.population for area in self.residential_areas) / total_population
        weighted_y = sum(area.position[1] * area.population for area in self.residential_areas) / total_population

        return weighted_x, weighted_y

    def _find_boundary_intersection(self, center_x: float, center_y: float, angle: float) -> Tuple[float, float]:
        intersections = []

        if math.cos(angle) < 0:  # left
            t = -center_x/math.cos(angle)
            y = center_y + t * math.sin(angle)
            if 0 <= y <= self.city_height:
                intersections.append((0, y))

        if math.cos(angle) > 0:  # right
            t = (self.city_width - center_x) / math.cos(angle)
            y = center_y + t * math.sin(angle)
            if 0 <= y <= self.city_height:
                intersections.append((self.city_width, y))

        if math.sin(angle) < 0:  # bottom
            t = -center_y/math.sin(angle)
            x = center_x + t * math.cos(angle)
            if 0 <= x <= self.city_width:
                intersections.append((x, 0))

        if math.sin(angle) > 0:  # top
            t = (self.city_height - center_y) / math.sin(angle)
            x = center_y + t * math.cos(angle)
            if 0 <= x <= self.city_width:
                intersections.append((x, self.city_height))

        if not intersections:
            max_distance = max(self.city_width, self.city_height)
            x = center_x + max_distance * math.cos(angle)
            y = center_y + max_distance * math.sin(angle)

            x = max(0, min(self.city_width, x))
            y = max(0, min(self.city_height, y))
            return x, y

        distances = [(x - center_x)**2 + (y - center_y)**2 for x, y in intersections]
        return intersections[np.argmin(distances)]

    def _gen_ring_road_network_old(self, n: int) -> List[Dict]:
        print("Generating old ring road network...")
        roads = []

        population_center = self._calculate_population_center()
        center_x, center_y = population_center

        n_rings = min(4, n // 3)
        n_radials = n - n_rings

        max_radius = min(self.city_width, self.city_height) * 0.45

        for i in range(n_rings):
            radius = max_radius * (i + 1) / (n_rings + 1)

            circle_points = []
            n_segments = 32
            for j in range(n_segments + 1):
                angle = 2 * math.pi * j / n_segments
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                x = max(0, min(self.city_width, x))
                y = max(0, min(self.city_height, y))
                circle_points.append((x, y))

            roads.append({
                'id': f'R_Ring{i + 1}',
                'type': 'arterial',
                'points': circle_points,
                'speed_limit': 60,
                'capacity': 1000,
                'is_circular': True
            })

        for i in range(n_radials):
            angle = 2 * math.pi * i / n_radials
            end_x, end_y = self._find_boundary_intersection(center_x, center_y, angle)

            roads.append({
                'id': f'R_Radial{i + 1}',
                'type': 'local',
                'points': [(center_x, center_y), (end_x, end_y)],
                'speed_limit': 50,
                'capacity': 700,
                'is_radial': True
            })

        return roads

    def _analyze_edge_rings(self, roads: List[Dict], center_x: float, center_y: float, city_radius: float):
        print("\n=== Edge Ring Road Analysis ===")
        edge_rings = [road for road in roads if road.get('is_edge_ring', False)]

        if not edge_rings:
            print("No edge ring roads found")
            return
        print(f"Edge ring roads: {len(edge_rings)}")

        for i, road in enumerate(edge_rings):
            radius = road.get('radius', 0)
            distance_to_boundary = city_radius - radius

            print(f"Edge Ring {i + 1}: radius={radius:.1f} km, "
                  f"distance to boundary={distance_to_boundary:.1f} km, "
                  f"truncated={road.get('is_truncated', False)}")

        if edge_rings:
            max_edge_radius = max(road.get('radius', 0) for road in edge_rings)
            coverage_ratio = max_edge_radius / city_radius * 100
            print(f"Edge ring coverage: {coverage_ratio:.1f}% of city radius")
            print(f"Edge zones have ring roads ")