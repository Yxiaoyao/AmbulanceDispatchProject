import json
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
# from scipy.spatial import Voronoi, voronoi_plot_2d
import random
from typing import List, Dict, Tuple
from param import Hospital, AmbulanceStation, ResidentialArea, Emergency, HospitalType


class AmbulanceSimulation:
    """
    救护车仿真地图
    """

    def __init__(self, city_width: float = 100, city_height: float = 100, random_seed: int = 42):
        """
        :param city_width: 城市宽度
        :param city_height: 城市长度
        """
        self.city_width = city_width
        self.city_height = city_height
        self.hospitals: List[Hospital] = []  # 医院
        self.ambulance_stations: List[AmbulanceStation] = []  # 救护站
        self.residential_areas: List[ResidentialArea] = []  # 居民区
        self.roads: List[dict] = []
        self.emergencies: List[Emergency] = []
        self.event_queue = []
        self.current_time = 0
        self.performance_metrics = {}
        self.formatted_time = datetime.now().strftime("%Y%m%d%H%M%S")

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
        print("gen roads")
        self.roads = self._gen_road_network(n_major_roads)

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
                x, y =self.city_width * 0.5, self.city_height * 0.5
                hospital_type = HospitalType.GENERAL
                capacity = np.random.randint(200, 350)
                emergency_beds = capacity // 3
                strategy = "cooperative"
            else:
                # 基于区域
                region = self._get_region_for_facility()
                x, y = self._get_position_in_region(region)

                if region == "city_center" or random.random() < 0.4:
                    hospital_type = HospitalType.GENERAL
                    capacity = np.random.randint(100, 250)
                else:
                    hospital_type = HospitalType.COMMUNITY
                    capacity = np.random.randint(50, 150)

                emergency_beds = max(10, capacity // 4)
                strategy = random.choices(strategies, weights=strategy_weights)[0]

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


    def _gen_road_network(self, n: int) -> List[Dict]:
        """gen road network"""
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
        ax.set_title('City Layout - Ambulance Dispatch Simulation', fontsize=16, fontweight='bold', pad=20)

        # roads
        road_legend_added = {'highway': False, 'arterial': False, 'local': False}
        for road in self.roads:
            points = np.array(road['points'])
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
        stats_text = f"""
                City Statistics:
                • Area: {self.city_width * self.city_height:.0f} km²
                • Hospitals: {len(self.hospitals)}
                • Ambulance Stations: {len(self.ambulance_stations)}
                • Residential Areas: {len(self.residential_areas)}
                • Total Population: {sum(area.population for area in self.residential_areas):,}
                • Total Ambulances: {sum(station.ambulance_count for station in self.ambulance_stations)}
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
