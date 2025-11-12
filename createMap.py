import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from scipy.spatial import Voronoi, voronoi_plot_2d
import geopandas as gpd
from shapely.geometry import Point, Polygon
import random
from typing import List, Dict, Tuple
import pandas as pd

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


        np.random.seed(random_seed)

    def gen_city_layout(self,
                        n_hospitals: int = 4,
                        n_stations: int = 8,
                        n_residential_areas: int = 15,
                        n_major_roads: int = 6):
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
                region = self._gen_region_for_facility()
                x, y = self._gen_position_in_region(region)

                if region == "city_center" or random.random() < 0.4:
                    hospital_type = HospitalType.GENERAL
                    capacity = np.random.randint(100, 250)
                else:
                    hospital_type = HospitalType.COMMUNITY
                    capacity = np.random.randint(50, 150)

                emergency_beds = max(10, capacity // 4)
                strategy = random.choice(strategies, weights=strategy_weights)[0]

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

        # hospital_types = [
        #     {'name': 'general_hospital ', 'capacity_range': (100, 300), 'weight': 0.4},
        #     {'name': 'community_hospital', 'capacity_range': (50, 150), 'weight': 0.6}
        # ]
        # downtown_center = (self.city_width * 0.5, self.city_height * 0.5)
        #
        # # first
        # x = np.random.normal(downtown_center[0], self.city_width * 0.1)
        # y = np.random.normal(downtown_center[1], self.city_height * 0.1)
        # x = np.clip(x, self.city_width * 0.2, self.city_width * 0.8)
        # y = np.clip(y, self.city_height * 0.2, self.city_height * 0.8)
        #
        # hospitals.append({
        #     'id': 'H1',
        #     'name': 'city_center_general_hospital',
        #     'position': (x, y),
        #     'capacity': np.random.randint(150, 300),
        #     'type': 'general_hospital'
        # })
        #
        # # 剩余的医院分布策略
        # for i in range(1, n):
        #     # 60%概率放在市中心，40%概率放在郊区
        #     if random.random() < 0.6 and i < n - 1:  # 确保不是最后一所医院
        #         # 市中心医院
        #         x = np.random.normal(downtown_center[0], self.city_width * 0.15)
        #         y = np.random.normal(downtown_center[1], self.city_height * 0.15)
        #         x = np.clip(x, self.city_width * 0.1, self.city_width * 0.9)
        #         y = np.clip(y, self.city_height * 0.1, self.city_height * 0.9)
        #         location = "city_center"
        #     else:
        #         # 郊区医院 - 分布在城市边缘但不要太偏远
        #         region = random.choice(['north', 'south', 'east', 'west'])
        #         if region == 'north':
        #             x = random.uniform(self.city_width * 0.2, self.city_width * 0.8)
        #             y = random.uniform(self.city_height * 0.7, self.city_height * 0.9)
        #         elif region == 'south':
        #             x = random.uniform(self.city_width * 0.2, self.city_width * 0.8)
        #             y = random.uniform(self.city_height * 0.1, self.city_height * 0.3)
        #         elif region == 'east':
        #             x = random.uniform(self.city_width * 0.7, self.city_width * 0.9)
        #             y = random.uniform(self.city_height * 0.2, self.city_height * 0.8)
        #         else:  # west
        #             x = random.uniform(self.city_width * 0.1, self.city_width * 0.3)
        #             y = random.uniform(self.city_height * 0.2, self.city_height * 0.8)
        #         location = "suburban"
        #
        #     # 选择医院类型 - 市中心倾向于综合医院，郊区倾向于普通医院
        #     if location == "city_center" and random.random() < 0.7:
        #         hospital_type = hospital_types[0]  # 综合医院
        #     else:
        #         hospital_type = hospital_types[1]  # 普通医院
        #
        #     capacity = np.random.randint(hospital_type['capacity_range'][0],
        #                                  hospital_type['capacity_range'][1])
        #
        #     hospitals.append({
        #         'id': f'H{i + 1}',
        #         'name': f'{location}{hospital_type["name"]}{i + 1}',
        #         'position': (x, y),
        #         'capacity': capacity,
        #         'type': hospital_type['name']
        #     })
        #
        # # """
        # # improve:
        # # 分地区
        # # """
        # # regions = [
        # #     {'name': 'city_center', 'center': (self.city_width * 0.5, self.city_height * 0.5),
        # #      'radius': self.city_width * 0.2},
        # #     {'name': 'north_suburban', 'center': (self.city_width * 0.5, self.city_height * 0.8),
        # #      'radius': self.city_width * 0.15},
        # #     {'name': 'south_suburban', 'center': (self.city_width * 0.5, self.city_height * 0.2),
        # #      'radius': self.city_width * 0.15},
        # #     {'name': 'west_suburban', 'center': (self.city_width * 0.8, self.city_height * 0.5),
        # #      'radius': self.city_width * 0.15},
        # #     {'name': 'east_suburban', 'center': (self.city_width * 0.2, self.city_height * 0.5),
        # #      'radius': self.city_width * 0.15},
        # #     {'name': 'west_north_rural', 'center': (self.city_width * 0.3, self.city_height * 0.7),
        # #      'radius': self.city_width * 0.1},
        # #     {'name': 'east_north_rural', 'center': (self.city_width * 0.7, self.city_height * 0.7),
        # #      'radius': self.city_width * 0.1},
        # #     {'name': 'west_south_rural', 'center': (self.city_width * 0.3, self.city_height * 0.3),
        # #      'radius': self.city_width * 0.1},
        # #     {'name': 'east_south_rural', 'center': (self.city_width * 0.7, self.city_height * 0.3),
        # #      'radius': self.city_width * 0.1}
        # # ]
        # # for i in range(min(n, len(regions))):
        # #     region = regions[i]
        # #     center_x, center_y = region['center']
        # #     radius = region['radius']
        # #
        # #     angle = random.uniform(0, 2 * np.pi)
        # #     distance = random.uniform(0, radius)
        # #     x = center_x + distance * np.cos(angle) * self.city_width
        # #     y = center_y + distance * np.sin(angle) * self.city_height
        # #     x = np.clip(x, self.city_width * 0.05, self.city_width * 0.95)
        # #     y = np.clip(y, self.city_height * 0.05, self.city_height * 0.95)
        # #
        # #     if 'city_center' in region['name']:
        # #         capacity = np.random.randint(100, 300)
        # #         hospital_type = '3_level'
        # #     elif 'suburban' in region['name']:
        # #         capacity = np.random.randint(50, 150)
        # #         hospital_type = '2_level'
        # #     else:
        # #         capacity = np.random.randint(20, 80)
        # #         hospital_type = '1_level'
        #
        # #     hospitals.append({
        # #         'id': f'H{i + 1}',
        # #         'name': f'{region["name"]}hospital',
        # #         'position': (x, y),
        # #         'capacity': capacity,
        # #         'type': hospital_type,
        # #         'region': region['name']
        # #     })
        # #
        # # # if need more
        # # for i in range(len(hospitals), n):
        # #     x = random.uniform(self.city_width * 0.1, self.city_width * 0.9)
        # #     y = random.uniform(self.city_height * 0.1, self.city_height * 0.9)
        # #     # region_weights = [0.1 if 'city_center' in r['name'] else 0.3 if 'city_side' in r['name'] else 0.6
        # #     #             for r in regions]
        # #     # chosen_region = random.choices(regions, weights=region_weights)[0]
        # #     # center_x, center_y = chosen_region['center']
        # #     # radius = chosen_region['radius']
        # #     #
        # #     # angle = random.uniform(0, 2 * np.pi)
        # #     # distance = random.uniform(0, radius)
        # #     # x = center_x + distance * np.cos(angle) * self.city_width
        # #     # y = center_y + distance * np.sin(angle) * self.city_height
        # #     #
        # #     # x = np.clip(x, self.city_width * 0.05, self.city_width * 0.95)
        # #     # y = np.clip(y, self.city_height * 0.05, self.city_height * 0.95)
        # #
        # #     hospitals.append({
        # #         'id': f'H{i + 1}',
        # #         'name': f'hospital{i + 1}',
        # #         'position': (x, y),
        # #         'capacity': np.random.randint(30, 120),
        # #         'type': 'hospital',
        # #         'region': 'random'
        # #     })
        # # """
        # # end
        # # """
        # # for i in range(n):
        # #     if i == 0:
        # #         # 位于市中心
        # #         x, y = self.city_width * 0.5, self.city_height * 0.5
        # #     elif i == 1:
        # #         # 居民区密集区域 + 1
        # #         x, y = self.city_width * 0.7, self.city_height * 0.7
        # #     else:
        # #         # 随机但偏向城市区域
        # #         x = np.random.normal(self.city_width * 0.5, self.city_height * 0.2)
        # #         y = np.random.normal(self.city_height * 0.5, self.city_height * 0.2)
        # #         x = np.clip(x, self.city_width * 0.1, self.city_width * 0.9)
        # #         y = np.clip(y, self.city_height * 0.1, self.city_height * 0.9)
        # #
        # #     hospitals.append({ 'id': f'H{i+1}', 'position': (x, y),
        # #                        'capacity': np.random.randint(50, 300),
        # #                        'type': 'general' if i == 0 else 'regional',
        # #                        'emergency_department': True})
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

        # for i in range(n):
        #     if i < 2:
        #         # 靠近市中心
        #         x = np.random.normal(self.city_width * 0.5, self.city_height * 0.1)
        #         y = np.random.normal(self.city_height * 0.5, self.city_height * 0.1)
        #     else:
        #         # 覆盖居民区
        #         residential_center = self._calculate_residential_centers()
        #         if len(residential_center) > 0:
        #             center_idx = i % len(residential_center)
        #             center = residential_center[center_idx]
        #             x = np.random.normal(center[0], self.city_width * 0.15)
        #             y = np.random.normal(center[1], self.city_height * 0.15)
        #         else:
        #             x = np.random.uniform(self.city_width * 0.1, self.city_height * 0.9)
        #             y = np.random.uniform(self.city_height * 0.1, self.city_height * 0.9)
        #     x = np.clip(x, self.city_width * 0.05, self.city_width * 0.95)
        #     y = np.clip(y, self.city_height * 0.05, self.city_height * 0.95)
        #     ambulance_stations.append({'id': f'AS{i+1}', 'position': (x, y),
        #                                'ambulance_count': np.random.randint(1, 7),
        #                                'coverage_radius': np.random.uniform(5, 20)})
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

        # centers = []
        # area_id = 0
        # # 定义不同类型的居民区分布
        # residential_types = [
        #     {'name': 'city_center_high', 'centers': [(0.5, 0.5)], 'density_range': (8, 15), 'weight': 0.2},
        #     {'name': 'suburban_medium', 'centers': [(0.3, 0.7), (0.7, 0.7), (0.3, 0.3), (0.7, 0.3)],
        #      'density_range': (4, 8), 'weight': 0.4},
        #     {'name': 'rural_low', 'centers': [(0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9),
        #                                        (0.2, 0.5), (0.5, 0.2), (0.8, 0.5), (0.5, 0.8)],
        #      'density_range': (1, 4), 'weight': 0.4}
        # ]
        # total_weight = sum(t['weight'] for t in residential_types)
        # type_counts = {}
        # remaining = n
        #
        # for i, res_type in enumerate(residential_types):
        #     if i == len(residential_types) - 1:
        #         count = remaining
        #     else:
        #         count = int(n * res_type['weight'] / total_weight)
        #         remaining -= count
        #     type_counts[res_type['name']] = count
        #
        #     # 生成各种类型的居民区
        # area_id = 0
        # for res_type in residential_types:
        #     count = type_counts[res_type['name']]
        #     density_range = res_type['density_range']
        #
        #     for _ in range(count):
        #         # 选择一个中心点
        #         center = random.choice(res_type['centers'])
        #         center_x, center_y = center
        #
        #         # 在中心周围生成位置
        #         if 'high' in res_type['name']:
        #             # 市中心，更集中
        #             x = np.random.normal(center_x, 0.05) * self.city_width
        #             y = np.random.normal(center_y, 0.05) * self.city_height
        #             area_size = np.random.uniform(0.5, 1.5)
        #         elif 'medium' in res_type['name']:
        #             # 郊区，中等分散
        #             x = np.random.normal(center_x, 0.08) * self.city_width
        #             y = np.random.normal(center_y, 0.08) * self.city_height
        #             area_size = np.random.uniform(1.0, 3.0)
        #         else:
        #             # 乡村，更分散
        #             x = np.random.normal(center_x, 0.12) * self.city_width
        #             y = np.random.normal(center_y, 0.12) * self.city_height
        #             area_size = np.random.uniform(2.0, 5.0)
        #
        #         x = np.clip(x, self.city_width * 0.02, self.city_width * 0.98)
        #         y = np.clip(y, self.city_height * 0.02, self.city_height * 0.98)
        #
        #         # 生成人口密度和人口
        #         density = np.random.uniform(density_range[0], density_range[1])
        #         population = int(density * area_size * 1000)
        #
        #         residential_areas.append({
        #             'id': f'R{area_id + 1}',
        #             'name': f'{res_type["name"]}residential_area{area_id + 1}',
        #             'position': (x, y),
        #             'population': population,
        #             'density': density,
        #             'area': area_size,
        #             'type': res_type['name']
        #         })
        #         area_id += 1

        return areas

        # for _ in range(max(3, n // 5)):
        #     # 中心点
        #     center_x = np.random.uniform(self.city_width * 0.2, self.city_height * 0.8)
        #     center_y = np.random.uniform(self.city_height * 0.2, self.city_height * 0.8)
        #     centers.append((center_x, center_y))
        #
        # for center in centers:
        #     # 围绕中心点生成居民区
        #     n_in_center = max(2, n // len(centers))
        #     for _ in range(n_in_center):
        #         if area_id >= n:
        #             break
        #
        #         x = np.random.normal(center[0], self.city_width * 0.1)
        #         y = np.random.normal(center[1], self.city_height * 0.1)
        #         x = np.clip(x, self.city_width * 0.05, self.city_width * 0.95)
        #         y = np.clip(y, self.city_height * 0.05, self.city_height * 0.95)
        #
        #         density = np.random.lognormal(2, 0.5)
        #         area = np.random.uniform(0.5, 6.0)
        #         population = int(density * area * 1000)
        #
        #         residential_areas.append({
        #             'id': f'R{area_id+1}',
        #             'position': (x, y),
        #             'population': population,
        #             'density': density,
        #             'area': area,
        #             'age_distribution': {
        #                 'young': np.random.uniform(0.1, 0.3),
        #                 'adult': np.random.uniform(0.4, 0.6),
        #                 'old': np.random.uniform(0.1, 0.3),
        #             }
        #         })
        #         area_id += 1
        # return residential_areas

    def _gen_road_network(self, n: int) -> List[Dict]:
        """gen road network"""
        roads = []
        # main roads (横向)
        for i in range(n // 2):
            y = self.city_height * (i+1) / (n // 2 + 1)
            roads.append({
                'id': f'R_H{i+1}',
                'type': 'highway' if i % 3 == 0 else 'arterial',
                'points': [(0, y), (self.city_width, y)],
                'speed_limit': 80 if i % 3 == 0 else 60
            })

        # main roads (纵向)
        for i in range(n // 2):
            x = self.city_width * (i+1) / (n // 2 + 1)
            roads.append({
                'id': f'R_V{i+1}',
                'type': 'highway' if i % 3 == 0 else 'arterial',
                'points': [(x, 0), (x, self.city_height)],
                'speed_limit': 80 if i % 3 == 0 else 60
            })

        # 如果还需要更多道路，随机生成连接线
        while len(roads) < n:
            # 连接两个随机点
            point1 = (random.uniform(0, 1), random.uniform(0, 1))
            point2 = (random.uniform(0, 1), random.uniform(0, 1))

            points = [(p[0] * self.city_width, p[1] * self.city_height) for p in [point1, point2]]
            roads.append({
                'id': f'Road_R{len(roads) + 1}',
                'type': 'local',
                'points': points,
                'speed_limit': 40,
                'name': f'local_road{len(roads) + 1}'
            })

        return roads

    def _get_region_for_residential(self, area_type: str) -> str:
        pass

    def _get_region_for_facility(self) -> str:
        pass

    def _get_position_in_region(self, region: str) -> Tuple[float, float]:
        pass

    def _distribute_by_weights(self, total: int, weights: List[float]) -> List[int]:
        pass

    def _calculate_residential_centers(self) -> List[Tuple[float, float]]:
        """calculate residential centers"""
        if not self.residential_areas:
            return []

        from sklearn.cluster import KMeans
        # use K-means find center
        positions = np.array([area['position'] for area in self.residential_areas])
        n_clusters = min(5, len(positions))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(positions)

        return kmeans.cluster_centers_.tolist()

    def gen_emergencies(self, n: int = 50):
        """generate emergencies"""
        self.emergencies = []
        emergency_types = {
            'cardiac': {'name': 'cardiac', 'priority': 1, 'color': 'red'},
            'trauma': {'name': 'trauma', 'priority': 2, 'color': 'orange'},
            'respiratory': {'name': 'respiratory', 'priority': 2, 'color': 'orange'},
            'other': {'name': 'other', 'priority': 3, 'color': 'yellow'}
        }
        # 定义不同区域的紧急事件分布权重
        region_weights = {
            'city_center': 0.6,  # 市中心人口密集，事件较多
            'suburban': 0.3,  # 郊区面积大，事件数量中等
            'rural': 0.1  # 乡村人口少但覆盖广，也有一定数量事件
        }
        region_counts = {}
        remaining = n
        regions = list(region_weights.keys())
        for i, region in enumerate(regions):
            if i == len(regions) - 1:
                count = remaining
            else:
                count = int(n * region_weights[region])
                remaining -= count
            region_counts[region] = count

        emergency_id = 0
        for region, count in region_counts.items():
            for _ in range(count):
                if region == 'city_center':
                    x = np.random.normal(0.5, 0.1) * self.city_width
                    y = np.random.normal(0.5, 0.1) * self.city_height
                elif region == 'suburban':
                    angle = random.uniform(0, 2 * np.pi)
                    distance = random.uniform(0.3, 0.6)
                    x = 0.5 + distance * np.cos(angle) * 0.5
                    y = 0.5 + distance * np.sin(angle) * 0.5
                    x *= self.city_width
                    y *= self.city_height
                else:
                    corner = random.choice([(0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)])
                    x = np.random.normal(corner[0], 0.15) * self.city_width
                    y = np.random.normal(corner[1], 0.15) * self.city_height
                x = np.clip(x, 0, self.city_width)
                y = np.clip(y, 0, self.city_height)

                nearest_residential = None
                min_distance = float('inf')

                for area in self.residential_areas:
                    dist = self._calculate_distance((x, y), area['position'])
                    if dist < min_distance:
                        min_distance = dist
                        nearest_residential = area

                emergency_type = random.choices(list(emergency_types.keys()),
                                                weights=[0.3, 0.25, 0.2, 0.25])[0]
                type_info = emergency_types[emergency_type]

                if region == 'rural' and random.random() < 0.4:
                    emergency_type = 'trauma'
                    type_info=emergency_types['trauma']

                self.emergencies.append({
                    'id': f'E{emergency_id + 1}',
                    'position': (x, y),
                    'type': emergency_type,
                    'type_name': type_info['name'],
                    'priority': type_info['priority'],
                    'color': type_info['color'],
                    'timestamp': emergency_id * 10,
                    'residential_area': nearest_residential['id'] if nearest_residential else None,
                    'status': 'pending',
                    'region': region
                })
                emergency_id += 1

        return self.emergencies

        # for i in range (n):
        #     if self.residential_areas:
        #         # 随机但倾向人口密集区域
        #         residential_weights = [area['population'] for area in self.residential_areas]
        #         chosen_area = random.choices(self.residential_areas, weights=residential_weights)[0]
        #
        #         # 在居民区周围生成事件位置
        #         base_x, base_y = chosen_area['position']
        #         x = np.random.normal(base_x, self.city_width * 0.05)
        #         y = np.random.normal(base_y, self.city_height * 0.05)
        #         residential_area_id = chosen_area['id']
        #     else:
        #         x = np.random.uniform(0, self.city_width)
        #         y = np.random.uniform(0, self.city_height)
        #         residential_area_id = None
        #
        #     x = np.clip(x, 0, self.city_width)
        #     y = np.clip(y, 0, self.city_height)
        #
        #     # 事件类型 & 优先级
        #     emergency_type = random.choices(list(emergency_types.keys()),
        #                                     weights=[0.3, 0.25, 0.2, 0.25])[0]
        #     type_info = emergency_types[emergency_type]
        #
        #     # if emergency_type == 'cardiac':
        #     #     priority = 1
        #     # elif emergency_type in ['respiratory', 'trauma']:
        #     #     priority = 2
        #     # else:
        #     #     priority = 3
        #
        #     self.emergencies.append({
        #         'id': f'E{i+1}',
        #         'position': (x, y),
        #         'type': emergency_type,
        #         'type_name': type_info['name'],
        #         'priority': type_info['priority'],
        #         'color': type_info['color'],
        #         'timestamp': i * 10,
        #         'residential_area': residential_area_id,
        #         'status': 'pending'  # pending, assigned, completed
        #     })

    def calculate_coverage_voronoi(self):
        """use Voronoi to calculate coverage"""
        if not self.ambulance_stations or len(self.ambulance_stations) < 3:
            return None

        try:
            points = np.array([station['position'] for station in self.ambulance_stations])
            boundary_buffer = 10
            boundary_points = [
                [-boundary_buffer, -boundary_buffer],
                [self.city_width + boundary_buffer, -boundary_buffer],
                [-boundary_buffer, self.city_height + boundary_buffer],
                [self.city_width + boundary_buffer, self.city_height + boundary_buffer]
            ]
            all_points = np.vstack([points, boundary_points])
            vor = Voronoi(all_points)
            return vor
        except Exception as e:
            print(f"计算Voronoi图时出错: {e}")
            return None

        # if not self.ambulance_stations:
        #     return None
        #
        # # 获取救护站位置
        # points = np.array([station['position'] for station in self.ambulance_stations])
        # vor = Voronoi(points)
        #
        # return vor

    def calculate_response_times(self):
        """calculate response times"""
        response_times = {}

        if not self.ambulance_stations or not self.emergencies:
            return {}

        for station in self.ambulance_stations:
            station_id = station['id']
            response_times[station_id] = {}

            for emergency in self.emergencies:
                # 直线距离 -- 简化计算
                distance = self._calculate_distance(station['position'], emergency['position'])

                base_speed = 40

                # 基于区域调整速度
                if emergency.get('region') == 'city_center':
                    traffic_factor = 0.6  # 市中心交通拥堵
                elif emergency.get('region') == 'suburban':
                    traffic_factor = 0.8  # 郊区交通较好
                else:  # 乡村
                    traffic_factor = 1.2  # 乡村道路畅通

                # 基于时间段调整速度
                time_of_day = emergency['timestamp'] % 1440
                if 420 <= time_of_day <= 600 or 1020 <= time_of_day <= 1200:
                    traffic_factor *= 0.6
                elif 600 < time_of_day < 1020:  # day
                    traffic_factor *= 0.8
                else:  # night
                    traffic_factor *= 1.2

                effective_speed = base_speed * traffic_factor
                response_time = (distance / effective_speed) * 60
                response_times[station_id][emergency['id']] = max(1, response_time)

        return response_times

    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """两点之间欧几里得距离"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def visualize_city_layout(self, show_voronoi: bool = True, show_emergencies: bool = True):
        """visualize city layout"""
        fig, ax = plt.subplots(1, 1, figsize=(15, 12))

        ax.set_facecolor('#f0f0f0')
        ax.set_xlim(0, self.city_width)
        ax.set_ylim(0, self.city_height)
        ax.set_xlabel('EW_Distance(km)')
        ax.set_ylabel('SN_Distance(km)')
        ax.set_title('city layout', fontsize=16, fontweight='bold', pad=20)

        # start
        # roads
        road_legend_added = False
        for road in self.roads:
            points = np.array(road['points'])
            if road['type'] == 'highway':
                line_style = {'color': '#e74c3c', 'linewidth': 4, 'alpha': 0.8,
                              'linestyle': '-', 'label': 'highway' if not road_legend_added else ""}
                road_legend_added = True
            elif road['type'] == 'arterial':
                line_style = {'color': '#f39c12', 'linewidth': 2.5, 'alpha': 0.7,
                              'linestyle': '-', 'label': 'arterial'}
            else:
                line_style = {'color': '#7f8c8d', 'linewidth': 2, 'alpha': 0.6,
                              'linestyle': '--', 'label': 'local_road' if not road_legend_added else ""}
            ax.plot(points[:, 0], points[:, 1], **line_style)

        # residential areas
        if self.residential_areas:
            populations = [area['population'] for area in self.residential_areas]
            max_pop = max(populations) if populations else 1

            for i, area in enumerate(self.residential_areas):
                x, y = area['position']
                population = area['population']

                if 'high' in area['type']:
                    color = '#e74c3c'  # 红色 - 高密度
                    size = 100 + (area['population'] / 5000) * 200
                    label = 'high density' if i == 0 else ""
                elif 'medium' in area['type']:
                    color = '#f39c12'  # 橙色 - 中密度
                    size = 80 + (area['population'] / 5000) * 150
                    label = 'medium density' if i == len(
                        [a for a in self.residential_areas if 'medium' in a['type']]) == 1 else ""
                else:
                    color = '#27ae60'  # 绿色 - 低密度
                    size = 60 + (area['population'] / 5000) * 100
                    label = 'low density' if i == len(
                        [a for a in self.residential_areas if 'low' in a['type']]) == 1 else ""

                ax.scatter(x, y, s=size, c=color, alpha=0.7,
                           edgecolors='darkblue', linewidth=1, label=label)

                # size = 80 + (population / max_pop) * 200
                # density_norm = min(1.0, area['density'] / 8)
                # color = plt.cm.Blues(0.3 + 0.7 * density_norm)
                # label = 'residential area(size=population)' if i == 0 else ""
                # ax.scatter(x, y, c=[color], s=size, alpha=0.6, label=label, edgecolors='darkblue', linewidths=1)
                #
                # if population > max_pop * 0.3:
                #     ax.annotate(area['id'], (x, y), xytext=(8, 8), textcoords = 'offset points',
                #                 fontsize=8, bbox=dict(boxstyle='round, pad=0.3', fc='white', alpha=0.7))

        # hospitals
        # hospital_plotted = False
        general_hospital_plotted = False
        regular_hospital_plotted = False
        for i, hospital in enumerate(self.hospitals):
            x, y = hospital['position']
            print(f"hospital {hospital['name']} location: ({x:.1f}, {y:.1f})")

            if hospital['type'] == 'general_hospital':
                marker, size, color = '^', 400, '#c0392b'  # 红色三角形 - 综合医院
                label = 'general_hospital' if not general_hospital_plotted else ""
                general_hospital_plotted = True
            else:
                marker, size, color = 's', 300, '#e74c3c'  # 橙色正方形 - 普通医院
                label = 'community_hospital' if not regular_hospital_plotted else ""
                regular_hospital_plotted = True

            ax.scatter(x, y, s=size, marker=marker, c=color,
                       edgecolors='darkred', linewidth=3,
                       label=label, zorder=10)

            ax.annotate(f"{hospital['name']}",
                        (x, y), xytext=(5, 15), textcoords='offset points',
                        fontsize=9, fontweight='bold', color=color,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.9),
                        zorder=11)
            # if '3_level' in hospital['type']:
            #     marker, size, color = '^', 400, '#c0392b'  # 红色 - 三级医院
            #     label = '3 level hospital' if not hospital_plotted else ""
            # elif '2_level' in hospital['type']:
            #     marker, size, color = 'D', 350, '#e74c3c'  # 橙色 - 二级医院
            #     label = '2 level hospital' if not hospital_plotted else ""
            # else:
            #     marker, size, color = 's', 300, '#f39c12'  # 黄色 - 一级医院
            #     label = '1 level hospital' if not hospital_plotted else ""

            # ax.scatter(x, y, s=size, marker=marker, c=color, edgecolors='darkred',
            #            linewidth=3, label=label, zorder=10)
            # ax.annotate(f"{hospital['name']}\n{hospital['type']}", (x, y),
            #             xytext=(5, 15), textcoords='offset points',fontsize=10, fontweight='bold',
            #             color=color,bbox=dict(boxstyle="round,pad=0.5", fc='white', ec=color, alpha=0.9),zorder=11,
            #             arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5)
            # )
            # hospital_plotted = True

        # stations
        for i, station in enumerate(self.ambulance_stations):
            x, y = station['position']
            label = 'ambulance station' if i == 0 else ""
            ax.scatter(x, y, s=200, marker='s', c='#27ae60', edgecolors='#229954',
                       linewidth=2, label=label,zorder=5)
            ax.annotate(f"{station['id']}\n{station['ambulance_count']}cars",
                        (x, y), xytext=(10, -20), textcoords='offset points',
                        fontsize=9, color='#229954',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#229954", alpha=0.8))

            # 覆盖范围
            if show_voronoi:
                circle = patches.Circle((x, y), station['coverage_radius'],
                                        fill=True, linestyle='--', alpha=0.15, color='#27ae60',
                                        label = 'station coverage' if i ==0 else "")
                ax.add_patch(circle)

        # emergencies
        if show_emergencies and self.emergencies:
            emergencies_to_show = self.emergencies[:min(25, len(self.emergencies))]
            # priority_markers = {
            #     1: {'marker': 'X', 'size': 120, 'label': 'Priority 1: cardiac'},
            #     2: {'marker': 'D', 'size': 90, 'label': 'Priority 2: trauma/respiratory'},
            #     3: {'marker': 'o', 'size': 60, 'label': 'Priority 3: other'}
            # }
            priority_plotted = {1: False, 2: False, 3: False}


            for emergency in emergencies_to_show:
                x, y = emergency['position']
                region = emergency.get('region', 'unknown')
                priority = emergency['priority']

                # 根据优先级选择标记
                if priority == 1:
                    marker, size, color = 'X', 100, 'red'
                    label = 'priority_1' if not priority_plotted[1] else ""
                    priority_plotted[1] = True
                elif priority == 2:
                    marker, size, color = 'D', 80, 'orange'
                    label = 'priority_2' if not priority_plotted[2] else ""
                    priority_plotted[2] = True
                else:
                    marker, size, color = 'o', 60, 'yellow'
                    label = 'priority_3' if not priority_plotted[3] else ""
                    priority_plotted[3] = True

                ax.scatter(x, y, s=size, marker=marker, c=color, alpha=0.7,
                           edgecolors='darkred', linewidth=1.5, label=label, zorder=3)

                # if region == 'city_center':
                #     marker, size = 'X', 120
                # elif region == 'suburban':
                #     marker, size = 'D', 90
                # else:
                #     marker, size = 'o', 60
                # ax.scatter(x, y, s=size, marker=marker,
                #            c=emergency['color'], alpha=0.8,
                #            edgecolors='darkred', linewidth=1.5)

                # priority_info = priority_markers[emergency['priority']]
                # ax.scatter(x, y, s=priority_info['size'],
                #            marker=priority_info['marker'],
                #            c=emergency['color'], alpha=0.8,
                #            edgecolors='darkred', linewidth=1.5,
                #            label=priority_info['label'])

        # Voronoi
        if show_voronoi and self.ambulance_stations and len(self.ambulance_stations) >= 3:
            vor = self.calculate_coverage_voronoi()
            if vor:
                voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='#3498db',
                                line_width=2, line_alpha=0.7, point_size=0)
                ax.plot([], [], color='#3498db', linewidth=2, label='Voronoi area')

        # handles, labels = ax.get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        #
        # ax.grid(True, alpha=0.3)
        # plt.tight_layout()

        legend_elements = [
            # 区域类型
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#f1c40f',
                   markersize=10, label='city_center', alpha=0.3),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ecc71',
                   markersize=10, label='suburban', alpha=0.3),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#3498db',
                   markersize=10, label='rural', alpha=0.3),

            # 医院类型
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#c0392b',
                   markersize=12, label='general_hospital', markeredgecolor='darkred', markeredgewidth=2),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#e74c3c',
                   markersize=10, label='community_hospital', markeredgecolor='darkred', markeredgewidth=2),
            # Line2D([0], [0], marker='^', color='w', markerfacecolor='#c0392b',
            #        markersize=12, label='3_level_hospital', markeredgecolor='darkred'),
            # Line2D([0], [0], marker='D', color='w', markerfacecolor='#e74c3c',
            #        markersize=10, label='2_level_hospital', markeredgecolor='darkred'),
            # Line2D([0], [0], marker='s', color='w', markerfacecolor='#f39c12',
            #        markersize=8, label='1_level_hospital', markeredgecolor='darkred'),

            # 居民区类型
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
                   markersize=10, label='high_density', alpha=0.7),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12',
                   markersize=8, label='medium_density', alpha=0.7),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60',
                   markersize=6, label='low_density', alpha=0.7),

            # 救护站
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#27ae60',
                   markersize=10, label='ambulance station', markeredgecolor='#229954'),

            # 道路
            Line2D([0], [0], color='#e74c3c', linewidth=4, label='highway'),
            Line2D([0], [0], color='#f39c12', linewidth=2.5, label='arterial'),
            Line2D([0], [0], color='#7f8c8d', linewidth=1.5, linestyle='--', label='local_road'),

            # 覆盖范围
            Line2D([0], [0], color='#27ae60', alpha=0.15, linewidth=10, label='amb_station_coverage'),
            Line2D([0], [0], color='#3498db', linewidth=2, label='Voronoi_coverage'),

            # 紧急事件
            Line2D([0], [0], marker='X', color='w', markerfacecolor='red',
                   markersize=10, label='priority_1', markeredgecolor='darkred'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='orange',
                   markersize=8, label='priority_2', markeredgecolor='darkred'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
                   markersize=6, label='priority_3', markeredgecolor='darkred'),

            # Line2D([0], [0], marker='^', color='w', markerfacecolor='#e74c3c',
            #        markersize=12, label='3_level_hospital', markeredgecolor='#c0392b', markeredgewidth=2),
            # Line2D([0], [0], marker='s', color='w', markerfacecolor='#27ae60',
            #        markersize=10, label='amb station', markeredgecolor='#229954', markeredgewidth=1.5),
            # Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
            #        markersize=10, label='residential area', alpha=0.7, markeredgecolor='darkblue'),
            #
            # # 道路
            # Line2D([0], [0], color='#e74c3c', linewidth=4, label='highway'),
            # Line2D([0], [0], color='#7f8c8d', linewidth=2, linestyle='--', label='arterial'),
            #
            # # 紧急事件优先级
            # Line2D([0], [0], marker='X', color='w', markerfacecolor='red',
            #        markersize=10, label='Priority 1: cardiac', markeredgecolor='darkred'),
            # Line2D([0], [0], marker='D', color='w', markerfacecolor='orange',
            #        markersize=8, label='Priority 2: trauma/respiratory', markeredgecolor='darkred'),
            # Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
            #        markersize=6, label='Priority 3: other', markeredgecolor='darkred'),
            #
            # # 覆盖范围
            # Line2D([0], [0], color='#27ae60', alpha=0.3, linewidth=10, label='amb station coverage'),
            # Line2D([0], [0], color='#3498db', linewidth=2, label='Voronoi area'),
        ]

        # 创建两个图例：主要图例和说明图例
        main_legend = ax.legend(handles=legend_elements,
                                loc='upper left',
                                bbox_to_anchor=(0.02, 0.98),
                                fontsize=10,
                                frameon=True,
                                fancybox=True,
                                shadow=True,
                                ncol=2,
                                title='Instruction',
                                title_fontsize=11)

        ax.add_artist(main_legend)

        # 添加统计信息框
        # stats_text = ""
        stats_text = f"""
        Statistical Information:
        • Area: {self.city_width * self.city_height:.0f} km²
        • Hospital: {len(self.hospitals)} 所
        • Ambulance station: {len(self.ambulance_stations)} 个
        • Residential area: {len(self.residential_areas)} 个
        • Population: {sum(area['population'] for area in self.residential_areas):,}
        • Ambulance: {sum(station['ambulance_count'] for station in self.ambulance_stations)}
        """

        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.8))

        # 添加网格和样式
        ax.grid(True, alpha=0.2, linestyle='-', color='gray')
        ax.set_axisbelow(True)

        # 添加比例尺
        scale_length = self.city_width * 0.2  # 20% of map width
        scale_y = self.city_height * 0.05
        ax.plot([self.city_width * 0.05, self.city_width * 0.05 + scale_length],
                [scale_y, scale_y], 'k-', linewidth=3)
        ax.text(self.city_width * 0.05 + scale_length / 2, scale_y - self.city_height * 0.02,
                f'{scale_length:.0f} km', ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        plt.tight_layout()

        return fig, ax

    def _draw_city_regions(self, ax):
        """绘制城市区域背景"""
        # 市中心区域
        downtown = patches.Circle((self.city_width * 0.5, self.city_height * 0.5),
                                  self.city_width * 0.2,
                                  fill=True, alpha=0.1, color='#f1c40f')
        ax.add_patch(downtown)

        # 郊区区域（环状）
        suburb_inner = self.city_width * 0.2
        suburb_outer = self.city_width * 0.4

        # 绘制四个象限的郊区
        angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        for angle in angles:
            x = self.city_width * 0.5 + suburb_inner * np.cos(angle)
            y = self.city_height * 0.5 + suburb_inner * np.sin(angle)

            # 使用矩形近似表示郊区区域
            if angle == 0:  # 东
                rect = patches.Rectangle((self.city_width * 0.5, self.city_height * 0.3),
                                         self.city_width * 0.3, self.city_height * 0.4,
                                         fill=True, alpha=0.1, color='#2ecc71')
            elif angle == np.pi / 2:  # 北
                rect = patches.Rectangle((self.city_width * 0.3, self.city_height * 0.5),
                                         self.city_width * 0.4, self.city_height * 0.3,
                                         fill=True, alpha=0.1, color='#2ecc71')
            elif angle == np.pi:  # 西
                rect = patches.Rectangle((self.city_width * 0.2, self.city_height * 0.3),
                                         self.city_width * 0.3, self.city_height * 0.4,
                                         fill=True, alpha=0.1, color='#2ecc71')
            else:  # 南
                rect = patches.Rectangle((self.city_width * 0.3, self.city_height * 0.2),
                                         self.city_width * 0.4, self.city_height * 0.3,
                                         fill=True, alpha=0.1, color='#2ecc71')
            ax.add_patch(rect)

        # 添加区域标签
        ax.text(self.city_width * 0.5, self.city_height * 0.5, 'city_center',
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))

        ax.text(self.city_width * 0.1, self.city_height * 0.1, 'rural',
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.7))
        ax.text(self.city_width * 0.1, self.city_height * 0.9, 'rural',
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.7))
        ax.text(self.city_width * 0.9, self.city_height * 0.1, 'rural',
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.7))
        ax.text(self.city_width * 0.9, self.city_height * 0.9, 'rural',
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.7))

    def visualize_response_analysis(self):
        """visualize response analysis"""
        if not self.emergencies:
            self.gen_emergencies(30)

        response_times = self.calculate_response_times()

        if not response_times:
            pass

        fig = plt.figure(figsize=(18, 12))
        # fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('response time analysis', fontsize=16, fontweight='bold', y=0.95)

        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, :])

        # response times
        all_response_times = []
        for station_times in response_times.values():
            all_response_times.extend(list(station_times.values()))

        n, bins, patches = ax1.hist(all_response_times, bins=15, alpha=0.7, color='skyblue', edgecolor='black')

        for i in range(len(patches)):
            if bins[i] < 8:
                patches[i].set_facecolor('#2ecc71')  # 绿色
            elif bins[i] < 12:
                patches[i].set_facecolor('#f39c12')  # 橙色
            else:
                patches[i].set_facecolor('#e74c3c')  # 红色

        mean_time = np.mean(all_response_times)
        ax1.axvline(mean_time, color='red', linestyle='--', linewidth=2,
                    label=f'average: {mean_time:.1f}min')
        ax1.axvline(8, color='blue', linestyle='-', linewidth=2,
                    label='8min standard', alpha=0.7)

        ax1.set_xlabel('response time(min)', fontsize=12)
        ax1.set_ylabel('count', fontsize=12)
        ax1.set_title('response time distribution\n', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 各救护站平均响应时间
        station_avg_times = []
        station_names = []
        station_colors = []

        for station_id, times in response_times.items():
            if times:
                avg_time = np.mean(list(times.values()))
                station_avg_times.append(avg_time)
                station_names.append(station_id)
                # 根据性能着色
                if avg_time <= 8:
                    station_colors.append('#2ecc71')  # 绿色 - 优秀
                elif avg_time <= 12:
                    station_colors.append('#f39c12')  # 橙色 - 一般
                else:
                    station_colors.append('#e74c3c')  # 红色 - 需改进

        bars = ax2.bar(station_names, station_avg_times, color=station_colors, alpha=0.7)
        ax2.axhline(8, color='blue', linestyle='-', linewidth=2, label='8min standard', alpha=0.7)

        ax2.set_xlabel('amb station', fontsize=12)
        ax2.set_ylabel('average respinse time(min)', fontsize=12)
        ax2.set_title('Performance Comparison of Emergency Medical Stations', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)

        # 在柱状图上添加数值和评级
        for bar, time in zip(bars, station_avg_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.2,
                     f'{time:.1f}', ha='center', va='bottom', fontweight='bold')
            if time <= 8:
                rating = 'outstanding'
            elif time <= 12:
                rating = 'normal'
            else:
                rating = 'need improvement'
            ax2.text(bar.get_x() + bar.get_width() / 2, height / 2,
                     rating, ha='center', va='center', color='white', fontweight='bold')

        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 按优先级的响应时间
        priority_times = {1: [], 2: [], 3: []}
        priority_names = {
            1: 'priority 1\n',
            2: 'priority 2\n',
            3: 'priority 3\n'
        }
        priority_colors = ['#e74c3c', '#f39c12', '#f1c40f']

        for emergency in self.emergencies:
            for station_times in response_times.values():
                if emergency['id'] in station_times:
                    priority_times[emergency['priority']].append(
                        station_times[emergency['id']]
                    )

        priorities = list(priority_times.keys())
        avg_times_by_priority = [
            np.mean(priority_times[p]) if priority_times[p] else 0
            for p in priorities
        ]

        bars = ax3.bar([priority_names[p] for p in priorities],
                       avg_times_by_priority,
                       color=priority_colors, alpha=0.7)
        ax3.axhline(8, color='blue', linestyle='-', linewidth=2,
                    label='8min standard', alpha=0.7)

        ax3.set_xlabel('Event Priority', fontsize=12)
        ax3.set_ylabel('Average Response Time (min)', fontsize=12)
        ax3.set_title('Response Time for Different Priorities', fontsize=14, fontweight='bold')

        for bar, time in zip(bars, avg_times_by_priority):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, height + 0.2,
                     f'{time:.1f}', ha='center', va='bottom', fontweight='bold')

        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 覆盖分析表格
        coverage_thresholds = [5, 8, 10, 12, 15]
        coverage_data = []

        for threshold in coverage_thresholds:
            coverage_rates = []
            for station_id, times in response_times.items():
                if times:
                    covered = sum(1 for t in times.values() if t <= threshold)
                    coverage_rate = covered / len(times) * 100
                    coverage_rates.append(coverage_rate)

            if coverage_rates:
                avg_coverage = np.mean(coverage_rates)
                coverage_data.append({
                    'threshold': threshold,
                    'avg_coverage': avg_coverage,
                    'rating': 'outstanding' if avg_coverage >= 90 else
                    'good' if avg_coverage >= 80 else
                    'normal' if avg_coverage >= 70 else 'need improvement'
                })

        # 创建覆盖分析表格
        table_data = []
        colors = []
        for data in coverage_data:
            table_data.append([
                f"{data['threshold']}min",
                f"{data['avg_coverage']:.1f}%",
                data['rating']
            ])
            # 根据评级着色
            if data['rating'] == 'outstanding':
                colors.append(['#d4edda'] * 3)
            elif data['rating'] == 'good':
                colors.append(['#fff3cd'] * 3)
            elif data['rating'] == 'normal':
                colors.append(['#ffeaa7'] * 3)
            else:
                colors.append(['#f8d7da'] * 3)

        table = ax4.table(cellText=table_data,
                          colLabels=['Time Standard', 'Mean Coverage Percentage', 'Rating'],
                          cellColours=colors,
                          loc='center',
                          cellLoc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)

        ax4.set_title('Coverage Performance with Different Time Standards', fontsize=14, fontweight='bold')
        ax4.axis('off')

        # 添加总体性能总结
        overall_coverage_8min = coverage_data[1]['avg_coverage'] if len(coverage_data) > 1 else 0
        performance_text = (f"Overall System Performance:\n• 8-minute coverage ratio: {overall_coverage_8min:.1f}%\n"
                            f"• Average Response Time: {mean_time:.1f}min")

        ax4.text(0.02, 0.95, performance_text, transform=ax4.transAxes,
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", alpha=0.8))

        plt.tight_layout()
        return fig, (ax1, ax2, ax3, ax4)

        # axes[0, 0].hist(all_response_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        # axes[0, 0].axvline(np.mean(all_response_times), color='red', linestyle='--',
        #                    label=f'average: {np.mean(all_response_times):.1f} min')
        # axes[0, 0].set_xlabel('response time')
        # axes[0, 0].set_ylabel('frequency')
        # axes[0, 0].set_title('response time distribution')
        # axes[0, 0].legend()
        # axes[0, 0].grid(True, alpha=0.3)

        # # response times in stations
        # station_avg_times = []
        # station_ids = []
        # for station_id, times in response_times.items():
        #     station_avg_times.append(np.mean(list(times.values())))
        #     station_ids.append(station_id)
        #
        # bars = axes[0, 1].bar(station_ids, station_avg_times, color='lightcoral', alpha=0.7)
        # axes[0, 1].set_xlabel('station')
        # axes[0, 1].set_ylabel('average response time')
        # axes[0, 1].set_title('average response time distribution in stations')
        #
        # for bar, time in zip(bars, station_avg_times):
        #     axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
        #                     f'{time:.1f}', ha='center', va='bottom')
        #
        # axes[0, 1].grid(True, alpha=0.3)
        #
        # # response time by priority
        # priority_times = {1: [], 2:[], 3:[]}
        # for emergency in self.emergencies:
        #     for station_times in response_times.values():
        #         if emergency['id'] in station_times:
        #             priority_times[emergency['priority']].append(station_times[emergency['id']])
        # priorities = list(priority_times.keys())
        # avg_time_by_priority = [np.mean(priority_times[p]) if priority_times[p] else 0 for p in priorities]
        #
        # bars = axes[1, 0].bar(priorities, avg_time_by_priority, color=['red', 'orange', 'yellow'], alpha=0.7)
        # axes[1, 0].set_xlabel('priorities')
        # axes[1, 0].set_ylabel('average response time')
        # axes[1, 0].set_title('average response time distribution in priorities')
        # axes[1, 0].set_xticks(priorities)
        #
        # for bar, time in zip(bars, avg_time_by_priority):
        #     axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
        #                     f'{time:.1f}', ha='center', va='bottom')
        # axes[1, 0].grid(True, alpha=0.3)
        #
        # # coverage
        # coverage_threshold = 8
        # coverage_by_station = {}
        # for station_id, times in response_times.items():
        #     covered = sum(1 for t in times.values() if t <= coverage_threshold)
        #     coverage_rate = covered / len(times) * 100 if times else 0
        #     coverage_by_station[station_id] = coverage_rate
        #
        # bars =axes[1, 1].bar(list(coverage_by_station.keys()), list(coverage_by_station.values()),
        #                      color='lightgreen', alpha=0.7)
        # axes[1, 1].axhline(90, color='red', linestyle='--', label='90%')
        # axes[1, 1].set_xlabel('ambulance station')
        # axes[1, 1].set_ylabel('coverage(%)')
        # axes[1, 1].set_title(f'{coverage_threshold}min coverage rate')
        # axes[1, 1].legend()
        # for bar, rate in zip(bars, coverage_by_station.values()):
        #     axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
        #                     f'{rate:.1f}%', ha='center', va='bottom')
        #
        # axes[1, 1].grid(True, alpha=0.3)
        # plt.tight_layout()
        # return fig, axes

    def get_simulation_stats(self) -> dict:
        """get simulation stats"""
        stats = {
            'city_area': self.city_width * self.city_height,
            'hospital_count': len(self.hospitals),
            'station_count': len(self.ambulance_stations),
            'residential_area_count': len(self.residential_areas),
            'total_population': sum(area['population'] for area in self.residential_areas),
            'total_ambulances': sum(station['ambulance_count'] for station in self.ambulance_stations)
        }
        if self.emergencies:
            response_times = self.calculate_response_times()
            all_times = []
            for station_times in response_times.values():
                all_times.extend(list(station_times.values()))

            stats.update({
                'emergency_count': len(self.emergencies),
                'avg_response_time': np.mean(all_times),
                'min_response_time': np.min(all_times),
                'max_response_time': np.max(all_times),
                'coverage_8min': sum(1 for t in all_times if t <= 8) / len(all_times) * 100
            })

        return stats

def demo_ambulance_simulation():
    """demo ambulance simulation"""
    print('demo ambulance simulation')
    simulation = AmbulanceSimulation(city_width=50, city_height=50)

    print('gen city layout =======')
    simulation.gen_city_layout(
        n_hospitals=4,
        n_stations=10,
        n_residential_areas=20,
        n_major_roads=8
    )

    print('gen emergencies =======')
    simulation.gen_emergencies(n=100)

    print('gen simulation stats =======')
    stats = simulation.get_simulation_stats()
    for key, value in stats.items():
        if 'population' in key:
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")

    print('save city_layout.png')
    fig1, ax1 = simulation.visualize_city_layout(show_voronoi=True, show_emergencies=True)
    plt.savefig('city_layout.png', dpi=300, bbox_inches='tight', facecolor='white')
    print('done')

    print('save response_analysis.png')
    fig2, ax2 = simulation.visualize_response_analysis()
    plt.savefig('response_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    print('done')

    plt.show()
    return simulation



