from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from create_map import AmbulanceSimulation
from event_simulation import EventSimulator
from game_theory import GameTheory
from param import MapType


class AmbulanceDispatch(AmbulanceSimulation):

    def __init__(self, city_width: float = 100, city_height: float = 100, random_seed: int = 42,
                 map_type: MapType = MapType.GRID):
        super().__init__(city_width, city_height, random_seed, map_type)
        self.event_sim = None
        self.game_theoretic_dispatcher = None
        self.simulation_results = {}

    def run_simulation(self, simulation_time: float = 24 * 60):
        self.gen_city_layout()

        self.event_sim = EventSimulator(self)
        self.game_theoretic_dispatcher = GameTheory(self)
        fig, ax = self.visualize_city_layout(save_plot=True)

        performance_stats = self.event_sim.run_simulation(simulation_time)
        self._analyze_simulation_results(performance_stats)
        return performance_stats

    def _analyze_simulation_results(self, stats: Dict):
        self.simulation_results = {
            'average_response_time': np.mean(stats['response_times']) if stats['response_times'] else 0,
            'response_time_std': np.std(stats['response_times']) if stats['response_times'] else 0,
            'coverage_8min': len([t for t in stats['response_times'] if t <= 8]) / len(stats['response_times']) * 100 if
            stats['response_times'] else 0,
            'total_emergencies_served': len(stats['response_times']),
            'ambulance_utilization_rate': np.mean(stats['ambulance_utilization']) if stats[
                'ambulance_utilization'] else 0,
            'strategy_performance': dict(stats['strategy_performance'])
        }

    def visualize_strategy_analysis(self):
        if not self.simulation_results.get('strategy_performance'):
            print("No strategy data available")
            return

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
        fig.suptitle('hospital strategy analysis', fontsize=16, fontweight='bold')

        strategy_data = self.simulation_results['strategy_performance']
        strategies = list(strategy_data.keys())

        avg_response_times = [np.mean(times) for times in strategy_data.values()]
        bars1 = axes[0, 0].bar(strategies, avg_response_times, color=['#2ecc71', '#e74c3c', '#f39c12'], alpha=0.7)
        axes[0, 0].set_title('Average response time')
        axes[0, 0].set_ylabel('Response time (min)')
        for bar, time in zip(bars1, avg_response_times):
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                            f'{time:.1f}', ha='center', va='bottom')

        coverage_rates = []
        for times in strategy_data.values():
            covered = len([t for t in times if t <= 8])
            coverage_rates.append(covered / len(times) * 100 if times else 0)

        bars2 = axes[0, 1].bar(strategies, coverage_rates, color=['#2ecc71', '#e74c3c', '#f39c12'], alpha=0.7)
        axes[0, 1].set_title('Coverage rate')
        axes[0, 1].set_ylabel('(%)')
        axes[0, 1].axhline(90, color='red', linestyle='--', alpha=0.7, label='')
        for bar, rate in zip(bars2, coverage_rates):
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f'{rate:.1f}%', ha='center', va='bottom')
        axes[0, 1].legend()

        all_times = []
        strategy_labels = []
        for strategy, times in strategy_data.items():
            all_times.extend(times)
            strategy_labels.extend([strategy] * len(times))

        if all_times:
            box_data = [strategy_data[s] for s in strategies if strategy_data[s]]
            axes[1, 0].boxplot(box_data, labels=strategies)
            axes[1, 0].set_title('Response time')
            axes[1, 0].set_ylabel('(min)')

        strategy_counts = {s: len(times) for s, times in strategy_data.items()}
        axes[1, 1].pie(strategy_counts.values(), labels=strategy_counts.keys(),
                       autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c', '#f39c12'])
        axes[1, 1].set_title('strategy pie')

        plt.tight_layout()
        return fig, axes

def demo_simulation():
    print("=== start ambulance dispatch simulation ===")

    grid_simulation = AmbulanceDispatch(city_width=70, city_height=70,
                                        map_type=MapType.GRID)
    grid_result = grid_simulation.run_simulation(simulation_time=8 * 60)

    for key, value in grid_simulation.simulation_results.items():
        if key != 'strategy_performance':
            print(f'{key}: {value}')

    ring_simulation = AmbulanceDispatch(city_width=70, city_height=70,
                                        map_type=MapType.RING)
    ring_result = ring_simulation.run_simulation(simulation_time=8 * 60)
    for key, value in ring_simulation.simulation_results.items():
        if key != 'strategy_performance':
            print(f'{key}: {value}')

    print("\n=== Compare ===")
    print(f"{'Metric':<25} {'Grid':<15} {'Ring':<15}")
    print("-" * 55)

    metrics_to_compare = [
        ('average_response_time', 'Avg Response Time'),
        ('response_time_std', 'Response Time Std'),
        ('coverage_8min', '8min Coverage Rate'),
        ('total_emergencies_served', 'Emergencies Served'),
        ('ambulance_utilization_rate', 'Ambulance Utilization')
    ]

    for metric_key, metric_name in metrics_to_compare:
        grid_val = grid_simulation.simulation_results.get(metric_key, 0)
        ring_val = ring_simulation.simulation_results.get(metric_key, 0)
        if isinstance(grid_val, float):
            print(f"{metric_name:<25} {grid_val:<15.2f} {ring_val:<15.2f}")
        else:
            print(f"{metric_name:<25} {grid_val:<15} {ring_val:<15}")

    return grid_simulation, ring_simulation

    # simulation = AmbulanceDispatch(city_width=70, city_height=70)
    # print("=== run ambulance dispatch simulation ===")
    # result = simulation.run_simulation(simulation_time=8*60)
    #
    # print("=== end ambulance dispatch simulation ===")
    # for key, value in simulation.simulation_results.items():
    #     if key != 'strategy_performance':
    #         print(f'{key}: {value}')
    #
    # return simulation


def compare_map_types():
    """比较两种地图类型的性能"""
    print("=== 地图类型比较分析 ===")

    # 多次运行以获得统计显著性
    n_runs = 5
    grid_results = []
    ring_results = []

    for i in range(n_runs):
        print(f"\n运行 {i + 1}/{n_runs}")

        # 方型
        grid_sim = AmbulanceDispatch(city_width=70, city_height=70,
                                     map_type=MapType.GRID, random_seed=i)
        grid_sim.gen_city_layout()
        grid_sim.event_sim = EventSimulator(grid_sim)
        grid_sim.game_theoretic_dispatcher = GameTheory(grid_sim)
        grid_stats = grid_sim.event_sim.run_simulation(simulation_time=4 * 60)
        grid_sim._analyze_simulation_results(grid_stats)
        grid_results.append(grid_sim.simulation_results)

        # 环形
        ring_sim = AmbulanceDispatch(city_width=70, city_height=70,
                                     map_type=MapType.RING, random_seed=i)
        ring_sim.gen_city_layout()
        ring_sim.event_sim = EventSimulator(ring_sim)
        ring_sim.game_theoretic_dispatcher = GameTheory(ring_sim)
        ring_stats = ring_sim.event_sim.run_simulation(simulation_time=4 * 60)
        ring_sim._analyze_simulation_results(ring_stats)
        ring_results.append(ring_sim.simulation_results)

        metrics = ['average_response_time', 'coverage_8min', 'ambulance_utilization_rate']
        print("\n=== Avg ===")
        print(f"{'Metric':<25} {'Grid (Avg)':<15} {'Ring (Avg)':<15} {'Difference':<15}")
        print("-" * 70)

        for metric in metrics:
            grid_avg = np.mean([r.get(metric, 0) for r in grid_results])
            ring_avg = np.mean([r.get(metric, 0) for r in ring_results])
            diff = ring_avg - grid_avg

            print(f"{metric:<25} {grid_avg:<15.2f} {ring_avg:<15.2f} {diff:<15.2f}")

        return grid_results, ring_results

if __name__ == '__main__':
    # simulation = demo_simulation()
    print("选择运行模式:")
    print("1. 单个地图类型模拟 (方格型)")
    print("2. 单个地图类型模拟 (环形)")
    print("3. 两种地图类型对比")
    print("4. 多次运行性能比较")

    choice = input("请输入选择 (1-4): ").strip()

    if choice == '1':
        simulation = AmbulanceDispatch(city_width=70, city_height=70,
                                       map_type=MapType.GRID)
        simulation.run_simulation(simulation_time=8 * 60)
    elif choice == '2':
        simulation = AmbulanceDispatch(city_width=70, city_height=70,
                                       map_type=MapType.RING)
        simulation.run_simulation(simulation_time=8 * 60)
    elif choice == '3':
        grid_sim, ring_sim = demo_simulation()
    elif choice == '4':
        grid_results, ring_results = compare_map_types()
    else:
        print("无效选择，默认运行方格型规划")
        simulation = AmbulanceDispatch(city_width=70, city_height=70,
                                       map_type=MapType.GRID)
        simulation.run_simulation(simulation_time=8 * 60)
# from createMap import demo_ambulance_simulation, AmbulanceSimulation
#
# if __name__ == '__main__':
#     simulation = demo_ambulance_simulation()
#
#     test_sim = AmbulanceSimulation(city_width=50, city_height=50)
#     test_sim.gen_city_layout(n_hospitals=4, n_stations=10, n_residential_areas=15)
#     test_sim.gen_emergencies(50)
#     test_stats = test_sim.get_simulation_stats()
#     print(f"救护车配置")


