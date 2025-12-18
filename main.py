import math
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from create_map import AmbulanceSimulation
from event_simulation import EventSimulator
from game_theory import GameTheory
from param import MapType


class AmbulanceDispatch(AmbulanceSimulation):

    def __init__(self, city_width: float = 100, city_height: float = 100, random_seed: int = 42,
                 map_type: MapType = MapType.GRID, ring_density_factor: float = 1.5):
        super().__init__(city_width, city_height, random_seed, map_type, ring_density_factor)
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

def ring_density_variations():
    print("=== different ring density variations ===")
    density_factors = [1.0, 1.5, 2.0, 3.0]
    results = {}

    for density_factor in density_factors:
        print(f"\n=== density factor: {density_factor} ===")
        simulation = AmbulanceDispatch(
            city_width=70, city_height=70,
            map_type=MapType.RING,
            ring_density_factor=density_factor,
            random_seed=42
        )
        simulation.gen_city_layout()
        simulation.visualize_city_layout(save_plot=True)
        center_x, center_y = simulation.city_center
        max_radius = min(simulation.city_width, simulation.city_height) * 0.45
        inner_radius = max_radius * 0.3
        outer_inner = max_radius * 0.7
        outer_radius = max_radius

        inner_roads = 0
        middle_roads = 0
        outer_roads = 0

        for road in simulation.roads:
            if 'points' in road and road['points']:
                avg_radius = 0.0
                for point in road['points']:
                    distance = math.sqrt((point[0] - center_x) ** 2 + (point[1] - center_y) ** 2)
                    avg_radius += distance
                avg_radius /= len(road['points'])

                if avg_radius <= inner_radius:
                    inner_roads += 1
                elif avg_radius <= outer_inner:
                    middle_roads += 1
                else:
                    outer_roads += 1

        inner_area = math.pi * inner_radius ** 2
        middle_area = math.pi * (outer_inner ** 2 - inner_radius ** 2)
        outer_area = math.pi * (outer_radius ** 2 - outer_inner ** 2)

        inner_density = inner_roads / inner_area if inner_area > 0 else 0
        middle_density = middle_roads / middle_area if middle_area > 0 else 0
        outer_density = outer_roads / outer_area if outer_area > 0 else 0

        results[density_factor] = {
            'inner_density': inner_density,
            'middle_density': middle_density,
            'outer_density': outer_density,
            'inner_roads': inner_roads,
            'middle_roads': middle_roads,
            'outer_roads': outer_roads,
            'density_ratio': inner_density / outer_density if outer_density > 0 else float('inf')
        }

    print("\n=== density factor compare ===")
    print(f"{'factor':<10} {'center':<12} {'middle':<12} {'side':<12} {'rate':<10}")
    print("-" * 60)

    for density_factor in density_factors:
        result = results[density_factor]
        print(f"{density_factor:<10.1f} {result['inner_density']:<12.3f} "
              f"{result['middle_density']:<12.3f} {result['outer_density']:<12.3f} "
              f"{result['density_ratio']:<10.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('density analyze', fontsize=16, fontweight='bold')

    ax1 = axes[0, 0]
    x = np.arange(len(density_factors))
    width = 0.25

    inner_densities = [results[df]['inner_density'] for df in density_factors]
    middle_densities = [results[df]['middle_density'] for df in density_factors]
    outer_densities = [results[df]['outer_density'] for df in density_factors]

    ax1.bar(x - width, inner_densities, width, label='center', color='#e74c3c')
    ax1.bar(x, middle_densities, width, label='middle', color='#f39c12')
    ax1.bar(x + width, outer_densities, width, label='side', color='#7f8c8d')

    ax1.set_xlabel('density factor')
    ax1.set_ylabel('road denstiy (counts/km²)')
    ax1.set_title('different area roads density')
    ax1.set_xticks(x)
    ax1.set_xticklabels(density_factors)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    density_ratios = [results[df]['density_ratio'] for df in density_factors]
    ax2.plot(density_factors, density_ratios, 'o-', linewidth=2, markersize=8, color='#2ecc71')
    ax2.set_xlabel('density factor')
    ax2.set_ylabel('center/side density rate')
    ax2.set_title('variation')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    inner_counts = [results[df]['inner_roads'] for df in density_factors]
    middle_counts = [results[df]['middle_roads'] for df in density_factors]
    outer_counts = [results[df]['outer_roads'] for df in density_factors]

    ax3.bar(density_factors, inner_counts, label='center', color='#e74c3c')
    ax3.bar(density_factors, middle_counts, bottom=inner_counts, label='middle', color='#f39c12')
    ax3.bar(density_factors, outer_counts, bottom=np.array(inner_counts) + np.array(middle_counts),
            label='side', color='#7f8c8d')

    ax3.set_xlabel('density factor')
    ax3.set_ylabel('roads count')
    ax3.set_title('different area roads count')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    angles = np.linspace(0, 2 * np.pi, len(density_factors), endpoint=False).tolist()
    angles += angles[:1]

    max_density = max(max(inner_densities), max(middle_densities), max(outer_densities))
    norm_inner = [d / max_density for d in inner_densities]
    norm_middle = [d / max_density for d in middle_densities]
    norm_outer = [d / max_density for d in outer_densities]
    norm_inner += norm_inner[:1]
    norm_middle += norm_middle[:1]
    norm_outer += norm_outer[:1]
    angles = np.linspace(0, 2 * np.pi, len(density_factors), endpoint=False).tolist()
    angles += angles[:1]

    ax4.plot(angles, norm_inner, 'o-', linewidth=2, label='center', color='#e74c3c')
    ax4.fill(angles, norm_inner, alpha=0.25, color='#e74c3c')
    ax4.plot(angles, norm_middle, 'o-', linewidth=2, label='middle', color='#f39c12')
    ax4.fill(angles, norm_middle, alpha=0.25, color='#f39c12')
    ax4.plot(angles, norm_outer, 'o-', linewidth=2, label='side', color='#7f8c8d')
    ax4.fill(angles, norm_outer, alpha=0.25, color='#7f8c8d')

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels([f'DF={df}' for df in density_factors])
    ax4.set_title('')
    ax4.legend(loc='upper right')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig('simulation_results/ring_density_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results

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
    print("2. 单个地图类型模拟 (环形-密度均匀)")
    print("3. 单个地图类型模拟 (环形-中心密集)")
    print("4. 两种地图类型对比")
    print("5. 多次运行性能比较")
    print("6. 环形规划密度梯度分析")


    choice = input("请输入选择 (1-6): ").strip()

    if choice == '1':
        simulation = AmbulanceDispatch(city_width=70, city_height=70,
                                       map_type=MapType.GRID)
        simulation.run_simulation(simulation_time=8 * 60)

    elif choice == '2':
        simulation = AmbulanceDispatch(city_width=70, city_height=70,
                                       map_type=MapType.RING)
        simulation.run_simulation(simulation_time=8 * 60)
    elif choice == '3':
        simulation = AmbulanceDispatch(
            city_width=70, city_height=70,
            map_type=MapType.RING,
            ring_density_factor=2.0,  # 中心密集
            random_seed=42
        )
        simulation.run_simulation(simulation_time=8 * 60)
    elif choice == '4':
        grid_simulation = AmbulanceDispatch(
            city_width=70, city_height=70,
            map_type=MapType.GRID,
            random_seed=42
        )
        grid_result = grid_simulation.run_simulation(simulation_time=4 * 60)

        ring_simulation = AmbulanceDispatch(
            city_width=70, city_height=70,
            map_type=MapType.RING,
            ring_density_factor=2.0,
            random_seed=42
        )
        ring_result = ring_simulation.run_simulation(simulation_time=4 * 60)
        if 'avg_response_time' in grid_simulation.simulation_results:
            grid_time = grid_simulation.simulation_results['avg_response_time']
            ring_time = ring_simulation.simulation_results['avg_response_time']
            diff = ring_time - grid_time
            print(f"{'ave response time (min)':<25} {grid_time:<15.2f} {ring_time:<15.2f} {diff:<15.2f}")

        if 'coverage_8min' in grid_simulation.simulation_results:
            grid_cov = grid_simulation.simulation_results['coverage_8min']
            ring_cov = ring_simulation.simulation_results['coverage_8min']
            diff = ring_cov - grid_cov
            print(f"{'8 min coverage (%)':<25} {grid_cov:<15.2f} {ring_cov:<15.2f} {diff:<15.2f}")

    elif choice == '5':
        grid_results, ring_results = compare_map_types()
    elif choice == '6':
        results = ring_density_variations()
    else:
        print("wrong choice")
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


