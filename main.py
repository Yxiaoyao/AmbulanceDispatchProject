from createMap import demo_ambulance_simulation, AmbulanceSimulation

if __name__ == '__main__':
    simulation = demo_ambulance_simulation()

    test_sim = AmbulanceSimulation(city_width=50, city_height=50)
    test_sim.gen_city_layout(n_hospitals=4, n_stations=10, n_residential_areas=15)
    test_sim.gen_emergencies(50)
    test_stats = test_sim.get_simulation_stats()
    print(f"救护车配置")

