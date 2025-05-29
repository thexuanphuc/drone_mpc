import numpy as np
from mpc_controller import QuadcopterMPC
from simulator import AdvancedSimulator, Plotter, export_simulation_data

if __name__ == '__main__':
    # MODIFIED: Set scenario to 1 and adjust target_pos to test "going over"
    scenario_to_run = 1

    mpc_freq = 20.0
    plant_freq = 100.0

    mpc_dt_sim = 1.0 / mpc_freq
    plant_dt_sim = 1.0 / plant_freq

    mpc_controller = QuadcopterMPC(mpc_sampling_time=mpc_dt_sim, scenario=scenario_to_run)
    mpc_controller.mpc_initialize()
    
    if scenario_to_run == 4: # This block is for the "around" tall walls case
        start_pos = [1.0, 1.0, 3.5] 
        target_pos = [5.5, 5.5, 1.7] 
    elif scenario_to_run == 1: # Test "over" for scenario 1 obstacles
        start_pos = [2.5, 2.5, 0.5] # Start low
        target_pos = [6.0, 6.0, 3.0] # Target high (above all scenario 1 obstacles)
    else: # Default for other scenarios
        start_pos = [2.5, 2.5, 1.5]
        target_pos = [6.0, 6.0, 2.0]

    start_angles = [0.0, 0.0, 0.0]
    start_world_vel = [0.0, 0.0, 0.0]; start_body_rates = [0.0, 0.0, 0.0]
    initial_state_list = start_pos + start_angles + start_world_vel + start_body_rates
    initial_state_np = np.array(initial_state_list)

    target_angles = [0.0, 0.0, np.deg2rad(0)] 
    target_world_vel = [0.0, 0.0, 0.0]; target_body_rates = [0.0, 0.0, 0.0]
    target_state_list = target_pos + target_angles + target_world_vel + target_body_rates
    target_state_np = np.array(target_state_list)

    print(f"Running Scenario: {scenario_to_run}")
    print(f"Initial state: {initial_state_np}")
    print(f"Target state: {target_state_np}")

    advanced_sim = AdvancedSimulator(mpc_controller, initial_state_np, target_state_np,
                                     plant_dt=plant_dt_sim, mpc_dt=mpc_dt_sim)
    
    simulation_duration = 25.0 
    advanced_sim.run_simulation(until_time=simulation_duration)

    if advanced_sim.time_log: 
        Plotter.trajectory(
            advanced_sim.actual_state_log,
            advanced_sim.mpc_predicted_log,
            initial_state_np, 
            target_state_np,  
            mpc_controller.obstacles,
            scenario_to_run
        )
        Plotter.states_vs_time(
            advanced_sim.time_log,
            advanced_sim.actual_state_log,
            advanced_sim.mpc_predicted_log,
            scenario_to_run
        )
        export_simulation_data(
            advanced_sim.time_log,
            advanced_sim.actual_state_log,
            txt_filename=f"drone_trajectory_scenario_{scenario_to_run}.txt",
            excel_filename=f"drone_trajectory_scenario_{scenario_to_run}.xlsx"
        )
    else:
        print("No data logged for plotting or export. Simulation might not have run correctly.")