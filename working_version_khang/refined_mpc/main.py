import numpy as np
from direct_collocation import QuadcopterMPC
# from mpc_controller import QuadcopterMPC
from simulator import AdvancedSimulator, Plotter

if __name__ == '__main__':
    mpc_freq = 20.0  # Hz
    plant_freq = 100.0 # Hz

    mpc_dt_sim = 1.0 / mpc_freq
    plant_dt_sim = 1.0 / plant_freq

    # Initialize MPC with its own sampling time, which must match mpc_dt_sim
    mpc_controller = QuadcopterMPC(mpc_sampling_time=mpc_dt_sim)
    mpc_controller.mpc_initialize()
    
    # State vector: (x,y,z, phi,theta,psi, vx_w,vy_w,vz_w, p,q,r)
    start_pos = [2.5, 2.5, 1.5]; start_angles = [0.0, 0.0, 0.0]
    start_world_vel = [0.0, 0.0, 0.0]; start_body_rates = [0.0, 0.0, 0.0]
    initial_state_list = start_pos + start_angles + start_world_vel + start_body_rates
    initial_state_np = np.array(initial_state_list)

    target_pos = [6.0, 6.0, 2.0]; target_angles = [0.0, 0.0, np.deg2rad(45)]
    target_world_vel = [0.0, 0.0, 0.0]; target_body_rates = [0.0, 0.0, 0.0]
    target_state_list = target_pos + target_angles + target_world_vel + target_body_rates
    target_state_np = np.array(target_state_list)

    print(f"Initial state: {initial_state_np}")
    print(f"Target state: {target_state_np}")

    advanced_sim = AdvancedSimulator(mpc_controller, initial_state_np, target_state_np,
                                     plant_dt=plant_dt_sim, mpc_dt=mpc_dt_sim)
    
    simulation_duration = 5.0 # seconds
    advanced_sim.run_simulation(until_time=simulation_duration)

    # Plotting
    if advanced_sim.time_log: # Check if simulation ran and logged data
        Plotter.trajectory(
            advanced_sim.actual_state_log,
            advanced_sim.mpc_predicted_log,
            initial_state_np, # Pass the numpy array directly
            target_state_np,  # Pass the numpy array directly
            mpc_controller.obstacles
        )
        Plotter.states_vs_time(
            advanced_sim.time_log,
            advanced_sim.actual_state_log,
            advanced_sim.mpc_predicted_log
        )
    else:
        print("No data logged for plotting. Simulation might not have run correctly.")