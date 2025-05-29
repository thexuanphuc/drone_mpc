import simpy
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os 
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas library not found. Excel export will be disabled. "
          "Install with 'pip install pandas openpyxl'")


class AdvancedSimulator:
    def __init__(self, mpc_controller, initial_state_np, target_state_np, plant_dt=0.01, mpc_dt=0.1):
        self.env = simpy.Environment()
        self.mpc = mpc_controller 
        self.plant_dt = plant_dt
        self.mpc_dt = mpc_dt

        if abs(self.mpc.T - self.mpc_dt) > 1e-6:
            print(f"Warning: MPC internal sampling time mpc.T ({self.mpc.T}s) "
                  f"differs from simulator's mpc_dt ({self.mpc_dt}s). ")

        self.current_state_np = np.array(initial_state_np).flatten() 
        self.target_state_np = np.array(target_state_np).flatten()
        
        hover_thrust_per_motor = (self.mpc.m * self.mpc.g) / self.mpc.n_controls
        self.latest_control_input_np = np.full(self.mpc.n_controls, hover_thrust_per_motor)

        self.time_log = []
        self.actual_state_log = []
        self.mpc_predicted_log = [] 
        self.control_input_log = []
        self.prev_w_opt = None 
        self.mpc_compute_time_log = []
        self.env.process(self.plant_process())
        self.env.process(self.mpc_process())
        self.u_log = []
    def plant_process(self):
        while True:
            control_to_apply_ca = ca.DM(self.latest_control_input_np) 
            current_state_ca = ca.DM(self.current_state_np)

            k1 = self.mpc.f(current_state_ca, control_to_apply_ca).full().flatten()
            k2 = self.mpc.f(current_state_ca + self.plant_dt/2 * k1, control_to_apply_ca).full().flatten()
            k3 = self.mpc.f(current_state_ca + self.plant_dt/2 * k2, control_to_apply_ca).full().flatten()
            k4 = self.mpc.f(current_state_ca + self.plant_dt * k3, control_to_apply_ca).full().flatten()
            self.current_state_np += (self.plant_dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            
            yield self.env.timeout(self.plant_dt)

    def mpc_process(self):
        iteration_count = 0
        initial_u_guess_flat = np.tile(self.latest_control_input_np, self.mpc.N)
        
        while True:
            current_state_for_mpc_np = self.current_state_np.copy()
            
            if np.any(np.isnan(current_state_for_mpc_np)) or np.any(np.isinf(current_state_for_mpc_np)):
                print(f"Error: Invalid state detected at Sim Time {self.env.now:.2f}s before MPC call: {current_state_for_mpc_np}")
                break 

            w_guess_for_solver = None
            if self.prev_w_opt is not None:
                prev_U_opt = ca.reshape(self.prev_w_opt[:self.mpc.n_controls*self.mpc.N], self.mpc.n_controls, self.mpc.N)
                prev_X_opt = ca.reshape(self.prev_w_opt[self.mpc.n_controls*self.mpc.N:], self.mpc.n_states, self.mpc.N+1)
                shifted_U_guess = ca.horzcat(prev_U_opt[:, 1:], prev_U_opt[:, -1]) 
                shifted_X_guess = ca.horzcat(prev_X_opt[:, 1:], prev_X_opt[:, -1]) 
                shifted_X_guess[:, 0] = current_state_for_mpc_np 
                w_guess_for_solver = ca.vertcat(
                    ca.reshape(shifted_U_guess, -1, 1),
                    ca.reshape(shifted_X_guess, -1, 1)
                )
            else: 
                initial_x_guess_flat = np.tile(current_state_for_mpc_np, self.mpc.N + 1)
                w_guess_for_solver = ca.vertcat(initial_u_guess_flat, initial_x_guess_flat)

            try:

                start_time = time.time()  # Start timing the MPC computation
                u_sequence_ca, x_sequence_ca, w_opt_solution = self.mpc.run_once(
                    current_state_for_mpc_np, 
                    self.target_state_np,
                    initial_guess_w=w_guess_for_solver 
                )
                end_time = time.time()  # End timing the MPC computation
                self.mpc_compute_time_log.append(end_time - start_time)
                self.u_log.append(u_sequence_ca[:, 0].full().reshape(4,-1))
                self.latest_control_input_np = u_sequence_ca[:, 0].full().flatten()
                self.prev_w_opt = w_opt_solution 

                self.time_log.append(self.env.now)
                self.actual_state_log.append(current_state_for_mpc_np) 
                self.mpc_predicted_log.append(x_sequence_ca[:, 1].full().flatten()) 
                self.control_input_log.append(self.latest_control_input_np.copy())
            except Exception as e:
                print(f"Error during MPC run_once at Sim Time {self.env.now:.2f}s: {e}")
                self.prev_w_opt = None 
            
            iteration_count += 1
            if iteration_count % 10 == 0: 
                print(f"Sim Time: {self.env.now:.2f}s, MPC Iter: {iteration_count}, "
                      f"Pos: [{current_state_for_mpc_np[0]:.2f}, {current_state_for_mpc_np[1]:.2f}, {current_state_for_mpc_np[2]:.2f}]")
                
                # print the mean MPC compute time
                if self.mpc_compute_time_log:
                    mean_mpc_time = np.mean(self.mpc_compute_time_log)
                    print(f"Mean MPC Compute Time --------------------------------------------: {mean_mpc_time*1000:.2f} ms")
            yield self.env.timeout(self.mpc_dt)

    def run_simulation(self, until_time):
        print(f"Starting simulation. Plant_dt={self.plant_dt*1000:.1f}ms ({(1/self.plant_dt):.1f}Hz), "
              f"MPC_dt={self.mpc_dt*1000:.1f}ms ({(1/self.mpc_dt):.1f}Hz).")
        self.env.run(until=until_time)
        # export mpc_compute_time_log to files to compare RK4 and collocation
        # if self.mpc_compute_time_log:
        #     np.savetxt("mpc_compute_time_log_collocation.txt", self.mpc_compute_time_log, fmt='%.6f', header='MPC Compute Time (seconds)')
        #     print("MPC compute time log exported to 'mpc_compute_time_log.txt' and 'mpc_compute_time_log.xlsx'.")
        print("Simulation finished.")


class Plotter:
    @staticmethod
    def trajectory(actual_log, predicted_log, start_np, target_np, obstacles_info, scenario_to_run):
        actual_np = np.array(actual_log)
        predicted_np = np.array(predicted_log)
        
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111, projection='3d')
        
        if actual_np.ndim == 2 and actual_np.shape[0] > 0 :
            ax.plot(actual_np[:,0], actual_np[:,1], actual_np[:,2], 'b-', lw=2, label='Simulated Path')
        if predicted_np.ndim == 2 and predicted_np.shape[0] > 0:
            for i in range(min(len(actual_np), len(predicted_np))):
                 ax.plot([actual_np[i,0], predicted_np[i,0]], 
                         [actual_np[i,1], predicted_np[i,1]], 
                         [actual_np[i,2], predicted_np[i,2]], 'r--', alpha=0.4)
            ax.plot([], [], 'r--', label='MPC Predicted Next Step')

        ax.scatter(*start_np[:3], c='lime', marker='o', s=150, label='Start', edgecolors='k', depthshade=True)
        ax.scatter(*target_np[:3], c='red', marker='x', s=150, label='Target', depthshade=True,linewidths=3)
        
        for obs in obstacles_info:
            if obs['type'] == 'cylinder':
                u_cyl = np.linspace(0, 2*np.pi, 30)
                z_cyl = np.linspace(obs['z_min'], obs['z_max'], 2)
                u_grid, z_grid = np.meshgrid(u_cyl, z_cyl)
                x_surf = obs['cx'] + obs['r'] * np.cos(u_grid)
                y_surf = obs['cy'] + obs['r'] * np.sin(u_grid)
                ax.plot_surface(x_surf, y_surf, z_grid, color='dimgray', alpha=0.6)
                ax.plot(obs['cx'] + obs['r']*np.cos(u_cyl), obs['cy'] + obs['r']*np.sin(u_cyl), obs['z_min'], color='k', alpha=0.7)
                ax.plot(obs['cx'] + obs['r']*np.cos(u_cyl), obs['cy'] + obs['r']*np.sin(u_cyl), obs['z_max'], color='k', alpha=0.7)
            elif obs['type'] == 'wall':
                x_coords = [obs['x_min'], obs['x_max'], obs['x_max'], obs['x_min'], obs['x_min']]
                y_coords = [obs['y_min'], obs['y_min'], obs['y_max'], obs['y_max'], obs['y_min']]
                ax.plot(x_coords, y_coords, obs['z_min'], color='k', alpha=0.7)
                ax.plot(x_coords, y_coords, obs['z_max'], color='k', alpha=0.7)
                for i in range(4):
                    ax.plot([x_coords[i], x_coords[i]], [y_coords[i], y_coords[i]], [obs['z_min'], obs['z_max']], color='k', alpha=0.7)
                
                x_face = np.array([[obs['x_min'], obs['x_max']], [obs['x_min'], obs['x_max']]])
                y_face = np.array([[obs['y_min'], obs['y_min']], [obs['y_max'], obs['y_max']]])
                z_face_bottom = np.full_like(x_face, obs['z_min'])
                z_face_top = np.full_like(x_face, obs['z_max'])
                ax.plot_surface(x_face, y_face, z_face_bottom, color='darkgrey', alpha=0.5) 
                ax.plot_surface(x_face, y_face, z_face_top, color='darkgrey', alpha=0.5)    

                y_face_side = np.array([[obs['y_min'], obs['y_max']], [obs['y_min'], obs['y_max']]])
                z_face_side = np.array([[obs['z_min'], obs['z_min']], [obs['z_max'], obs['z_max']]])
                x_face_xmin = np.full_like(y_face_side, obs['x_min'])
                x_face_xmax = np.full_like(y_face_side, obs['x_max'])
                ax.plot_surface(x_face_xmin, y_face_side, z_face_side, color='darkgrey', alpha=0.5) 
                ax.plot_surface(x_face_xmax, y_face_side, z_face_side, color='darkgrey', alpha=0.5) 

                x_face_front = np.array([[obs['x_min'], obs['x_max']], [obs['x_min'], obs['x_max']]])
                z_face_front = np.array([[obs['z_min'], obs['z_min']], [obs['z_max'], obs['z_max']]])
                y_face_ymin = np.full_like(x_face_front, obs['y_min'])
                y_face_ymax = np.full_like(x_face_front, obs['y_max'])
                ax.plot_surface(x_face_front, y_face_ymin, z_face_front, color='darkgrey', alpha=0.5) 
                ax.plot_surface(x_face_front, y_face_ymax, z_face_front, color='darkgrey', alpha=0.5) 

        ax.set_xlabel('X [m]', fontsize=12); ax.set_ylabel('Y [m]', fontsize=12); ax.set_zlabel('Z [m]', fontsize=12)
        ax.set_title('Quadcopter 3D Trajectory with Obstacles', fontsize=14)
        
        all_x = [start_np[0], target_np[0]]; all_y = [start_np[1], target_np[1]]; all_z = [start_np[2], target_np[2]]
        if actual_np.ndim == 2 and actual_np.shape[0] > 0:
            all_x.extend(actual_np[:,0]); all_y.extend(actual_np[:,1]); all_z.extend(actual_np[:,2])
        for obs in obstacles_info:
            if obs['type'] == 'cylinder':
                all_x.extend([obs['cx'] - obs['r'], obs['cx'] + obs['r']])
                all_y.extend([obs['cy'] - obs['r'], obs['cy'] + obs['r']])
            elif obs['type'] == 'wall':
                all_x.extend([obs['x_min'], obs['x_max']])
                all_y.extend([obs['y_min'], obs['y_max']])
            all_z.extend([obs.get('z_min', 0), obs.get('z_max', 0)]) 
        
        if all_x: 
            ax.set_xlim(min(all_x)-0.5, max(all_x)+0.5)
            ax.set_ylim(min(all_y)-0.5, max(all_y)+0.5)
            ax.set_zlim(0, max(all_z)+0.5 if all_z else 5) 

        ax.legend(fontsize=10); ax.grid(True)

        # --- Code to save the plot ---
        output_folder = 'media'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder) # Create the 'media' folder if it doesn't exist
        
        # Define the path to save the plot
        file_path = os.path.join(output_folder, f'trajectory_plot_{scenario_to_run}.png')
        plt.savefig(file_path) # Save the figure
        print(f"Plot saved to: {file_path}")
        # --- End of saving code ---
        plt.show()

    @staticmethod
    def states_vs_time(time_log, actual_log, predicted_log, scenario_to_run):
        actual_np = np.array(actual_log)
        predicted_np = np.array(predicted_log)
        time_np = np.array(time_log)

        labels = ['x (m)','y (m)','z (m)',
                'phi (rad)','theta (rad)','psi (rad)',
                'vx_w (m/s)','vy_w (m/s)','vz_w (m/s)',
                'p (rad/s)','q (rad/s)','r (rad/s)']
        
        if not time_np.size or not actual_np.size:
            print("Not enough data to plot states vs time.")
            return
        
        len_pred = predicted_np.shape[0]
        num_states_to_plot = actual_np.shape[1]
        fig, axs = plt.subplots(4, 3, figsize=(15,12), sharex=True) 
        for i, ax in enumerate(axs.flatten()):
            if i < num_states_to_plot:
                ax.plot(time_np, actual_np[:,i], 'b-', label='Simulated (at MPC rate)')
                if len_pred > 0 and i < predicted_np.shape[1]: 
                    ax.plot(time_np[:len_pred], predicted_np[:len_pred,i], 'r--', label='MPC Predicted Next Step')
                ax.set_title(labels[i])
                ax.legend(fontsize=8); ax.grid(True)
                if i >= (num_states_to_plot - axs.shape[1]): 
                    ax.set_xlabel('Time [s]')
            else:
                ax.axis('off') 
        fig.suptitle('State Trajectories vs. Time', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # --- Code to save the plot ---
        output_folder = 'media'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder) # Create the 'media' folder if it doesn't exist
        
        # Define the path to save the plot
        file_path = os.path.join(output_folder, f'state_trajectories_{scenario_to_run}.png') 
        plt.savefig(file_path) # Save the figure
        print(f"Plot saved to: {file_path}")
        # --- End of saving code ---

        plt.show() # Display the plot (optional, remove if you only want to save)


def export_simulation_data(time_log, actual_state_log, txt_filename="drone_data.txt", excel_filename="drone_data.xlsx"):
    if not time_log or not actual_state_log:
        print("No data available to export.")
        return

    times = np.array(time_log)
    states = np.array(actual_state_log) 

    if states.ndim != 2 or states.shape[1] < 6: 
        print(f"State log has unexpected shape: {states.shape}. Cannot export.")
        return

    data_to_export = {
        'Time (s)': times,
        'X_pos (m)': states[:, 0],
        'Y_pos (m)': states[:, 1],
        'Z_pos (m)': states[:, 2],
        'Phi_rot (rad)': states[:, 3],
        'Theta_rot (rad)': states[:, 4],
        'Psi_rot (rad)': states[:, 5]
    }
    txt_filename = os.path.join("data", txt_filename)
    excel_filename = os.path.join("data", excel_filename)
    try:
        with open(txt_filename, 'w') as f_txt:
            header = ",".join(data_to_export.keys())
            f_txt.write(header + "\n")
            for i in range(len(times)):
                row_values = [
                    f"{times[i]:.3f}",
                    f"{states[i, 0]:.4f}", f"{states[i, 1]:.4f}", f"{states[i, 2]:.4f}",
                    f"{states[i, 3]:.4f}", f"{states[i, 4]:.4f}", f"{states[i, 5]:.4f}"
                ]
                f_txt.write(",".join(row_values) + "\n")
        print(f"Drone data successfully exported to {txt_filename}")
    except IOError as e:
        print(f"Error exporting data to TXT file {txt_filename}: {e}")

    if PANDAS_AVAILABLE:
        try:
            df = pd.DataFrame(data_to_export)
            df.to_excel(excel_filename, index=False, engine='openpyxl')
            print(f"Drone data successfully exported to {excel_filename}")
        except Exception as e:
            print(f"Error exporting data to Excel file {excel_filename}: {e}")
    else:
        print(f"Skipping Excel export as pandas is not available.")


# def plot_u(time_log, u_log, scenario_to_run):
#     actual_np = np.array(u_log)  # reshape to 4 lines
#     time_np = np.array(time_log)
    
#     # Plot each of the 4 lines
#     for i in range(4):
#         plt.plot(time_np, actual_np[:,i,0], label=f'Line {i+1}')

#     plt.xlabel('Time')
#     plt.ylabel('u_log values')
#     plt.title(f'Plot of u_log for scenario {scenario_to_run}')
#     plt.legend()
#     plt.show()  # Display the plot


def plot_u(time_log, u_log, scenario_to_run):
    actual_np = np.array(u_log)  # shape: (N, 4, ?)
    time_np = np.array(time_log)
    print("the shape of actual_np", actual_np.shape)
    print("the shape of time_np", time_np.shape)
    
    plt.style.use('seaborn-darkgrid')  # Apply a nice style with grid
    
    # Define colors and line styles for clarity
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'd']
    
    plt.figure(figsize=(10, 6))  # Larger figure for better visibility
    
    # Plot each of the 4 lines with distinct styles
    for i in range(4):
        plt.plot(time_np, actual_np[:, i, 0], 
                 label=f'Line {i+1}',
                 color=colors[i], 
                 linestyle=linestyles[i], 
                 marker=markers[i], 
                 markersize=5, 
                 linewidth=2)
    
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('u_log values', fontsize=14)
    plt.title(f'Plot of u_log for scenario {scenario_to_run}', fontsize=16, fontweight='bold')
    
    plt.legend(title='Lines', fontsize=12)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)  # Major and minor grid
    
    # Optionally customize ticks (example: set major ticks every 1 unit)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', length=7, width=1.2)
    plt.tick_params(axis='both', which='minor', length=4, color='gray')
    
    plt.tight_layout()  # Adjust layout to prevent clipping
    
    plt.show()  # Display the plot