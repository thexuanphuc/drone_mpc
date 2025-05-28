import simpy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import casadi as ca
import time

class AdvancedSimulator:
    def __init__(self, mpc_controller, initial_state_np, target_state_np, plant_dt=0.01, mpc_dt=0.1):
        self.env = simpy.Environment()
        self.mpc = mpc_controller # This is an instance of QuadcopterMPC
        self.plant_dt = plant_dt
        self.mpc_dt = mpc_dt

        if abs(self.mpc.T - self.mpc_dt) > 1e-6:
            print(f"Warning: MPC internal sampling time mpc.T ({self.mpc.T}s) "
                  f"differs from simulator's mpc_dt ({self.mpc_dt}s). "
                  f"Ensure mpc.T is set to {self.mpc_dt} for consistency.")
            # It's critical that mpc.T (used in MPC's internal model) matches mpc_dt
            # The QuadcopterMPC is now initialized with mpc_sampling_time, so this should be fine.

        self.current_state_np = np.array(initial_state_np).flatten() # Ensure it's a flat numpy array
        self.target_state_np = np.array(target_state_np).flatten()
        
        # Initial control input (e.g., hover thrust)
        hover_thrust_per_motor = (self.mpc.m * self.mpc.g) / self.mpc.n_controls
        self.latest_control_input_np = np.full(self.mpc.n_controls, hover_thrust_per_motor)
    
        self.time_log = []
        self.mpc_compute_time_log = []
        self.actual_state_log = []
        self.mpc_predicted_log = []
        self.control_input_log = []
    
        self.env.process(self.plant_process())
        self.env.process(self.mpc_process())

    def plant_process(self):
        """Simulates the drone dynamics at a high frequency using RK4."""
        while True:
            control_to_apply_ca = ca.DM(self.latest_control_input_np) # Convert to CasADi DM
            current_state_ca = ca.DM(self.current_state_np)

            # RK4 integration for plant step
            k1 = self.mpc.f(current_state_ca, control_to_apply_ca).full().flatten()
            k2 = self.mpc.f(current_state_ca + self.plant_dt/2 * k1, control_to_apply_ca).full().flatten()
            k3 = self.mpc.f(current_state_ca + self.plant_dt/2 * k2, control_to_apply_ca).full().flatten()
            k4 = self.mpc.f(current_state_ca + self.plant_dt * k3, control_to_apply_ca).full().flatten()
            self.current_state_np += (self.plant_dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            
            yield self.env.timeout(self.plant_dt)

    def mpc_process(self):
        """Runs the MPC controller at a lower frequency."""
        iteration_count = 0
        while True:
            current_state_for_mpc_np = self.current_state_np.copy()
            start_time = time.time()  # Start timing the MPC computation
            u_sequence_ca, x_sequence_ca = self.mpc.run_once(current_state_for_mpc_np, self.target_state_np)
            
            self.latest_control_input_np = u_sequence_ca[:, 0].full().flatten()
            end_time = time.time()  # End timing the MPC computation
            self.mpc_compute_time_log.append(end_time - start_time)
            # Log data at MPC rate
            self.time_log.append(self.env.now)
            self.actual_state_log.append(current_state_for_mpc_np) 
            self.mpc_predicted_log.append(x_sequence_ca[:, 1].full().flatten()) 
            self.control_input_log.append(self.latest_control_input_np.copy())
            
            iteration_count += 1
            if iteration_count % 10 == 0: # Print status every 10 MPC steps
                print(f"Sim Time: {self.env.now:.2f}s, MPC Iter: {iteration_count}, "
                      f"Pos: [{current_state_for_mpc_np[0]:.2f}, {current_state_for_mpc_np[1]:.2f}, {current_state_for_mpc_np[2]:.2f}]")
                # log the mean MPC compute time
                if self.mpc_compute_time_log:
                    mean_mpc_time = np.mean(self.mpc_compute_time_log)
                    print(f"Mean MPC Compute Time --------------------------------------------: {mean_mpc_time*1000:.2f} ms")
            # plot the result in the end
            if iteration_count >= 3000:  # Limit the number of MPC iterations for testing 
                # plot MPC compute time over time
                plt.figure(figsize=(10, 5))
                plt.plot(self.time_log, self.mpc_compute_time_log, label='MPC Compute Time', color='orange')
                plt.xlabel('Simulation Time (s)')
                plt.ylabel('MPC Compute Time (s)')
                plt.title('MPC Compute Time Over Simulation')
                plt.grid()
                plt.legend()
                plt.show()
                   
            yield self.env.timeout(self.mpc_dt)

    def run_simulation(self, until_time):
        print(f"Starting simulation. Plant_dt={self.plant_dt*1000:.1f}ms ({(1/self.plant_dt):.1f}Hz), "
              f"MPC_dt={self.mpc_dt*1000:.1f}ms ({(1/self.mpc_dt):.1f}Hz).")
        self.env.run(until=until_time)
        print("Simulation finished.")

class Plotter: # From original MPC_v5.py
    @staticmethod
    def trajectory(actual_log, predicted_log, start_np, target_np, obstacles):
        # Convert logs to numpy arrays for plotting if they are lists of arrays
        actual_np = np.array(actual_log)
        predicted_np = np.array(predicted_log)
        
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        
        if actual_np.ndim == 2 and actual_np.shape[0] > 0 :
            ax.plot(actual_np[:,0], actual_np[:,1], actual_np[:,2], 'b-', label='Simulated Path')
        if predicted_np.ndim == 2 and predicted_np.shape[0] > 0:
             # Plot MPC predicted path (first step of each prediction)
             # To make it more comparable, we can plot segments from actual_np[i] to predicted_np[i]
            for i in range(min(len(actual_np), len(predicted_np))):
                 ax.plot([actual_np[i,0], predicted_np[i,0]], 
                         [actual_np[i,1], predicted_np[i,1]], 
                         [actual_np[i,2], predicted_np[i,2]], 'r--', alpha=0.5)
            # Add a single legend entry for these dashed lines
            ax.plot([], [], 'r--', label='MPC Predicted Next Step')

        ax.scatter(*start_np[:3], c='g', marker='o', s=100, label='Start')
        ax.scatter(*target_np[:3], c='r', marker='x', s=100, label='Target')
        
        u_cyl = np.linspace(0, 2*np.pi, 30)
        for obs in obstacles:
            # Use min_z and max_z if available, otherwise fallback to cz and h_cyl_abs
            min_z = obs.get('min_z', obs.get('cz', 0.0))
            max_z = obs.get('max_z', obs.get('cz', 3.0))
            h_plot = np.linspace(min_z, max_z, 2)
            Uc, Hc = np.meshgrid(u_cyl, h_plot)
            Xc = obs['cx'] + obs['r'] * np.cos(Uc)
            Yc = obs['cy'] + obs['r'] * np.sin(Uc)
            Zc = Hc
            ax.plot_surface(Xc, Yc, Zc, color='gray', alpha=0.3)
        
        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
        ax.set_title('Quadcopter Trajectory')
        ax.legend()
        plt.show()

    @staticmethod
    def states_vs_time(time_log, actual_log, predicted_log):
        actual_np = np.array(actual_log)
        predicted_np = np.array(predicted_log)
        time_np = np.array(time_log)

        # State labels according to (x,y,z, phi,theta,psi, vx_w,vy_w,vz_w, p,q,r)
        labels = ['x (m)','y (m)','z (m)',
                  'phi (rad)','theta (rad)','psi (rad)',
                  'vx_w (m/s)','vy_w (m/s)','vz_w (m/s)',
                  'p (rad/s)','q (rad/s)','r (rad/s)']
        
        if actual_np.shape[0] != len(time_np) or \
           (predicted_np.shape[0] > 0 and predicted_np.shape[0] != len(time_np)):
            print("Warning: Log array lengths mismatch. Plotting might be incomplete or incorrect.")
            print(f"Time log: {len(time_np)}, Actual log: {actual_np.shape[0]}, Predicted log: {predicted_np.shape[0]}")
            # Fallback if predicted_np is empty but actual_np matches time_np
            if predicted_np.shape[0] == 0 and actual_np.shape[0] == len(time_np):
                 predicted_np = np.empty((0, actual_np.shape[1])) # Create empty array with correct num columns
            else: # If lengths are truly problematic, return early
                return

        num_states_to_plot = actual_np.shape[1]
        fig, axs = plt.subplots(4, 3, figsize=(15,12)) # 4 rows, 3 columns for 12 states
        for i, ax in enumerate(axs.flatten()):
            if i < num_states_to_plot:
                if actual_np.shape[0] > 0: # Check if there's data to plot
                    ax.plot(time_np, actual_np[:,i], 'b-', label='Simulated')
                if predicted_np.shape[0] > 0 and i < predicted_np.shape[1]: # Check if there's predicted data for this state
                    ax.plot(time_np, predicted_np[:,i], 'r--', label='MPC Predicted Next Step')
                ax.set_title(labels[i])
                ax.legend()
                ax.grid(True)
            else:
                ax.axis('off') # Hide unused subplots
        plt.tight_layout()
        plt.show()