import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import simpy
import time # For simple profiling if needed

class QuadcopterMPC:
    def __init__(self, mpc_sampling_time=0.1): # Added mpc_sampling_time as argument
        # MPC parameters
        self.T = mpc_sampling_time      # sampling time (crucial: matches MPC execution rate)
        self.N = 25                     # prediction horizon (from scratch.py)
        
        # Physical parameters from scratch.py
        self.m = 0.5                    # Mass of the quadrotor [kg]
        self.g = 9.81                   # Gravity [m/s^2]
        self.Ix = 2.32e-3               # Moment of inertia about Bx axis [kg.m^2]
        self.Iy = 2.32e-3               # Moment of inertia about By axis [kg.m^2]
        self.Iz = 4.00e-3               # Moment of inertia about Bz axis [kg.m^2]
        self.L_arm = 0.25               # Quadrotor arm length [m]
        self.c_yaw_coeff = 0.01         # Yaw torque coefficient 
        
        # Obstacles
        self.obstacles = [
            {'cx':3.0,'cy':3.0,'r':0.2},
            {'cx':3.5,'cy':4.5,'r':0.2},
            {'cx':4.5,'cy':3.5,'r':0.2},
            {'cx':5.5,'cy':4.0,'r':0.2},
            {'cx':4.5,'cy':5.5,'r':0.2}

        ]
        self.buffer = 0.1               # Safety buffer for obstacles
        self.n_states = 12              # Will be confirmed by states.numel()
        self.n_controls = 4             # Will be confirmed by controls.numel()


    def mpc_initialize(self):
        # State vector: (x,y,z, phi,theta,psi, vx_w,vy_w,vz_w, p,q,r)
        x_pos = ca.SX.sym('x_pos'); y_pos = ca.SX.sym('y_pos'); z_pos = ca.SX.sym('z_pos')
        phi = ca.SX.sym('phi'); theta = ca.SX.sym('theta'); psi = ca.SX.sym('psi')
        vx_w = ca.SX.sym('vx_w'); vy_w = ca.SX.sym('vy_w'); vz_w = ca.SX.sym('vz_w')
        p_body = ca.SX.sym('p_body'); q_body = ca.SX.sym('q_body'); r_body = ca.SX.sym('r_body')

        states = ca.vertcat(x_pos, y_pos, z_pos, phi, theta, psi, vx_w, vy_w, vz_w, p_body, q_body, r_body)
        self.n_states = states.numel()

        T1 = ca.SX.sym('T1'); T2 = ca.SX.sym('T2'); T3 = ca.SX.sym('T3'); T4 = ca.SX.sym('T4')
        controls = ca.vertcat(T1, T2, T3, T4)
        self.n_controls = controls.numel()

        U1_total_thrust = ca.sum1(controls) 
        tau_phi   = self.L_arm * (-controls[0] + controls[1] + controls[2] - controls[3])
        tau_theta = self.L_arm * (-controls[0] - controls[1] + controls[2] + controls[3])
        tau_psi   = self.c_yaw_coeff * (-controls[0] + controls[1] - controls[2] + controls[3])

        s_phi = states[3]; s_theta = states[4]; s_psi = states[5]
        s_vx_w = states[6]; s_vy_w = states[7]; s_vz_w = states[8]
        s_p_body = states[9]; s_q_body = states[10]; s_r_body = states[11]

        cphi = ca.cos(s_phi); sphi = ca.sin(s_phi)
        ctheta = ca.cos(s_theta); stheta = ca.sin(s_theta)
        cpsi = ca.cos(s_psi); spsi = ca.sin(s_psi)
        ttheta = stheta / ctheta 

        x_pos_dot = s_vx_w; y_pos_dot = s_vy_w; z_pos_dot = s_vz_w
        phi_dot   = s_p_body + s_q_body*sphi*ttheta + s_r_body*cphi*ttheta
        theta_dot =              s_q_body*cphi         - s_r_body*sphi
        psi_dot   =             (s_q_body*sphi + s_r_body*cphi) / ctheta
        vx_w_dot = (U1_total_thrust / self.m) * (cphi*stheta*cpsi + sphi*spsi)
        vy_w_dot = (U1_total_thrust / self.m) * (cphi*stheta*spsi - sphi*cpsi)
        vz_w_dot = (U1_total_thrust / self.m) * (cphi*ctheta) - self.g
        p_body_dot = (tau_phi   + (self.Iy - self.Iz) * s_q_body * s_r_body) / self.Ix
        q_body_dot = (tau_theta + (self.Iz - self.Ix) * s_p_body * s_r_body) / self.Iy
        r_body_dot = (tau_psi   + (self.Ix - self.Iy) * s_p_body * s_q_body) / self.Iz

        RHS = ca.vertcat(
            x_pos_dot, y_pos_dot, z_pos_dot, phi_dot, theta_dot, psi_dot,
            vx_w_dot, vy_w_dot, vz_w_dot, p_body_dot, q_body_dot, r_body_dot
        )
        self.f = ca.Function('f', [states, controls], [RHS])

        X = ca.SX.sym('X', self.n_states, self.N + 1)
        U = ca.SX.sym('U', self.n_controls, self.N)
        P = ca.SX.sym('P', self.n_states * 2)

        # Q_diag_values = [20,20,20, 50,50,50, 0.5,0.5,0.5, 0.1,0.1,0.1]
        Q_diag_values = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
        Q = ca.diag(Q_diag_values)
        R_diag_values = [3.0, 3.0, 3.0, 3.0] 
        R = ca.diag(R_diag_values)
        Q_terminal_diag_values = [100,100,100, 200,200,200, 2,2,2, 0.5,0.5,0.5]
        Qf = ca.diag(Q_terminal_diag_values)

        obj = 0
        g_eq_constraints = [X[:,0] - P[:self.n_states]]
        g_ineq_constraints = []

        for k in range(self.N):
            st = X[:,k]; ct = U[:,k]
            e = st - P[self.n_states:(2*self.n_states)] 
            obj += ca.mtimes([e.T, Q, e]) + ca.mtimes([ct.T, R, ct])
            k1 = self.f(st, ct); k2 = self.f(st + self.T/2*k1, ct)
            k3 = self.f(st + self.T/2*k2, ct); k4 = self.f(st + self.T*k3, ct)
            st_next_rk4 = st + self.T/6*(k1 + 2*k2 + 2*k3 + k4)
            g_eq_constraints.append(X[:,k+1] - st_next_rk4)
            current_state_in_horizon = X[:, k+1]
            for obs in self.obstacles:
                dx = current_state_in_horizon[0] - obs['cx'] 
                dy = current_state_in_horizon[1] - obs['cy'] 
                g_ineq_constraints.append(dx**2 + dy**2 - (obs['r'] + self.buffer)**2)

        eT = X[:,self.N] - P[self.n_states:(2*self.n_states)]
        obj += ca.mtimes([eT.T, Qf, eT])

        opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))
        all_g_constraints_list = g_eq_constraints + g_ineq_constraints
        g_constraints_stacked = ca.vertcat(*all_g_constraints_list)
        
        nlp = {'x': opt_variables, 'f': obj, 'g': g_constraints_stacked, 'p': P}
        opts = {'ipopt.max_iter': 800, 'ipopt.print_level': 0, 
                'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-7}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        lbg_list = [0.0] * self.n_states * (self.N + 1)
        ubg_list = [0.0] * self.n_states * (self.N + 1)
        lbg_list.extend([0.0] * len(g_ineq_constraints))
        ubg_list.extend([ca.inf] * len(g_ineq_constraints))

        lb_controls_list = [0.0] * (self.N * self.n_controls)
        ub_controls_list = [15.0] * (self.N * self.n_controls)
        state_bounds_lb_template = [-15.0,-15.0,0.0, -ca.pi/4,-ca.pi/4,-ca.pi, -2.0,-2.0,-2.0, -3*ca.pi,-3*ca.pi,-3*ca.pi]
        state_bounds_ub_template = [15.0,15.0,5.0, ca.pi/4,ca.pi/4,ca.pi, 2.0,2.0,2.0, 3*ca.pi,3*ca.pi,3*ca.pi]
        lb_states_list = []; ub_states_list = []
        for _ in range(self.N + 1):
            lb_states_list.extend(state_bounds_lb_template)
            ub_states_list.extend(state_bounds_ub_template)
        lbx_list = lb_controls_list + lb_states_list
        ubx_list = ub_controls_list + ub_states_list
        
        self.args = {'lbg': ca.vertcat(*lbg_list), 'ubg': ca.vertcat(*ubg_list),
                     'lbx': ca.vertcat(*lbx_list), 'ubx': ca.vertcat(*ubx_list)}

    def run_once(self, x0_np, xt_np):
        x0_ca = ca.DM(x0_np) # Ensure CasADi DM type
        xt_ca = ca.DM(xt_np)
        P_val = ca.vertcat(x0_ca, xt_ca)
        init_guess_size = self.n_controls * self.N + self.n_states * (self.N + 1)
        init_w = ca.DM.zeros(init_guess_size, 1)
        
        sol = self.solver(x0=init_w, p=P_val, **self.args)
        w_opt = sol['x']
        
        u_seq = ca.reshape(w_opt[:self.n_controls*self.N], self.n_controls, self.N)
        x_seq = ca.reshape(w_opt[self.n_controls*self.N:], self.n_states, self.N+1)
        return u_seq, x_seq


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

            u_sequence_ca, x_sequence_ca = self.mpc.run_once(current_state_for_mpc_np, self.target_state_np)
            
            self.latest_control_input_np = u_sequence_ca[:, 0].full().flatten()

            # Log data at MPC rate
            self.time_log.append(self.env.now)
            self.actual_state_log.append(current_state_for_mpc_np) 
            self.mpc_predicted_log.append(x_sequence_ca[:, 1].full().flatten()) 
            self.control_input_log.append(self.latest_control_input_np.copy())
            
            iteration_count += 1
            if iteration_count % 10 == 0: # Print status every 10 MPC steps
                print(f"Sim Time: {self.env.now:.2f}s, MPC Iter: {iteration_count}, "
                      f"Pos: [{current_state_for_mpc_np[0]:.2f}, {current_state_for_mpc_np[1]:.2f}, {current_state_for_mpc_np[2]:.2f}]")

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
            ax.plot(actual_np[:,0], actual_np[:,1], actual_np[:,2], 'b-', label='Simulated Path (at MPC rate)')
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
        h_cyl_abs = 3.0 # Assuming a common height for plotting obstacles if not specified
        for obs in obstacles:
            h_plot = np.linspace(0, h_cyl_abs, 2) # Finite height for visualization
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
                    ax.plot(time_np, actual_np[:,i], 'b-', label='Simulated (at MPC rate)')
                if predicted_np.shape[0] > 0 and i < predicted_np.shape[1]: # Check if there's predicted data for this state
                    ax.plot(time_np, predicted_np[:,i], 'r--', label='MPC Predicted Next Step')
                ax.set_title(labels[i])
                ax.legend()
                ax.grid(True)
            else:
                ax.axis('off') # Hide unused subplots
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    mpc_freq = 10.0  # Hz
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