import casadi as ca
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def shift_movement(T, t0, x0, u, x_f, f):
    """
    Shifts the control and state trajectories for the next MPC iteration using RK4 integration.

    Args:
        T (float): Sampling time in seconds.
        t0 (float): Current simulation time in seconds.
        x0 (np.array): Current state vector (12x1) [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r].
        u (np.array): Optimal control sequence (Nx4) [T1, T2, T3, T4] over the prediction horizon.
        x_f (np.array): Optimal state trajectory (Nx12) over the prediction horizon.
        f (ca.Function): CasADi function representing the system dynamics dx/dt = f(x, u).

    Returns:
        tuple: Updated time (t0 + T), new state, shifted control sequence, shifted state trajectory.
    """
    k1 = f(x0, u[0, :]).full()
    k2 = f(x0 + T/2 * k1, u[0, :]).full()
    k3 = f(x0 + T/2 * k2, u[0, :]).full()
    k4 = f(x0 + T * k3, u[0, :]).full()
    st = x0 + (T/6) * (k1 + 2*k2 + 2*k3 + k4)

    t = t0 + T
    u_end = np.vstack((u[1:], u[-1:]))
    x_f_end = np.vstack((x_f[1:], x_f[-1:]))

    return t, st, u_end, x_f_end

if __name__ == '__main__':
    # --- System Parameters ---
    T = 0.1
    N = 20
    m = 0.5
    g_gravity = 9.81
    L = 0.25
    c = 0.01
    Ixx = 2.32e-3
    Iyy = 2.32e-3
    Izz = 4.00e-3

    # --- Obstacle Parameters ---
    # Define cylindrical obstacles: center_x, center_y, radius, height_min, height_max
    obstacles_params = [
        {'name': 'Obstacle1', 'center_x': 2.8, 'center_y': 2.75, 'radius': 0.15, 'height_min': 0.0, 'height_max': 4.5},
        # {'name': 'Obstacle2', 'center_x': 2.8, 'center_y': 2.9, 'radius': 0.05, 'height_min': 0.0, 'height_max': 2.5}
    ]
    # Drone start: [2.5, 2.5, 1.5], target: [3.0, 3.0, 2.0]

    # --- State and Control Variables ---
    states = ca.vertcat(
        ca.SX.sym('x_pos'), ca.SX.sym('y_pos'), ca.SX.sym('z_pos'),
        ca.SX.sym('phi'), ca.SX.sym('theta'), ca.SX.sym('psi'),
        ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz'),
        ca.SX.sym('p'), ca.SX.sym('q'), ca.SX.sym('r')
    )
    n_states = states.size1()

    controls = ca.vertcat(
        ca.SX.sym('T1'), ca.SX.sym('T2'), ca.SX.sym('T3'), ca.SX.sym('T4')
    )
    n_controls = controls.size1()
    print(f"Number of states: {n_states}, Number of controls: {n_controls}")
    # --- Quadcopter Dynamics ---
    U1 = ca.sum1(controls)
    tau_phi = L * (-controls[0] + controls[1] + controls[2] - controls[3])
    tau_theta = L * (-controls[0] - controls[1] + controls[2] + controls[3])
    tau_psi = c * (-controls[0] + controls[1] - controls[2] + controls[3])

    rhs = ca.vertcat(
        states[6], states[7], states[8],
        states[9] + states[10] * ca.sin(states[3]) * ca.tan(states[4]) + states[11] * ca.cos(states[3]) * ca.tan(states[4]),
        states[10] * ca.cos(states[3]) - states[11] * ca.sin(states[3]),
        states[10] * ca.sin(states[3]) / ca.cos(states[4]) + states[11] * ca.cos(states[3]) / ca.cos(states[4]),
        (U1 / m) * (ca.cos(states[3]) * ca.sin(states[4]) * ca.cos(states[5]) + ca.sin(states[3]) * ca.sin(states[5])),
        (U1 / m) * (ca.cos(states[3]) * ca.sin(states[4]) * ca.sin(states[5]) - ca.sin(states[3]) * ca.cos(states[5])),
        (U1 / m) * (ca.cos(states[3]) * ca.cos(states[4])) - g_gravity,
        (tau_phi + (Iyy - Izz) * states[10] * states[11]) / Ixx,
        (tau_theta + (Izz - Ixx) * states[9] * states[11]) / Iyy,
        (tau_psi + (Ixx - Iyy) * states[9] * states[10]) / Izz
    )
    f = ca.Function('f', [states, controls], [rhs], ['state', 'control'], ['rhs'])

    # --- MPC Setup ---
    X_nlp = ca.SX.sym('X_nlp', n_states, N + 1)
    U_nlp = ca.SX.sym('U_nlp', n_controls, N)
    P_param = ca.SX.sym('P_param', n_states + n_states)

    Q_diag = [20, 20, 20, 50, 50, 50, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1]
    Q = ca.diag(Q_diag)
    Q_terminal = ca.diag([1000, 1000, 1000, 200, 200, 200, 2, 2, 2, 0.5, 0.5, 0.5])
    R_diag = [5.0, 5.0, 5.0, 5.0]
    R = ca.diag(R_diag)

    obj = 0
    g_eq_constraints = [X_nlp[:, 0] - P_param[:n_states]]
    g_ineq_constraints = []

    for i in range(N):
        state_dev = X_nlp[:, i] - P_param[n_states:2*n_states]
        obj += ca.mtimes([state_dev.T, Q, state_dev]) + ca.mtimes([U_nlp[:, i].T, R, U_nlp[:, i]])

        k1 = f(X_nlp[:, i], U_nlp[:, i])
        k2 = f(X_nlp[:, i] + T/2 * k1, U_nlp[:, i])
        k3 = f(X_nlp[:, i] + T/2 * k2, U_nlp[:, i])
        k4 = f(X_nlp[:, i] + T * k3, U_nlp[:, i])
        x_next_rk4 = X_nlp[:, i] + (T/6) * (k1 + 2*k2 + 2*k3 + k4)
        g_eq_constraints.append(X_nlp[:, i+1] - x_next_rk4)

        # current_state_in_horizon = X_nlp[:, i+1]
        for obs in obstacles_params:
            dx = X_nlp[0, i+1] - obs['center_x']
            dy = X_nlp[1, i+1] - obs['center_y']
            dz_top = X_nlp[2, i+1] - obs['height_max']
            dz_bottom = obs['height_min'] - X_nlp[2, i+1]

            horizontal_dist_sq = dx**2 + dy**2

            clearance_margin = 1e-2
            obstacle_clearance_expr = horizontal_dist_sq - (obs['radius'] + clearance_margin)**2
            g_ineq_constraints.append(obstacle_clearance_expr)

            # is_above = X_nlp[2, i+1] > obs['height_max']
            # is_below = X_nlp[2, i+1] < obs['height_min']
            # is_clear_vertically = ca.logic_or(is_above, is_below)
            # is_clear_horizontally = obstacle_clearance_expr > 0
            # # is_clear = ca.logic_or(is_clear_vertically, is_clear_horizontally)
            # is_clear = is_clear_horizontally

            # print(f"Iteration : Obstacle constraint  is_clear_horizontally values \n: {is_clear_horizontally}"  )

            # # # Debug prints for inspection
            # # print(f"[i={i}][{obs['name']}] x={current_state_in_horizon[0]} y={current_state_in_horizon[1]} z={current_state_in_horizon[2]} | dx={dx} dy={dy} dz_top={dz_top} dz_bottom={dz_bottom} | horiz_dist_sq={horizontal_dist_sq} | is_above={is_above} is_below={is_below}")

            # obstacle_constraint_expr = ca.if_else(is_clear, 10.0, -10.0)
            # print("\n\n")
            # g_ineq_constraints.append(obstacle_constraint_expr)
            # print(f"Iteration : Obstacle constraint  is_clear_horizontally values \n: {is_clear_horizontally}"  )

            # # Debug prints for inspection
            # print(f"[i={i}][{obs['name']}] x={current_state_in_horizon[0]} y={current_state_in_horizon[1]} z={current_state_in_horizon[2]} | dx={dx} dy={dy} dz_top={dz_top} dz_bottom={dz_bottom} | horiz_dist_sq={horizontal_dist_sq} | is_above={is_above} is_below={is_below}")

    state_dev_terminal = X_nlp[:, N] - P_param[n_states:2*n_states]
    obj += ca.mtimes([state_dev_terminal.T, Q_terminal, state_dev_terminal])
    print("the shape of g_eq_constraints ", len(g_eq_constraints))

    print("the shape of g_ineq_constraints ", len(g_ineq_constraints))
    # all_g_constraints_list = g_eq_constraints + g_ineq_constraints
    all_g_constraints_list = g_ineq_constraints

    # print("the shape of all_g_constraints_list ", len(all_g_constraints_list))
    
    opt_variables = ca.vertcat(ca.reshape(U_nlp, -1, 1), ca.reshape(X_nlp, -1, 1))
    nlp_prob = {'f': obj, 'x': opt_variables, 'p': P_param, 'g': ca.vertcat(*all_g_constraints_list)}

    opts = {
        'ipopt.max_iter': 500,
        'ipopt.print_level': 0, # Consider increasing to 1 or 2 for detailed solver output
        'print_time': 0,
        'ipopt.acceptable_tol': 1e-10,
        'ipopt.acceptable_obj_change_tol': 1e-8,
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.mu_strategy': 'adaptive',
        'ipopt.hessian_approximation': 'exact'
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    # --- Constraints Bounds ---
    lbg = []
    ubg = []

    # for _ in range(len(g_eq_constraints)):
    #     lbg.extend([0.0] * n_states)
    #     ubg.extend([0.0] * n_states)
        
    # For inequality constraints: horizontal_clearance_sq >= 0 (or small positive margin)
    # To add a small safety margin, change 0.0 to a small positive value e.g., 1e-4
    # This forces the drone to be slightly further than the exact radius.
    # lbg.extend([1e-4] * len(g_ineq_constraints)) # Example with safety margin
    lbg.extend([0.01] * len(g_ineq_constraints))
    ubg.extend([ca.inf] * len(g_ineq_constraints))

    # Bounds for optimization variables
    lbx = []
    ubx = []
    min_thrust = 0.0
    max_thrust = 5.0
    for _ in range(N):
        lbx.extend([min_thrust] * n_controls)
        ubx.extend([max_thrust] * n_controls)

    min_xyz = -5.0
    max_xyz = 5.0
    min_angle = -ca.pi/4
    max_angle = ca.pi/4
    max_psi = ca.pi
    max_vel = 2.0
    max_ang_vel = ca.pi
    for _ in range(N + 1):
        lbx.extend([min_xyz, min_xyz, 0.0] + [min_angle, min_angle, -max_psi] +
                   [-max_vel] * 3 + [-max_ang_vel] * 3)
        ubx.extend([max_xyz, max_xyz, max_xyz] + [max_angle, max_angle, max_psi] +
                   [max_vel] * 3 + [max_ang_vel] * 3)

    # --- Simulation Loop ---
    t0 = 0.0
    x0 = np.array([2.5, 2.5, 1.5] + [0.0] * 9).reshape(-1, 1)
    x0_history = x0.copy()
    xs = np.array([3.0, 3.0, 2.0] + [0.0] * 9).reshape(-1, 1)

    hover_thrust = (m * g_gravity) / 4.0
    u_guess = np.full((N, n_controls), hover_thrust)
    x_guess = np.tile(x0, (1, N + 1))

    sim_x_trajectory = []
    sim_u_applied = []
    sim_time_steps = []
    actual_states_history = [x0]
    sim_total_time = 10.0 # Simulation time
    mpciter = 0
    solver_times = []

    main_loop_start_time = time.time()
    while np.linalg.norm(x0[:3] - xs[:3]) > 0.01 and mpciter * T < sim_total_time:
        current_params = np.vstack((x0, xs))
        initial_nlp_guess = ca.vertcat(u_guess.reshape(-1, 1), x_guess.reshape(-1, 1))
        
        iter_start_time = time.time()
        try:
            res = solver(x0=initial_nlp_guess, p=current_params, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)

            g_constraint_values = res['g'][len(g_eq_constraints):]
            # print(f"Iteration : Inequality constraints values \n: {g_constraint_values}")
            # print("\n\n")

            iter_end_time = time.time()
            solver_times.append(iter_end_time - iter_start_time)

            optimal_u_vector = np.array(res['x'][:n_controls*N])
            optimal_x_vector = np.array(res['x'][n_controls*N:])
            print("the length of optimal_x_vector ", len(optimal_x_vector))
            print("value at the 312 313 314 index of optimal_x_vector ", np.array(res['x'][n_controls*N:n_controls*N+12]))
            u_optimal_horizon = optimal_u_vector.reshape((N, n_controls))
            x_optimal_horizon = optimal_x_vector.reshape((N + 1, n_states))
            
            sim_x_trajectory.append(x_optimal_horizon)
            sim_u_applied.append(u_optimal_horizon[0, :])
            sim_time_steps.append(t0)

            t0, x0, u_guess, x_guess = shift_movement(T, t0, x0, u_optimal_horizon, x_optimal_horizon, f)
            
            actual_states_history.append(x0)
            
            mpciter += 1
            print(f"Iter: {mpciter}, Time: {t0:.2f}s, Pos: [{x0[0,0]:.2f}, {x0[1,0]:.2f}, {x0[2,0]:.2f}], Dist: {np.linalg.norm(x0[:3] - xs[:3]):.3f}")

        except RuntimeError as e:
            print(f"Solver failed at iteration {mpciter} (Time: {t0:.2f}s): {e}")
            break
        
    main_loop_total_time = time.time() - main_loop_start_time
    print(f"\nTotal MPC loop time: {main_loop_total_time:.4f} seconds")

    # --- Results Summary ---
    if solver_times:
        solver_times_np = np.array(solver_times)
        print(f"Average solve time: {solver_times_np.mean():.4f} seconds")
        print(f"Max solve time: {solver_times_np.max():.4f} seconds")
        print(f"Min solve time: {solver_times_np.min():.4f} seconds")
    else:
        print("No successful solves completed.")
        
    print(f"Total simulation time elapsed: {t0:.2f} seconds")
    if len(actual_states_history) > 1:
      final_pos = actual_states_history[-1][:3].flatten()
      print(f"Final position: {final_pos}")
      print(f"Target position: {xs[:3].flatten()}")
      print(f"Distance to target: {np.linalg.norm(final_pos - xs[:3]):.4f} meters")
    else:
      print("Simulation did not run or complete any steps.")

    # --- Visualization ---
    fig1 = plt.figure(figsize=(12, 9))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    if len(actual_states_history) > 1:
        actual_states_np = np.hstack(actual_states_history).T
        ax1.plot(actual_states_np[:, 0], actual_states_np[:, 1], actual_states_np[:, 2], 'b-', linewidth=2, label='Actual Trajectory')
    
    ax1.plot(x0_history[0,0], x0_history[1,0], x0_history[2,0], 'go', markersize=10, label='Start')
    ax1.plot(xs[0,0], xs[1,0], xs[2,0], 'rx', markersize=10, markeredgewidth=3, label='Target')

    for obs_idx, obs in enumerate(obstacles_params):
        u_cyl = np.linspace(0, 2 * np.pi, 30)
        h_cyl = np.linspace(obs['height_min'], obs['height_max'], 2)
        u_grid, h_grid = np.meshgrid(u_cyl, h_cyl)
        
        x_surf = obs['center_x'] + obs['radius'] * np.cos(u_grid)
        y_surf = obs['center_y'] + obs['radius'] * np.sin(u_grid)
        z_surf = h_grid
        
        # Label each obstacle surface individually if desired, or group them
        ax1.plot_surface(x_surf, y_surf, z_surf, color='grey', alpha=0.6, label=f'{obs["name"]}')
        
        x_circle_cap = obs['center_x'] + obs['radius'] * np.cos(u_cyl)
        y_circle_cap = obs['center_y'] + obs['radius'] * np.sin(u_cyl)
        ax1.plot(x_circle_cap, y_circle_cap, obs['height_max'], color='black', alpha=0.7)
        ax1.plot(x_circle_cap, y_circle_cap, obs['height_min'], color='black', alpha=0.7)

    ax1.set_xlabel('X [m]', fontsize=12)
    ax1.set_ylabel('Y [m]', fontsize=12)
    ax1.set_zlabel('Z [m]', fontsize=12)
    ax1.set_title('Quadcopter MPC Trajectory with Obstacle Avoidance', fontsize=14)
    
    all_points_x = [x0_history[0,0], xs[0,0]]
    all_points_y = [x0_history[1,0], xs[1,0]]
    all_points_z = [x0_history[2,0], xs[2,0]]
    if len(actual_states_history) > 1:
        all_points_x.extend(actual_states_np[:,0].tolist())
        all_points_y.extend(actual_states_np[:,1].tolist())
        all_points_z.extend(actual_states_np[:,2].tolist())
    for obs in obstacles_params:
        all_points_x.extend([obs['center_x'] - obs['radius'], obs['center_x'] + obs['radius']])
        all_points_y.extend([obs['center_y'] - obs['radius'], obs['center_y'] + obs['radius']])
        all_points_z.extend([obs['height_min'], obs['height_max']])
    
    ax1.set_xlim(min(all_points_x) - 0.5, max(all_points_x) + 0.5)
    ax1.set_ylim(min(all_points_y) - 0.5, max(all_points_y) + 0.5)
    ax1.set_zlim(0, max(all_points_z) + 0.5)

    ax1.legend(fontsize=10)
    ax1.grid(True)
    plt.savefig('quadcopter_trajectory_with_obstacles.png')

    # 2. State Comparison Plots
    if len(actual_states_history) > 1 and sim_time_steps:
        actual_states_for_plot = np.hstack(actual_states_history).T
        time_array_for_plot = np.array(sim_time_steps)
        
        ref_states_for_plot = np.tile(xs.T, (len(time_array_for_plot), 1))

        fig2, axs = plt.subplots(2, 3, figsize=(18, 10))
        state_labels = ['X [m]', 'Y [m]', 'Z [m]', 'Roll (φ) [rad]', 'Pitch (θ) [rad]', 'Yaw (ψ) [rad]']
        
        plot_len = len(time_array_for_plot)
        for i in range(6):
            row, col = divmod(i, 3)
            axs[row, col].plot(time_array_for_plot, actual_states_for_plot[1:plot_len+1, i], 'b-', label='Actual')
            axs[row, col].plot(time_array_for_plot, ref_states_for_plot[:plot_len, i], 'r--', label='Reference')
            axs[row, col].set_xlabel('Time [s]', fontsize=10)
            axs[row, col].set_ylabel(state_labels[i], fontsize=10)
            axs[row, col].set_title(f'{state_labels[i]} vs. Time', fontsize=12)
            axs[row, col].grid(True)
            axs[row, col].legend(fontsize=9)
        fig2.suptitle('State Tracking: Actual vs Reference (with Obstacles)', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('state_comparison_with_obstacles.png')

    plt.show()