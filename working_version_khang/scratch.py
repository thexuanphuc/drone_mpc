import casadi as ca
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def shift_movement(T, t0, x0, u, x_f, f):
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
    T = 0.1
    N = 25
    m = 0.5
    g_gravity = 9.81
    L = 0.25
    c = 0.01
    Ixx = 2.32e-3
    Iyy = 2.32e-3
    Izz = 4.00e-3

    # Actual geometric parameters of the obstacles
    obstacles_params_geometry = [
        {'name': 'Obstacle1', 'center_x': 2.7, 'center_y': 2.7, 'radius': 0.05, 'height_min': 0.0, 'height_max': 3},
        {'name': 'Obstacle2', 'center_x': 3.0, 'center_y': 2.8, 'radius': 0.05, 'height_min': 0.0, 'height_max': 3},
        {'name': 'Obstacle3', 'center_x': 2.8, 'center_y': 3.0, 'radius': 0.05, 'height_min': 0.0, 'height_max': 3}
    ]
    # For visualization, use finite height
    obstacles_for_plotting = [
        {'name': 'Obstacle1', 'center_x': 2.7, 'center_y': 2.7, 'radius': 0.05, 'height_min': 0.0, 'height_max': 3.0},
        {'name': 'Obstacle2', 'center_x': 3.0, 'center_y': 2.8, 'radius': 0.05, 'height_min': 0.0, 'height_max': 3.0},
        {'name': 'Obstacle3', 'center_x': 2.8, 'center_y': 3.0, 'radius': 0.05, 'height_min': 0.0, 'height_max': 3.0}
    ]
    
    # Safety buffer to add to the radius for planning
    planning_radius_safety_buffer = 0.1 # User suggested 0.1m

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

    X_nlp = ca.SX.sym('X_nlp', n_states, N + 1)
    U_nlp = ca.SX.sym('U_nlp', n_controls, N)
    P_param = ca.SX.sym('P_param', n_states + n_states)

    Q_diag = [20, 20, 20, 50, 50, 50, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1]
    Q = ca.diag(Q_diag)
    Q_terminal = ca.diag([100, 100, 100, 200, 200, 200, 2, 2, 2, 0.5, 0.5, 0.5])
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

        current_state_in_horizon = X_nlp[:, i+1]
        for obs_geom in obstacles_params_geometry: 
            dx = current_state_in_horizon[0] - obs_geom['center_x']
            dy = current_state_in_horizon[1] - obs_geom['center_y']
            # effective_radius = obs_geom['radius'] + planning_radius_safety_buffer
            horizontal_clearance_sq = (dx**2 + dy**2) - (obs_geom['radius'] + planning_radius_safety_buffer)**2

            obstacle_constraint_expr = horizontal_clearance_sq
            g_ineq_constraints.append(obstacle_constraint_expr)

    state_dev_terminal = X_nlp[:, N] - P_param[n_states:2*n_states]
    obj += ca.mtimes([state_dev_terminal.T, Q_terminal, state_dev_terminal])
    all_g_constraints_list = g_eq_constraints + g_ineq_constraints
    opt_variables = ca.vertcat(ca.reshape(U_nlp, -1, 1), ca.reshape(X_nlp, -1, 1))
    nlp_prob = {'f': obj, 'x': opt_variables, 'p': P_param, 'g': ca.vertcat(*all_g_constraints_list)}

    opts = {
        'ipopt.max_iter': 800,
        'ipopt.print_level': 3, # CRITICAL: Set to 3 or higher
        'print_time': 0,
        'ipopt.acceptable_tol': 1e-9, 
        'ipopt.acceptable_obj_change_tol': 1e-7, 
        'ipopt.warm_start_init_point': 'yes',
        'ipopt.mu_strategy': 'adaptive', 
        'ipopt.hessian_approximation': 'exact'
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    lbg = []
    ubg = []
    for _ in range(len(g_eq_constraints)):
        lbg.extend([0.0] * n_states)
        ubg.extend([0.0] * n_states)
    
    # For inequality constraints: horizontal_clearance_sq (with buffered radius) >= 0
    ineq_lbg_margin = 0.0 # Set to 0.0 if radius buffer is primary, or small e.g. 1e-5 for extra strictness
    lbg.extend([ineq_lbg_margin] * len(g_ineq_constraints))
    ubg.extend([ca.inf] * len(g_ineq_constraints))

    lbx = []; ubx = []
    min_thrust = 0.0; max_thrust = 15.0
    for _ in range(N):
        lbx.extend([min_thrust] * n_controls)
        ubx.extend([max_thrust] * n_controls)
    min_xyz = -5.0; max_xyz = 5.0
    min_angle = -ca.pi/4; max_angle = ca.pi/4
    max_psi = ca.pi; max_vel = 2.0; max_ang_vel = ca.pi
    for _ in range(N + 1):
        lbx.extend([min_xyz, min_xyz, 0.0] + [min_angle, min_angle, -max_psi] + [-max_vel]*3 + [-max_ang_vel]*3)
        ubx.extend([max_xyz, max_xyz, max_xyz] + [max_angle, max_angle, max_psi] + [max_vel]*3 + [max_ang_vel]*3)

    t0 = 0.0
    x0 = np.array([2.5, 2.5, 1.5] + [0.0] * 9).reshape(-1, 1)
    x0_history = x0.copy()
    xs = np.array([3.0, 3.0, 2.0] + [0.0] * 9).reshape(-1, 1)
    hover_thrust = (m * g_gravity) / 4.0
    u_guess = np.full((N, n_controls), hover_thrust)
    x_guess = np.tile(x0, (1, N + 1))

    sim_x_trajectory = []; sim_u_applied = []; sim_time_steps = []
    actual_states_history = [x0]
    sim_total_time = 10.0
    mpciter = 0
    solver_times = []
    main_loop_start_time = time.time()

    while np.linalg.norm(x0[:3] - xs[:3]) > 0.01 and mpciter * T < sim_total_time:
        current_params = np.vstack((x0, xs))
        initial_nlp_guess = ca.vertcat(u_guess.reshape(-1, 1), x_guess.reshape(-1, 1))
        iter_start_time = time.time()
        try:
            res = solver(x0=initial_nlp_guess, p=current_params, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
            iter_end_time = time.time()
            solver_times.append(iter_end_time - iter_start_time)

            optimal_u_vector = np.array(res['x'][:n_controls*N])
            optimal_x_vector = np.array(res['x'][n_controls*N:])
            u_optimal_horizon = optimal_u_vector.reshape((N, n_controls))
            x_optimal_horizon = optimal_x_vector.reshape((N + 1, n_states))

            if mpciter < 3 and obstacles_params_geometry: # Debug print for first few iterations
                print(f"\n--- MPC Iteration {mpciter + 1} ({t0:.2f}s) - Obstacle Clearance Debug ---")
                obs_geom_to_check = obstacles_params_geometry[0] 
                effective_radius_check = obs_geom_to_check['radius'] + planning_radius_safety_buffer
                print(f"Checking clearance for: {obs_geom_to_check['name']} at ({obs_geom_to_check['center_x']:.2f}, {obs_geom_to_check['center_y']:.2f}), Actual_R={obs_geom_to_check['radius']:.2f}, Effective_R_Plan={effective_radius_check:.2f}")
                lbg_val_to_check = ineq_lbg_margin 
                for k_step in range(N): 
                    predicted_state = x_optimal_horizon[k_step + 1, :]
                    dx_pred = predicted_state[0] - obs_geom_to_check['center_x']
                    dy_pred = predicted_state[1] - obs_geom_to_check['center_y']
                    pred_dist_sq = dx_pred**2 + dy_pred**2
                    pred_dist_xy = np.sqrt(pred_dist_sq)
                    # This is clearance sq against EFFECTIVE radius
                    h_clearance_sq_pred_vs_effective_r = pred_dist_sq - effective_radius_check**2
                    
                    print(f"  Hrz k={k_step}: Prd(x,y,z)=({predicted_state[0]:.3f},{predicted_state[1]:.3f},{predicted_state[2]:.3f})")
                    print(f"    DistXY_to_Center={pred_dist_xy:.4f} (EffectivePlan_R={effective_radius_check:.2f}), HClrSq_vs_Eff_R={h_clearance_sq_pred_vs_effective_r:.6f} (LBG>={lbg_val_to_check:.2e})")

                    if h_clearance_sq_pred_vs_effective_r < (lbg_val_to_check - 1e-9):
                        print(f"    !!!!!! PLANNED VIOLATION of Effective Radius at horizon k={k_step} !!!!!!")
            
            sim_x_trajectory.append(x_optimal_horizon)
            sim_u_applied.append(u_optimal_horizon[0, :])
            sim_time_steps.append(t0)
            t0, x0, u_guess, x_guess = shift_movement(T, t0, x0, u_optimal_horizon, x_optimal_horizon, f)
            actual_states_history.append(x0)
            mpciter += 1
            print(f"Iter: {mpciter}, Time: {t0:.2f}s, Pos: [{x0[0,0]:.2f}, {x0[1,0]:.2f}, {x0[2,0]:.2f}], Dist: {np.linalg.norm(x0[:3] - xs[:3]):.3f}")

        except RuntimeError as e:
            print(f"Solver failed at iteration {mpciter} (Time: {t0:.2f}s): {e}")
            if 'Restoration_Phase_Failed' in str(e) or 'penalty_parameter_too_small' in str(e) or 'Problem_Infeasible' in str(e):
                print("Solver indicates potential infeasibility or numerical difficulties. Try relaxing constraints/tolerances or increasing obstacle radius if it's too small.")
            break
            
    main_loop_total_time = time.time() - main_loop_start_time
    print(f"\nTotal MPC loop time: {main_loop_total_time:.4f} seconds")

    if solver_times:
        solver_times_np = np.array(solver_times)
        print(f"Average solve time: {solver_times_np.mean():.4f} seconds (Max: {solver_times_np.max():.4f}, Min: {solver_times_np.min():.4f})")
    else:
        print("No successful solves completed.")
    print(f"Total simulation time elapsed: {t0:.2f} seconds")
    if len(actual_states_history) > 1:
      final_pos = actual_states_history[-1][:3].flatten()
      print(f"Final position: {final_pos}, Target: {xs[:3].flatten()}, Dist: {np.linalg.norm(final_pos - xs[:3]):.4f} m")
    else:
      print("Simulation did not run or complete any steps.")

    fig1 = plt.figure(figsize=(12, 9))
    ax1 = fig1.add_subplot(111, projection='3d')
    if len(actual_states_history) > 1:
        actual_states_np = np.hstack(actual_states_history).T
        ax1.plot(actual_states_np[:, 0], actual_states_np[:, 1], actual_states_np[:, 2], 'b-', linewidth=2, label='Actual Trajectory')
    ax1.plot(x0_history[0,0], x0_history[1,0], x0_history[2,0], 'go', markersize=10, label='Start')
    ax1.plot(xs[0,0], xs[1,0], xs[2,0], 'rx', markersize=10, markeredgewidth=3, label='Target')

    for obs_idx, obs_plot_params in enumerate(obstacles_for_plotting):
        u_cyl = np.linspace(0, 2 * np.pi, 30)
        h_cyl = np.linspace(obs_plot_params['height_min'], obs_plot_params['height_max'], 2)
        u_grid, h_grid = np.meshgrid(u_cyl, h_cyl)
        x_surf = obs_plot_params['center_x'] + obs_plot_params['radius'] * np.cos(u_grid) # Use actual radius for plotting
        y_surf = obs_plot_params['center_y'] + obs_plot_params['radius'] * np.sin(u_grid) # Use actual radius for plotting
        z_surf = h_grid
        ax1.plot_surface(x_surf, y_surf, z_surf, color='grey', alpha=0.4, label=f'{obs_plot_params["name"]} (actual radius)')
        
        # Plot effective radius for planning (optional, for debugging visualization)
        # effective_radius_plot = obs_plot_params['radius'] + planning_radius_safety_buffer
        # x_surf_eff = obs_plot_params['center_x'] + effective_radius_plot * np.cos(u_grid)
        # y_surf_eff = obs_plot_params['center_y'] + effective_radius_plot * np.sin(u_grid)
        # ax1.plot_surface(x_surf_eff, y_surf_eff, z_surf, color='red', alpha=0.1, label=f'{obs_plot_params["name"]} (effective plan radius)')

        x_circle_cap = obs_plot_params['center_x'] + obs_plot_params['radius'] * np.cos(u_cyl)
        y_circle_cap = obs_plot_params['center_y'] + obs_plot_params['radius'] * np.sin(u_cyl)
        ax1.plot(x_circle_cap, y_circle_cap, obs_plot_params['height_max'], color='black', alpha=0.7)
        ax1.plot(x_circle_cap, y_circle_cap, obs_plot_params['height_min'], color='black', alpha=0.7)

    ax1.set_xlabel('X [m]', fontsize=12); ax1.set_ylabel('Y [m]', fontsize=12); ax1.set_zlabel('Z [m]', fontsize=12)
    ax1.set_title('Quadcopter MPC Trajectory (Buffered Radius, Infinite Z Obstacle)', fontsize=14)
    all_points_x = [x0_history[0,0], xs[0,0]]; all_points_y = [x0_history[1,0], xs[1,0]]; all_points_z = [x0_history[2,0], xs[2,0]]
    if len(actual_states_history) > 1:
        all_points_x.extend(actual_states_np[:,0].tolist()); all_points_y.extend(actual_states_np[:,1].tolist()); all_points_z.extend(actual_states_np[:,2].tolist())
    for obs_plot_params in obstacles_for_plotting: 
        all_points_x.extend([obs_plot_params['center_x'] - (obs_plot_params['radius'] + planning_radius_safety_buffer), obs_plot_params['center_x'] + (obs_plot_params['radius'] + planning_radius_safety_buffer)])
        all_points_y.extend([obs_plot_params['center_y'] - (obs_plot_params['radius'] + planning_radius_safety_buffer), obs_plot_params['center_y'] + (obs_plot_params['radius'] + planning_radius_safety_buffer)])
        all_points_z.extend([obs_plot_params['height_min'], obs_plot_params['height_max']])
    ax1.set_xlim(min(all_points_x)-0.5, max(all_points_x)+0.5); ax1.set_ylim(min(all_points_y)-0.5, max(all_points_y)+0.5); ax1.set_zlim(0, max(all_points_z)+0.5)
    
    # Ensure legend handles unique labels
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), fontsize=10)
    ax1.grid(True)
    plt.savefig('quadcopter_trajectory_buffered_radius.png')

    if len(actual_states_history) > 1 and sim_time_steps:
        actual_states_for_plot = np.hstack(actual_states_history).T; time_array_for_plot = np.array(sim_time_steps)
        ref_states_for_plot = np.tile(xs.T, (len(time_array_for_plot), 1)); fig2, axs = plt.subplots(2, 3, figsize=(18, 10))
        state_labels = ['X [m]', 'Y [m]', 'Z [m]', 'Roll (φ) [rad]', 'Pitch (θ) [rad]', 'Yaw (ψ) [rad]']
        plot_len = len(time_array_for_plot)
        for i in range(6):
            row, col = divmod(i, 3)
            axs[row, col].plot(time_array_for_plot, actual_states_for_plot[1:plot_len+1, i], 'b-', label='Actual')
            axs[row, col].plot(time_array_for_plot, ref_states_for_plot[:plot_len, i], 'r--', label='Reference')
            axs[row, col].set_xlabel('Time [s]', fontsize=10); axs[row, col].set_ylabel(state_labels[i], fontsize=10)
            axs[row, col].set_title(f'{state_labels[i]} vs. Time', fontsize=12)
            axs[row, col].grid(True); axs[row, col].legend(fontsize=9)
        fig2.suptitle('State Tracking (Buffered Radius, Infinite Z Obstacle)', fontsize=16); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('state_comparison_buffered_radius.png')
    plt.show()