import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle as PlotCircle
import time

# --------------------------------------------------------------------------
# --- 1. MPC and System Parameters -----------------------------------------
# --------------------------------------------------------------------------
N = 20              # Prediction horizon
dt = 0.1            # Time step (discretization time)
T_sim = 25.0        # Total simulation time

# Drone Model Parameters
n_states = 6        # [x, y, z, vx, vy, vz]
n_controls = 3      # [ax, ay, az]

# Weighting Matrices - Increase z-velocity weight
Q = np.diag([20, 20, 100, 2, 2, 10])  # Increased vz weight from 2 to 10
R = np.diag([0.1, 0.1, 0.1])

# Input Constraints (accelerations m/s^2) - Increase az upper bound
u_min = np.array([-5.0, -5.0, -5.0])
u_max = np.array([5.0, 5.0, 15.0])  # Allow az to go up to 15 m/s^2 to counteract gravity

# Drone and Obstacle Parameters
drone_radius = 0.25

# Reference Trajectory Parameters
circle_radius_ref = 5.0
circle_center_z_ref = 2.0
angular_speed_ref = 0.4  # rad/s

# Obstacle Definitions: center (x,y,z), radius
obstacles_params = [
    {'center': np.array([circle_radius_ref, 0, circle_center_z_ref]), 'radius': 0.75},
    {'center': np.array([0, circle_radius_ref, circle_center_z_ref]), 'radius': 0.75},
    {'center': np.array([-circle_radius_ref, 0, circle_center_z_ref]), 'radius': 0.75},
    {'center': np.array([0, -circle_radius_ref, circle_center_z_ref]), 'radius': 0.75},
]

# --------------------------------------------------------------------------
# --- 2. Drone Dynamics Model (Continuous and Discrete - RK4) --------------
# --------------------------------------------------------------------------
x_sym = ca.SX.sym('x', n_states)
u_sym = ca.SX.sym('u', n_controls)

x_dot_expr = ca.vertcat(
    x_sym[3], x_sym[4], x_sym[5],
    u_sym[0], u_sym[1], u_sym[2] - 9.81
)
f_continuous = ca.Function('f_continuous', [x_sym, u_sym], [x_dot_expr])

def rk4_step(f, x_k, u_k, dt_k):
    k1 = f(x_k, u_k)
    k2 = f(x_k + dt_k/2 * k1, u_k)
    k3 = f(x_k + dt_k/2 * k2, u_k)
    k4 = f(x_k + dt_k * k3, u_k)
    x_k_plus_1 = x_k + dt_k/6 * (k1 + 2*k2 + 2*k3 + k4)
    return x_k_plus_1

x_k_sym = ca.SX.sym('x_k_sym', n_states)
u_k_sym = ca.SX.sym('u_k_sym', n_controls)
dt_sym = ca.SX.sym('dt_sym')
f_discrete_rk4 = ca.Function('f_discrete_rk4', [x_k_sym, u_k_sym, dt_sym], 
                              [rk4_step(f_continuous, x_k_sym, u_k_sym, dt_sym)])

# --------------------------------------------------------------------------
# --- 3. Optimal Control Problem (OCP) Formulation -----------------------
# --------------------------------------------------------------------------
opti = ca.Opti()
X_dv = opti.variable(n_states, N + 1)
U_dv = opti.variable(n_controls, N)
x0_param = opti.parameter(n_states, 1)
X_ref_param = opti.parameter(n_states, N + 1)

objective = 0
Q_mat = ca.DM(Q)
R_mat = ca.DM(R)

for k in range(N):
    state_error = X_dv[:, k] - X_ref_param[:, k]
    objective += ca.mtimes([state_error.T, Q_mat, state_error])
    control_input = U_dv[:, k]
    objective += ca.mtimes([control_input.T, R_mat, control_input])

terminal_state_error = X_dv[:, N] - X_ref_param[:, N]
Q_terminal = ca.DM(np.diag([50, 50, 50, 5, 5, 5]))
objective += ca.mtimes([terminal_state_error.T, Q_terminal, terminal_state_error])
opti.minimize(objective)

opti.subject_to(X_dv[:, 0] == x0_param)
for k in range(N):
    x_next_pred = f_discrete_rk4(X_dv[:, k], U_dv[:, k], dt)
    opti.subject_to(X_dv[:, k+1] == x_next_pred)

for k in range(N):
    opti.subject_to(opti.bounded(u_min[0], U_dv[0, k], u_max[0]))
    opti.subject_to(opti.bounded(u_min[1], U_dv[1, k], u_max[1]))
    opti.subject_to(opti.bounded(u_min[2], U_dv[2, k], u_max[2]))

for k in range(1, N + 1):
    pos_k = X_dv[0:3, k]
    for obs in obstacles_params:
        obs_center_dm = ca.DM(obs['center'])
        distance_to_obstacle_center = ca.norm_2(pos_k - obs_center_dm)
        min_safe_distance = drone_radius + obs['radius']
        opti.subject_to(distance_to_obstacle_center >= min_safe_distance)

solver_opts = {
    'ipopt.max_iter': 1000, 'ipopt.print_level': 0, 'print_time': 0,
    'ipopt.acceptable_tol': 1e-6, 'ipopt.tol': 1e-7,
    'ipopt.warm_start_init_point': 'yes'
}
opti.solver('ipopt', solver_opts)

# --------------------------------------------------------------------------
# --- 4. Reference Trajectory Generation -----------------------------------
# --------------------------------------------------------------------------
def get_circular_reference(t_start, N_horizon, dt_val, ang_speed, radius, z_alt):
    X_ref = np.zeros((n_states, N_horizon + 1))
    current_angle_offset = ang_speed * t_start
    for k_horizon in range(N_horizon + 1):
        time_k_horizon = k_horizon * dt_val
        angle = current_angle_offset + ang_speed * time_k_horizon
        X_ref[0, k_horizon] = radius * np.cos(angle)
        X_ref[1, k_horizon] = radius * np.sin(angle)
        X_ref[2, k_horizon] = z_alt
        X_ref[3, k_horizon] = -radius * ang_speed * np.sin(angle)
        X_ref[4, k_horizon] = radius * ang_speed * np.cos(angle)
        X_ref[5, k_horizon] = 0
    return X_ref

# --------------------------------------------------------------------------
# --- 5. Simulation Loop (Receding Horizon Control) ------------------------
# --------------------------------------------------------------------------
start_angle = np.pi / 4
start_x = circle_radius_ref * np.cos(start_angle)
start_y = circle_radius_ref * np.sin(start_angle)
start_vx = -circle_radius_ref * angular_speed_ref * np.sin(start_angle)
start_vy = circle_radius_ref * angular_speed_ref * np.cos(start_angle)

x_current = np.array([start_x,
                      start_y,
                      circle_center_z_ref,
                      start_vx,
                      start_vy,
                      0])
print(f"Initial x_current: {x_current}")
first_obstacle_center = obstacles_params[0]['center']
dist_to_first_obs_init = np.linalg.norm(x_current[0:3] - first_obstacle_center)
min_dist_to_obs_center = drone_radius + obstacles_params[0]['radius'] + 0.1
print(f"Calculated distance to center of first obstacle: {dist_to_first_obs_init:.3f} m (should be >= {min_dist_to_obs_center:.3f} m)")

sim_steps = int(T_sim / dt)
history_x_actual = np.zeros((n_states, sim_steps + 1))
history_u_applied = np.zeros((n_controls, sim_steps))
history_x_ref_current = np.zeros((n_states, sim_steps))
history_x_pred_at_each_step = []

history_x_actual[:, 0] = x_current

X_ref_horizon_init_guess = get_circular_reference(0, N, dt,
                                           angular_speed_ref, circle_radius_ref, circle_center_z_ref)
sol_X_dv_prev = X_ref_horizon_init_guess
sol_U_dv_prev = np.tile(np.array([0, 0, 9.81]), (N, 1)).T

print("Starting MPC Simulation...")
total_solver_time = 0
successful_solves = 0

for i in range(sim_steps):
    current_sim_time = i * dt
    solver_time = 0.0

    X_ref_horizon = get_circular_reference(current_sim_time, N, dt,
                                           angular_speed_ref, circle_radius_ref, circle_center_z_ref)
    history_x_ref_current[:, i] = X_ref_horizon[:, 0]

    opti.set_value(x0_param, x_current)
    opti.set_value(X_ref_param, X_ref_horizon)

    if i == 0:
        sol_X_dv_guess = X_ref_horizon_init_guess
        sol_U_dv_guess = np.tile(np.array([0, 0, 9.81]), (N, 1)).T
    else:
        sol_X_dv_guess = np.hstack((sol_X_dv_prev[:, 1:], sol_X_dv_prev[:, -1].reshape(-1,1)))
        sol_U_dv_guess = np.hstack((sol_U_dv_prev[:, 1:], sol_U_dv_prev[:, -1].reshape(-1,1)))

    opti.set_initial(X_dv, sol_X_dv_guess)
    opti.set_initial(U_dv, sol_U_dv_guess)

    t_start_solve = time.time()
    try:
        sol = opti.solve()
        solver_time = time.time() - t_start_solve
        total_solver_time += solver_time
        successful_solves += 1

        sol_X_dv_prev = sol.value(X_dv)
        sol_U_dv_prev = sol.value(U_dv)

        u_optimal_applied = sol_U_dv_prev[:, 0]
        history_x_pred_at_each_step.append(sol_X_dv_prev)

    except RuntimeError as e:
        print(f"WARNING: Solver failed at step {i}, time {current_sim_time:.2f}s: {e}")
        print(f"  Current state (x_current): {x_current.flatten()}")
        print(f"  Target reference (X_ref_horizon[:,0]): {X_ref_horizon[:,0].flatten()}")
        print(f"  Initial guess for X_dv[:,0:2] was:\n{sol_X_dv_guess[:,0:2]}")

        if i > 0 and successful_solves > 0:
            print("  Fallback: Applying previous control.")
            u_optimal_applied = history_u_applied[:, i-1]
        else:
            print("  Fallback: Applying safe hover control with velocity correction.")
            vel_correction = -0.5 * x_current[3:6]
            u_optimal_applied = np.array([vel_correction[0], vel_correction[1], 9.81])

        history_x_pred_at_each_step.append(sol_X_dv_guess)

    x_next_actual_full = f_discrete_rk4(x_current, u_optimal_applied, dt)
    if isinstance(x_next_actual_full, ca.DM):
        x_next_actual = x_next_actual_full.full().flatten()
    else:
        x_next_actual = x_next_actual_full.flatten()

    history_u_applied[:, i] = u_optimal_applied
    x_current = x_next_actual
    history_x_actual[:, i+1] = x_current

    avg_solve_time_ms = (total_solver_time / successful_solves * 1000) if successful_solves > 0 else -1.0
    current_solve_time_ms = solver_time * 1000
    if (i + 1) % (sim_steps // 20) == 0 or i == 0:
        print(f"Sim step {i+1}/{sim_steps}. Solver time: {current_solve_time_ms:.2f} ms (avg: {avg_solve_time_ms:.2f} ms). Pos: ({x_current[0]:.2f}, {x_current[1]:.2f}, {x_current[2]:.2f})")

print("Simulation Finished.")
if successful_solves > 0:
    print(f"Average solver time over successful solves: { (total_solver_time / successful_solves) * 1000:.3f} ms")
    print(f"Total successful solves: {successful_solves}/{sim_steps}")
else:
    print("No successful solves.")

# --------------------------------------------------------------------------
# --- 6. Plotting Results --------------------------------------------------
time_vec = np.arange(0, T_sim + dt, dt)
time_vec_ctrl = np.arange(0, T_sim, dt)

fig1 = plt.figure(figsize=(12, 10))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot(history_x_actual[0, :], history_x_actual[1, :], history_x_actual[2, :], 'b-', linewidth=2, label='Drone Actual Trajectory')
ax1.scatter(history_x_actual[0, 0], history_x_actual[1, 0], history_x_actual[2, 0], c='lime', marker='o', s=100, label='Start', depthshade=True)
ax1.scatter(history_x_actual[0, -1], history_x_actual[1, -1], history_x_actual[2, -1], c='red', marker='x', s=100, label='End', depthshade=True)
theta_ref_circle = np.linspace(0, 2 * np.pi, 200)
x_ref_circ = circle_radius_ref * np.cos(theta_ref_circle)
y_ref_circ = circle_radius_ref * np.sin(theta_ref_circle)
z_ref_circ = np.full_like(x_ref_circ, circle_center_z_ref)
ax1.plot(x_ref_circ, y_ref_circ, z_ref_circ, 'r--', alpha=0.7, label='Reference Path')
for obs in obstacles_params:
    center = obs['center']
    radius = obs['radius']
    u_sph = np.linspace(0, 2 * np.pi, 30)
    v_sph = np.linspace(0, np.pi, 30)
    x_sph = center[0] + radius * np.outer(np.cos(u_sph), np.sin(v_sph))
    y_sph = center[1] + radius * np.outer(np.sin(u_sph), np.sin(v_sph))
    z_sph = center[2] + radius * np.outer(np.ones(np.size(u_sph)), np.cos(v_sph))
    ax1.plot_surface(x_sph, y_sph, z_sph, color='dimgray', alpha=0.6, rcount=10, ccount=10)
pred_idx_plot = sim_steps // 2
if successful_solves > 0 and pred_idx_plot < len(history_x_pred_at_each_step) and history_x_pred_at_each_step[pred_idx_plot] is not None:
    x_pred_example = history_x_pred_at_each_step[pred_idx_plot]
    ax1.plot(x_pred_example[0, :], x_pred_example[1, :], x_pred_example[2, :], 'g:', linewidth=1.5, label=f'Predicted Traj. (t={pred_idx_plot*dt:.1f}s)')
ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
ax1.set_title('3D Drone Trajectory Tracking with MPC'); ax1.legend()
ax1.view_init(elev=30, azim=45)
ax1.set_xlim([-circle_radius_ref-1.5, circle_radius_ref+1.5]); ax1.set_ylim([-circle_radius_ref-1.5, circle_radius_ref+1.5]); ax1.set_zlim([0, circle_center_z_ref + 2.5])
ax1.grid(True)

fig2, axs2 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
pos_labels = ['Px (m)', 'Py (m)', 'Pz (m)']
full_ref_path_plot = get_circular_reference(0, sim_steps, dt, angular_speed_ref, circle_radius_ref, circle_center_z_ref)
for i_state in range(3):
    axs2[i_state].plot(time_vec, history_x_actual[i_state, :], 'b-', label='Actual State')
    axs2[i_state].plot(time_vec_ctrl, full_ref_path_plot[i_state, :-1], 'r--', label='Target Reference')
    plot_pred_indices = [sim_steps // 4, sim_steps // 2, 3 * sim_steps // 4]
    for k_sim_idx in plot_pred_indices:
        if successful_solves > 0 and k_sim_idx < len(history_x_pred_at_each_step) and history_x_pred_at_each_step[k_sim_idx] is not None:
            pred_horizon_data = history_x_pred_at_each_step[k_sim_idx]
            time_pred_axis = time_vec_ctrl[k_sim_idx] + np.arange(N + 1) * dt
            valid_indices = time_pred_axis <= T_sim
            axs2[i_state].plot(time_pred_axis[valid_indices], pred_horizon_data[i_state, valid_indices], linestyle=':', alpha=0.7, label=f'Predicted (t={time_vec_ctrl[k_sim_idx]:.1f}s)' if i_state==0 and k_sim_idx == plot_pred_indices[0] else None)
    axs2[i_state].set_ylabel(pos_labels[i_state]); axs2[i_state].grid(True)
    if i_state == 0: axs2[i_state].legend(loc='best', fontsize='small')
axs2[-1].set_xlabel('Time (s)'); fig2.suptitle('Position States: Actual vs. Target Reference vs. Predicted', fontsize=16); fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

fig3, axs3 = plt.subplots(n_controls, 1, figsize=(14, 8), sharex=True)
control_labels = ['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)']
for i_ctrl in range(n_controls):
    axs3[i_ctrl].plot(time_vec_ctrl, history_u_applied[i_ctrl, :], 'b-', label='Applied Control')
    axs3[i_ctrl].plot(time_vec_ctrl, np.ones_like(time_vec_ctrl) * u_min[i_ctrl], 'k--', alpha=0.7, label='Min Limit' if i_ctrl==0 else None)
    axs3[i_ctrl].plot(time_vec_ctrl, np.ones_like(time_vec_ctrl) * u_max[i_ctrl], 'k--', alpha=0.7, label='Max Limit' if i_ctrl==0 else None)
    if i_ctrl == 2: axs3[i_ctrl].plot(time_vec_ctrl, np.ones_like(time_vec_ctrl) * 9.81, 'g:', alpha=0.7, label='Gravity Comp.')
    axs3[i_ctrl].set_ylabel(control_labels[i_ctrl]); axs3[i_ctrl].grid(True)
    if i_ctrl == 0: axs3[i_ctrl].legend(loc='best', fontsize='small')
axs3[-1].set_xlabel('Time (s)'); fig3.suptitle('Control Actions and Constraints', fontsize=16); fig3.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()


