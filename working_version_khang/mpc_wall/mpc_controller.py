import casadi as ca
import numpy as np

## Smooth Cost Function
# This function calculates a smooth cost that is high inside the buffered box
# and approaches zero outside the buffered box.
# It uses a product of sigmoid-like smooth step functions for each dimension.
beta = 10.0      # Smoothness parameter: higher beta means a sharper transition

def smooth_box_cost_casadi(x, y, z, obs, buffer_xy=0.08, buffer_z=0.05):
    # Maximum cost when inside the box
    cost_multiplier = 2000.0

    # Smooth step for the lower bound of each dimension
    s_x_lower = 1 / (1 + ca.exp(-beta * (x - (obs['x_min'] - buffer_xy))))
    s_y_lower = 1 / (1 + ca.exp(-beta * (y - (obs['y_min'] - buffer_xy))))
    s_z_lower = 1 / (1 + ca.exp(-beta * (z - (obs['z_min'] - buffer_z))))

    # Smooth step for the upper bound of each dimension
    s_x_upper = 1 / (1 + ca.exp(-beta * ((obs['x_max'] + buffer_xy) - x)))
    s_y_upper = 1 / (1 + ca.exp(-beta * ((obs['y_max'] + buffer_xy) - y)))
    s_z_upper = 1 / (1 + ca.exp(-beta * ((obs['z_max'] + buffer_z) - z)))

    # Product for inside/outside box
    cost_value = cost_multiplier * s_x_lower * s_x_upper * \
                 s_y_lower * s_y_upper * s_z_lower * s_z_upper

    return - cost_value + 1


class QuadcopterMPC:
    def __init__(self, mpc_sampling_time=0.1, scenario=1): # Added scenario argument
        self.T = mpc_sampling_time
        self.N = 20
        
        self.m = 0.5
        self.g = 9.81
        self.Ix = 2.32e-3
        self.Iy = 2.32e-3
        self.Iz = 4.00e-3
        self.L_arm = 0.25
        self.c_yaw_coeff = 0.01
        
        obstacle_sets = {
            1: [ 
                {'type': 'cylinder', 'cx':3.0,'cy':3.0,'r':0.2, 'z_min': 0.0, 'z_max': 1.5},
                {'type': 'cylinder', 'cx':4.5,'cy':3.5,'r':0.2, 'z_min': 0.00, 'z_max': 2.5},
                {'type': 'cylinder', 'cx':5.0,'cy':5.0,'r':1.0, 'z_min': 0.0, 'z_max': 1.1},
            ],
            2: [ 
                {'type': 'cylinder', 'cx':3.0,'cy':2.0,'r':0.3, 'z_min': 0.0, 'z_max': 2.0},
                {'type': 'cylinder', 'cx':5.0,'cy':4.0,'r':0.4, 'z_min': 0.0, 'z_max': 1.8},
            ],
            3: [ 
                {'type': 'cylinder', 'cx':2.0,'cy':2.0,'r':0.25, 'z_min': 0.0, 'z_max': 1.0},
                {'type': 'cylinder', 'cx':4.0,'cy':4.0,'r':0.35, 'z_min': 0.5, 'z_max': 2.2},
                {'type': 'cylinder', 'cx':3.0,'cy':5.0,'r':0.2, 'z_min': 0.0, 'z_max': 1.7},
            ],
            4: [ 
                {'type': 'wall', 'x_min': 3.75, 'x_max': 4.25, 'y_min': 2.0, 'y_max': 5.0, 'z_min': 1.4, 'z_max': 4.0},
                {'type': 'wall', 'x_min': 2.0, 'x_max': 5.0, 'y_min': 3.75, 'y_max': 4.25, 'z_min': 1.4, 'z_max': 4.0},
            ]
        }
        
        self.obstacles = obstacle_sets.get(scenario, obstacle_sets[1]) 
        print(f"Running with obstacle scenario: {scenario}")

        self.buffer_xy = 0.08
        # MODIFIED: Reduced Z buffer to encourage going over
        self.buffer_z = 0.05 

        self.n_states = 12
        self.n_controls = 4

    def mpc_initialize(self):
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

        # MODIFIED: State cost weights Q to encourage Z movement
        # Order: x,y,z, phi,theta,psi, vx_w,vy_w,vz_w, p,q,r
        Q_diag_values = [
            50, 50, 10,  # x, y, z positions 
            40, 40, 40,  # phi, theta, psi angles
            10, 10, 20,   # vx, vy, vz_w (significantly reduced vz_w cost from 40 to 5)
            1,  1,  1    # p, q, r body angular rates 
        ]
        Q = ca.diag(Q_diag_values)
        
        R_diag_values = [0.3, 0.3, 0.3, 0.3] 
        R = ca.diag(R_diag_values)
        
        Q_terminal_diag_values = [1000,1000,1000, 2000,2000,2000, 20,20,20, 5,5,5]
        Qf = ca.diag(Q_terminal_diag_values)

        obj = 0
        g_eq_constraints = [X[:,0] - P[:self.n_states]]
        g_ineq_constraints = []

        # direct collocation method
        f_previous = self.f(X[:,0], U[:,0])  # Initialize with the first state and control
        for k in range(self.N-1):
            # dynamic constraints using collocation
            # f_previous = self.f(X[:,k], U[:,k])
            f_next = self.f(X[:,k+1], U[:,k+1])
            x_k_half = (X[:,k] + X[:,k+1]) / 2 + (f_previous - f_next) * self.T  / 8
            # u_k_half = (U[:,k]  + U[:,k+1]) / 2
            u_k_half = U[:,k]
            f_k_half = self.f(x_k_half, u_k_half)
            x_k_dot_half = - 3 * (X[:,k] - X[:,k+1]) / (2 * self.T) - (f_previous + f_next) / 4
            # constraints for the collocation method
            g_eq_constraints.append(f_k_half - x_k_dot_half)
            f_previous = f_next

        for k in range(self.N):
            st = X[:,k]; ct = U[:,k]
            e = st - P[self.n_states:(2*self.n_states)] 
            obj += ca.mtimes([e.T, Q, e]) + ca.mtimes([ct.T, R, ct])
            
            current_state_in_horizon = X[:, k+1]
            pred_x = current_state_in_horizon[0]
            pred_y = current_state_in_horizon[1]
            pred_z = current_state_in_horizon[2]



            for obs in self.obstacles:
                if obs['type'] == 'cylinder':
                    dx = pred_x - obs['cx'] 
                    dy = pred_y - obs['cy'] 
                    dz = pred_z - obs['z_max']

                    k_sigmoid = 100.0  # Steepness parameter for sigmoid (tune as needed)
                    sigmoid_z = 1.0 / (1.0 + ca.exp(k_sigmoid * dz))  # Sigmoid: ~1 if z < max_z, ~0 if z > max_z
                    g_ineq_constraints.append(dx**2 + dy**2 - sigmoid_z * (obs['r'] + self.buffer_xy)**2)

                elif obs['type'] == 'wall':
                    smooth_cost = smooth_box_cost_casadi(
                        pred_x, pred_y, pred_z, obs, 
                        buffer_xy=self.buffer_xy, buffer_z=self.buffer_z
                    )
                    g_ineq_constraints.append(smooth_cost)

        eT = X[:,self.N] - P[self.n_states:(2*self.n_states)]
        obj += ca.mtimes([eT.T, Qf, eT])

        opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))
        all_g_constraints_list = g_eq_constraints + g_ineq_constraints
        g_constraints_stacked = ca.vertcat(*all_g_constraints_list)
        
        nlp = {'x': opt_variables, 'f': obj, 'g': g_constraints_stacked, 'p': P}
        opts = {'ipopt.max_iter': 800, 
                'ipopt.print_level': 0, 
                'ipopt.acceptable_tol': 1e-8, 
                'ipopt.acceptable_obj_change_tol': 1e-7,
                'ipopt.warm_start_init_point': 'yes',
                'ipopt.constr_viol_tol': 1e-4, 
                'ipopt.max_cpu_time': 60.0 
               }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        lbg_list = [0.0] * self.n_states * (self.N) 
        ubg_list = [0.0] * self.n_states * (self.N)
        
        lbg_list.extend([0.0] * len(g_ineq_constraints)) 
        ubg_list.extend([ca.inf] * len(g_ineq_constraints))

        lb_controls_list = [0.0] * (self.N * self.n_controls)
        ub_controls_list = [15.0] * (self.N * self.n_controls)
        state_bounds_lb_template = [-15.0,-15.0,0.0, -ca.pi/4,-ca.pi/4,-ca.pi, -2.0,-2.0,-2.0, -3*ca.pi,-3*ca.pi,-3*ca.pi]
        state_bounds_ub_template = [15.0,15.0,6.0, ca.pi/4,ca.pi/4,ca.pi, 2.0,2.0,2.0, 3*ca.pi,3*ca.pi,3*ca.pi]
        lb_states_list = []; ub_states_list = []
        for _ in range(self.N + 1):
            lb_states_list.extend(state_bounds_lb_template)
            ub_states_list.extend(state_bounds_ub_template)
        lbx_list = lb_controls_list + lb_states_list
        ubx_list = ub_controls_list + ub_states_list
        
        self.args = {'lbg': ca.vertcat(*lbg_list), 'ubg': ca.vertcat(*ubg_list),
                     'lbx': ca.vertcat(*lbx_list), 'ubx': ca.vertcat(*ubx_list)}

    def run_once(self, x0_np, xt_np, initial_guess_w=None):
        x0_ca = ca.DM(x0_np) 
        xt_ca = ca.DM(xt_np)
        P_val = ca.vertcat(x0_ca, xt_ca)
        
        if initial_guess_w is None:
            init_guess_size = self.n_controls * self.N + self.n_states * (self.N + 1)
            initial_guess_w = ca.DM.zeros(init_guess_size, 1)
        
        sol = self.solver(x0=initial_guess_w, p=P_val, **self.args)
        w_opt = sol['x'] 
        
        u_seq = ca.reshape(w_opt[:self.n_controls*self.N], self.n_controls, self.N)
        x_seq = ca.reshape(w_opt[self.n_controls*self.N:], self.n_states, self.N+1)
        return u_seq, x_seq, w_opt