import casadi as ca
import numpy as np

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
            {'cx':3.0,'cy':3.0,'cz':2.0,'r':0.2,'min_z':0.0,'max_z':4.0},
            {'cx':3.5,'cy':4.5,'cz':2.0,'r':0.2,'min_z':0.0,'max_z':4.0},
            {'cx':4.5,'cy':4.0,'cz':2.0,'r':0.2,'min_z':0.0,'max_z':1.0},
            {'cx':5.5,'cy':4.0,'cz':2.0,'r':0.2,'min_z':0.0,'max_z':4.0},
            {'cx':4.5,'cy':5.5,'cz':2.0,'r':0.2,'min_z':0.0,'max_z':4.0},
        ]
        self.buffer = 0.05               # Safety buffer for obstacles
        self.n_states = 12              # Will be confirmed by states.numel()
        self.n_controls = 4             # Will be confirmed by controls.numel()
        init_guess_size = self.n_controls * self.N + self.n_states * (self.N + 1)
        self.init_w = ca.DM.zeros(init_guess_size, 1)

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

        # opts_f = {"jit": True, "compiler": "shell", "jit_options": {"flags": ["-O3"]}}
        self.f = ca.Function('f', [states, controls], [RHS]) #, opts_f)

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
        lbg_list = [0.0] * self.n_states
        ubg_list = [0.0] * self.n_states
        g_ineq_constraints = []
        f_previous = self.f(X[:,0], U[:,0])  # Initialize previous f

        for k in range(self.N-2):
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
            lbg_list.extend([0.0] * self.n_states)
            ubg_list.extend([0.0] * self.n_states)
            f_previous = f_next
        # # last step (not using collocation)
        # g_eq_constraints = [X[:,self.] - P[:self.n_states]]
        # lbg_list = [0.0] * self.n_states
        # ubg_list = [0.0] * self.n_states



        for k in range(self.N):
            st = X[:,k]; ct = U[:,k]
            e = st - P[self.n_states:(2*self.n_states)] 
            obj += ca.mtimes([e.T, Q, e]) + ca.mtimes([ct.T, R, ct])

            # for obstacle avoidance constraints
            current_state_in_horizon = X[:, k+1]
            for obs in self.obstacles:
                # Correct z-position index and compute sigmoid for smooth constraint switching
                dz = current_state_in_horizon[2] - obs['max_z']  # Use z-position (index 2) instead of x-position
                dx = current_state_in_horizon[0] - obs['cx'] 
                dy = current_state_in_horizon[1] - obs['cy'] 
                # Smoothly enforce x-y constraint only when z <= max_z using sigmoid
                # k_sigmoid = 200.0  # Steepness parameter for sigmoid (tune as needed)
                # sigmoid_z = 1.0 / (1.0 + ca.exp(k_sigmoid * dz))  # Sigmoid: ~1 if z < max_z, ~0 if z > max_z
                # g_ineq_constraints.append(dx**2 + dy**2 - sigmoid_z * (obs['r'] + self.buffer)**2)
                g_ineq_constraints.append(dx**2 + dy**2 - 1 * (obs['r'] + self.buffer)**2)
        
        eT = X[:,self.N] - P[self.n_states:(2*self.n_states)]
        obj += ca.mtimes([eT.T, Qf, eT])

        opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))
        all_g_constraints_list = g_eq_constraints + g_ineq_constraints
        g_constraints_stacked = ca.vertcat(*all_g_constraints_list)
        
        nlp = {'x': opt_variables, 'f': obj, 'g': g_constraints_stacked, 'p': P}
        opts = {
            'ipopt.max_iter': 800,
            'ipopt.print_level': 0,
            'ipopt.acceptable_tol': 1e-4,
            'ipopt.acceptable_obj_change_tol': 1e-5,
            # 'ipopt.linear_solver': 'ma27',
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # dynamics constraint bounds
        # lbg_list = [0.0] * self.n_states * len(g_eq_constraints)
        # ubg_list = [0.0] * self.n_states * len(g_eq_constraints)

        # inequality constraints bounds (obstacle avoidance)
        lbg_list.extend([0.0] * len(g_ineq_constraints))
        ubg_list.extend([ca.inf] * len(g_ineq_constraints))

        # input controls bounds
        lb_controls_list = [-1.0] * (self.N * self.n_controls)
        ub_controls_list = [15.0] * (self.N * self.n_controls)
        # state bounds
        state_bounds_lb_template = [-15.0,-15.0,0.0, -ca.pi/4,-ca.pi/4,-ca.pi, -2.0,-2.0,-2.0, -3*ca.pi,-3*ca.pi,-3*ca.pi]
        state_bounds_ub_template = [15.0,15.0,5.0, ca.pi/4,ca.pi/4,ca.pi, 2.0,2.0,2.0, 3*ca.pi,3*ca.pi,3*ca.pi]
        lb_states_list = []; ub_states_list = []
        for _ in range(self.N + 1):
            lb_states_list.extend(state_bounds_lb_template)
            ub_states_list.extend(state_bounds_ub_template)

        # the final bounds for decision variables (u, x)
        lbx_list = lb_controls_list + lb_states_list
        ubx_list = ub_controls_list + ub_states_list
        
        self.args = {'lbg': ca.vertcat(*lbg_list), 'ubg': ca.vertcat(*ubg_list),
                     'lbx': ca.vertcat(*lbx_list), 'ubx': ca.vertcat(*ubx_list)}

    def run_once(self, x0_np, xt_np):
        x0_ca = ca.DM(x0_np) # Ensure CasADi DM type
        xt_ca = ca.DM(xt_np)
        P_val = ca.vertcat(x0_ca, xt_ca)
        sol = self.solver(x0=self.init_w, p=P_val, **self.args)
        w_opt = sol['x']
        
        u_seq = ca.reshape(w_opt[:self.n_controls*self.N], self.n_controls, self.N)
        x_seq = ca.reshape(w_opt[self.n_controls*self.N:], self.n_states, self.N+1)

        # take the shifted previous solution as initial guess for next run
        self.init_w = ca.vertcat(
            ca.vertcat(w_opt[1:self.n_controls*self.N], w_opt[self.n_controls*self.N-1]), 
            ca.vertcat(w_opt[self.n_controls*self.N+1:], w_opt[-1])
        )
        return u_seq, x_seq
