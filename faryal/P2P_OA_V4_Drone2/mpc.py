import casadi as ca
import numpy as np
from time import time
from define_variables import Variables_Defination
from extract_data import Data_Extraction
from plot_drone_arena import Drone_Arena_Plot
from plot_states import States_Plot

class P2P_OA_MPC(Variables_Defination):

    def __init__(self):
        super().drone_variables()
        super().obstacle_variables()
        super().gate_variables()

    def mpc_initialize(self):
        self.T = 0.1                # step horizon
        self.N = 12                  # prediciton horizon

        #---- Input Limits
        motor_force_min = 0     # N
        motor_force_max = ((self.m*self.g)/4)*3      # N

        #---- State Limits
        x_min = -5.2
        x_max = 5.2
        
        y_min = -3.2
        y_max = 3.2

        z_min = 0.25
        z_max = 3.0

        u_min = v_min = w_min = -15  # meters/s
        u_max = v_max = w_max =  15  # meters/s


        # roll and pitch
        phi_min = theta_min = -89 * (np.pi/180) # rad
        phi_max = theta_max =  89 * (np.pi/180) # rad

        # yaw 
        psi_min = -ca.inf
        psi_max = ca.inf


        # limit for angular velocities
        p_min = q_min = r_min =  -360 * 3 * (np.pi/180) # rad/s
        p_max = q_max = r_max =  360 * 3 * (np.pi/180) # rad/s

        #---- States
        #  position
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        z = ca.SX.sym('z')
        # linear velocities
        u = ca.SX.sym('u')
        v = ca.SX.sym('v')
        w = ca.SX.sym('w')
        # angles
        phi = ca.SX.sym('phi')
        theta = ca.SX.sym('theta')
        psi = ca.SX.sym('psi')
        # angular velocities
        p = ca.SX.sym('p')
        q = ca.SX.sym('q')
        r = ca.SX.sym('r')

        states = ca.vertcat(
                            x,
                            y,
                            z,
                            u,
                            v,
                            w,
                            phi,
                            theta,
                            psi,
                            p,
                            q,
                            r
                            )

        self.n_states = states.numel()

        #---- Control Inputs
        u1 = ca.SX.sym('u1')
        u2 = ca.SX.sym('u2')
        u3 = ca.SX.sym('u3')
        u4 = ca.SX.sym('u4')

        controls = ca.vertcat(
                            u1,
                            u2,
                            u3,
                            u4
                            )

        self.n_controls = controls.numel()

        # draw confifuration
        #  (Motor 1)        (Motor 2)
        #      1 ------------- 2
        #       \           /
        #        \         /
        #         \       /
        #          \     /
        #          (Center)
        #          /     \
        #         /       \
        #        /         \
        #       /           \
        #      4 ------------- 3
        #  (Motor 4)        (Motor 3)        
        # TODO draw it and write the dynamics equations
        # f1 - throttle on z direction
        f1 = u1+u2+u3+u4
        f2 = u3+u2-u1-u4
        f3 = u4+u3-u1-u2
        f4 = u1-u2+u3-u4

        # Kd = 1
        Kd = 0.008

        #---- Dynamics
        RHS = ca.vertcat(
                        # x_dot
                        u,
                        # y_dot
                        v,
                        # z_dot
                        w,
                        # u_dot
                        (1/self.m)*(ca.cos(phi) * ca.sin(theta) * ca.cos(psi) + ca.sin(phi) * ca.sin(psi)) * f1,
                        # v_dot
                        (1/self.m)*(ca.cos(phi) * ca.sin(theta) * ca.sin(psi) - ca.sin(phi) * ca.cos(psi)) * f1,
                        # w_dot
                        (1/self.m)*(ca.cos(phi) * ca.cos(theta)) * f1 - self.g,
                        # phi_dot
                        p,
                        # theta_dot
                        q,
                        # psi_dot
                        r,
                        # p_dot
                        (q * r * (self.Iy - self.Iz) / self.Ix) + (f2 * self.l1 / self.Ix),
                        # q_dot
                        (p * r * (self.Iz - self.Ix) / self.Iy) + (f3 * self.l2 / self.Iy),
                        # r_dot
                        (p * q * (self.Ix - self.Iy) / self.Iz) + f4*Kd/self.Iz
                        )

        self.f = ca.Function('f', [states, controls], [RHS])

        U = ca.SX.sym('U', self.n_controls, self.N)

        X = ca.SX.sym('X', self.n_states, (self.N+1))

        P = ca.SX.sym('P', 2 * self.n_states)  # [initial_state, goal_state]
        OBS = ca.SX.sym('OBS', 3)

        #---- State Weights
        Qx = 25
        Qy = 25
        Qz = 25
        Qu = 5
        Qv = 5
        Qw = 5
        Qphi = 25
        Qtheta = 25
        Qpsi = 25
        Qp = 5
        Qq = 5
        Qr = 5

        Q = ca.diagcat(Qx, Qy, Qz, Qu, Qv, Qw, Qphi, Qtheta, Qpsi, Qp, Qq, Qr)
        
        #---- Control Input Weights
        Ru1 = 0.1
        Ru2 = 0.1
        Ru3 = 0.1
        Ru4 = 0.1

        R = ca.diagcat(Ru1, Ru2, Ru3, Ru4)

        #---- Objective Function and Range Kutta Integration

        obj = 0 
        g = []  # equality constraints
        g.append(X[:, 0] - P[:self.n_states])
        k_avoidance = 10
        k_avoidance2 = 5
        epsilon = 0.01

        for i in range(self.N):
            st = X[:, i]
            con = U[:, i]
            obj = obj + (st-P[self.n_states:2*self.n_states]).T@Q@(st-P[self.n_states:2*self.n_states]) + con.T@R@con

            for obstacle in self.obstacles:
                obs_z = 0
                safe_distance = self.drone_radius + obstacle[2]
                horizontal_distance_to_axis = ca.sqrt((st[0] - obstacle[0])**2 + (st[1] - obstacle[1])**2)

                horizontal_penalty = k_avoidance / (horizontal_distance_to_axis - safe_distance + epsilon)
                obj += ca.fmax(0, horizontal_penalty)
                
                vertical_penalty = ca.if_else(
                    ca.logic_and(st[2] >= obs_z, st[2] <= obs_z + obstacle[3]),
                    k_avoidance2,
                    0
                )
                obj += vertical_penalty

            st_next = X[:, i+1]
            k1 = self.f(st, con)
            k2 = self.f(st + self.T/2*k1, con)
            k3 = self.f(st + self.T/2*k2, con)
            k4 = self.f(st + self.T/2*k3, con)
            st_next_rk4 = st + (self.T/6)*(k1 + (2*k2) + (2*k3) + k4)
            g.append(st_next - st_next_rk4)

        # Equality constraints: all should be zero
        lbg = [0.0] * ((self.N + 1) * self.n_states)
        ubg = [0.0] * ((self.N + 1) * self.n_states)

        #---- Casadi Non-linear Problem
        opt_variables = ca.vertcat(
            ca.reshape(U, -1, 1),
            ca.reshape(X, -1, 1)
                                )

        opt_params = ca.vertcat(
                                ca.reshape(P, -1, 1),
                                ca.reshape(OBS, -1, 1)
                                                    )
        
        nlp_prob = {
            'f': obj,
            'x': opt_variables,
            'g': ca.vcat(g),
            'p': opt_params
            }

        solver_opts = {
                    'ipopt': 
                            {
                                'max_iter': 200,
                                'print_level': 0,
                                'acceptable_tol': 1e-1,
                                'acceptable_obj_change_tol': 1e-1
                                },  
                    'print_time': 0
                    }
            
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, solver_opts)

        #---- Control and State Limits
        lbx = []
        ubx = []
        # Control input bounds for all N steps
        lbx.extend(np.kron(np.ones(self.N), [motor_force_min]*self.n_controls).tolist())
        ubx.extend(np.kron(np.ones(self.N), [motor_force_max]*self.n_controls).tolist())
        # Create state bounds vectors
        state_lbx = [x_min, y_min, z_min, u_min, v_min, w_min, phi_min, theta_min, psi_min, p_min, q_min, r_min]
        state_ubx = [x_max, y_max, z_max, u_max, v_max, w_max, phi_max, theta_max, psi_max, p_max, q_max, r_max]
        lbx.extend(np.kron(np.ones(self.N + 1), state_lbx).tolist())
        ubx.extend(np.kron(np.ones(self.N + 1), state_ubx).tolist())
            
        self.args = {
                'lbg': lbg,  
                'ubg': ubg,  
                'lbx': lbx,
                'ubx': ubx
                }
    
    def shift_timestep(self, step_horizon, t0, state_init, u, f):
        f_value = f(state_init, u[:, 0])
        next_state = ca.DM.full(state_init + (step_horizon * f_value))
        t0 = t0 + step_horizon
        u0 = ca.horzcat(
                        u[:, 1:],
                        ca.reshape(u[:, -1], -1, 1)
                        )
        return t0, next_state, u0

    def DM2Arr(self, dm):
        return np.array(dm.full())
    
    def run_mpc(self):
        t0 = 0
        # initial_point = np.array([-4, 2, 0.75]) 
        # final_point = np.array([4, -1, 0.75]) 

        initial_point = np.array([-3, 2.5, 0.75]) 
        final_point = np.array([3, -1, 0.75]) 

        state_init   = ca.DM([initial_point[0], initial_point[1], initial_point[2], 0, 0, 0, 0, 0, 0, 0, 0, 0])
        state_target = ca.DM([final_point[0], final_point[1], final_point[2], 0, 0, 0, 0, 0, np.deg2rad(50), 0, 0, 0])

        object_position = ca.DM([0, 0, 0.75])

        t = ca.DM(t0)

        u0 = ((self.m * self.g) / 4 ) * ca.DM.ones((self.n_controls, self.N)) 
        X0 = ca.repmat(state_init, 1, self.N+1)        

        cat_states = self.DM2Arr(X0)
        cat_controls = self.DM2Arr(u0[:, 0])
        times = np.array([[0]])

        mpc_iter = 0
        step_horizon = self.T
        sim_time = 15

        time_data = []
        time_data.append(t0)

        while (ca.norm_2(state_init - state_target) > 0.05) and (mpc_iter * step_horizon < sim_time):
            t1 = time()
            self.args['p'] = ca.vertcat(
                                state_init,    # current state
                                state_target,   # target state
                                object_position
                                )
            
            # optimization variable current state
            self.args['x0'] = ca.vertcat(
                                    ca.reshape(u0, self.n_controls*self.N, 1),
                                    ca.reshape(X0, self.n_states*(self.N+1), 1)
                                    )
            
            sol = self.solver(
                x0=self.args['x0'],
                lbx=self.args['lbx'],
                ubx=self.args['ubx'],
                lbg=self.args['lbg'],
                ubg=self.args['ubg'],
                p=self.args['p']
            )
            
            u = ca.reshape(sol['x'][:self.n_controls*(self.N)], self.n_controls, self.N)
            X0 = ca.reshape(sol['x'][self.n_controls*(self.N):], self.n_states, self.N+1)
            
            cat_states = np.dstack((
                                    cat_states,
                                    self.DM2Arr(X0)
                                    ))

            cat_controls = np.vstack((
                                    cat_controls,
                                    self.DM2Arr(u[:, 0])
                                    ))
            t = np.vstack((
                        t,
                        t0
                        ))

            t0, state_init, u0 = self.shift_timestep(step_horizon, t0, state_init, u, self.f)

            time_data.append(t0)

            X0 = ca.horzcat(X0[:, 1:], ca.reshape(X0[:, -1], -1, 1))

            t2 = time()
            times = np.vstack((
                            times,
                            t2-t1
                            ))

            mpc_iter = mpc_iter + 1
        
        print(f'avg calc time: {np.mean(times):.5f} seconds = {(np.mean(times))*(10**3):.2f} milliseconds')

        return time_data, cat_states, cat_controls, initial_point, final_point
    
if __name__ == "__main__":
    drone_mpc = P2P_OA_MPC()
    drone_mpc.mpc_initialize()
    time_data, cat_states, cat_controls, initial_point, final_point  = drone_mpc.run_mpc()
    extract = Data_Extraction()
    x, y, z, u, v, w, phi, theta, psi, p, q, r, u1, u2, u3, u4 = extract.data(cat_states, cat_controls)
    drone_arena_plot = Drone_Arena_Plot()
    drone_arena_plot.show_drone_arena(x, y, z, initial_point, final_point)
    states_plot = States_Plot()
    states_plot.plot_states_figures(time_data, x, y, z, u, v, w, phi, theta, psi, p, q, r, u1, u2, u3, u4)