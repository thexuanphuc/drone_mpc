import numpy as np

class Variables_Defination():

    def drone_variables(self):
        self.m = 0.80               # Mass of the quadrotor [kg]
        self.g = 9.81               # Gravity [m/s^2]
        self.Ix = 0.0056            # Moment of inertia about Bx axis [kg.m^2]
        self.Iy = 0.0071            # Moment of inertia about By axis [kg.m^2] 
        self.Iz = 0.0121            # Moment of inertia about Bz axis [kg.m^2]
        self.l1 = 0.14              # Quadrotor arm length [m]
        self.l2 = 0.12              # Quadrotor arm length [m]
        self.kd = 0.008             # Torque Coefficient
        self.drone_radius = 0.26     # Radius of drone

    def obstacle_variables(self):
        # [x, y, radius, height]  ---> cylinderical obstacle
        obstacle1 = [-1,1,0.1,3]
        obstacle2 = [2,0,0.1,3]
        self.obstacles = np.array([obstacle1, obstacle2])

    def gate_variables(self):
        # Enter gate bottom left (x,y,z) coordinate and angle. 
        # 0 degrees is horizontal gate aligned with y-axis
        # [x, y, z, angle]        ---> gate obstacle
        gate1 = [-2, 1.5/2, 0, 0]
        gate2 = [2, 1.5/2, 0, 90]
        self.gates = np.array([gate1, gate2])