import numpy as np
from define_variables import Variables_Defination

class Drone(Variables_Defination):

    def __init__(self):
        super().drone_variables()
    
    def drone_points_for_visualisation(self, x_traj, y_traj, z_traj):
        self.drone_x_points = []
        self.drone_y_points = []
        self.drone_z_points = []
        angles = np.linspace(0,2*np.pi,30)
        for x1_opt, x2_opt, x3_opt in zip(x_traj, y_traj, z_traj):
            drone_x = self.drone_radius*np.cos(angles) + x1_opt
            drone_y = self.drone_radius*np.sin(angles) + x2_opt
            drone_z = x3_opt
            self.drone_x_points.append(drone_x)
            self.drone_y_points.append(drone_y)
            self.drone_z_points.append(drone_z)