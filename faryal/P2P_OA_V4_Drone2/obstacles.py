import numpy as np
from define_variables import Variables_Defination

class Obstacles(Variables_Defination):

    def __init__(self):
        super().obstacle_variables()
    
    def obstacle_points_for_visualisation(self):
        self.obstacle_x_points = []
        self.obstacle_y_points = []
        self.obstacle_z_points = []
        angles = np.linspace(0,2*np.pi,30)
        for obstacle in self.obstacles:
            obstacle_base_x = obstacle[2]*np.cos(angles) + obstacle[0]
            obstacle_base_y = obstacle[2]*np.sin(angles) + obstacle[1]
            obstacle_vertical_z = np.linspace(0,obstacle[3], 100)
            self.obstacle_x_points.append(obstacle_base_x)
            self.obstacle_y_points.append(obstacle_base_y)
            self.obstacle_z_points.append(obstacle_vertical_z)