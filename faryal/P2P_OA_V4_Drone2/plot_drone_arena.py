import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from obstacles import Obstacles
from gates import Gates
from drone import Drone

class Drone_Arena_Plot(Obstacles, Gates, Drone):

    def __init__(self):
        Obstacles.__init__(self)
        Gates.__init__(self)
        Drone.__init__(self)
        super().obstacle_points_for_visualisation()
        super().gate_points_for_visualisation()

        self.fig1 = plt.figure()
        self.ax_3d = self.fig1.add_subplot(111, projection='3d')
        self.fig1.set_size_inches(8, 7)
        self.ax_3d.grid()
        self.ax_3d.set_xlabel('x (m)', fontsize=10)
        self.ax_3d.set_ylabel('y (m)', fontsize=10)
        self.ax_3d.set_zlabel('z (m)', fontsize=10)
        self.ax_3d.set_xlim([-5-2, 5+2])
        self.ax_3d.set_ylim([-3-2, 3+2])
        self.ax_3d.set_zlim([0, 3])
        self.ax_3d.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax_3d, f'get_{a}lim')() for a in 'xyz')])
        self.ax_3d.set_title('Drone Arena', fontsize=10)

    def plot_arena(self):
        x = np.linspace(-5, 5, 13)
        y = np.linspace(-3, 3, 13)
        self.ax_3d.plot(x, 3*np.ones(len(x)), color='Black', linewidth=2)
        self.ax_3d.plot(x, -3*np.ones(len(x)), color='Black', linewidth=2)
        self.ax_3d.plot(5*np.ones(len(x)), y, color='Black', linewidth=2)
        self.ax_3d.plot(-5*np.ones(len(x)), y, color='Black', linewidth=2)

    def plot_obstacles(self):
        for x,y,z in zip(self.obstacle_x_points, self.obstacle_y_points, self.obstacle_z_points):
            for z_vertical in z:
                self.ax_3d.plot(x, y, z_vertical, color='Orange', linewidth=2)
    
    def plot_gates(self):
        for x,y,z in zip(self.gate_x_points, self.gate_y_points, self.gate_z_points):
            self.ax_3d.plot(x, y, z, color='Purple', linewidth=2)
        for x,y,z in zip(self.gate_inner_x_points, self.gate_inner_y_points, self.gate_inner_z_points):
            self.ax_3d.plot(x, y, z, color='Purple', linewidth=2)

    def plot_trajectory(self, x, y, z, initial_point, final_point):
        self.ax_3d.plot(x, y, z, color='Black', linewidth=2)
        self.ax_3d.scatter(initial_point[0], initial_point[1], initial_point[2])
        self.ax_3d.scatter(final_point[0], final_point[1], final_point[2])

    def plot_drone(self, x, y, z):
        super().drone_points_for_visualisation(x, y, z)
        for drone_x, drone_y, drone_z in zip(self.drone_x_points, self.drone_y_points, self.drone_z_points):
            self.ax_3d.plot(drone_x, drone_y, drone_z, color='Blue')

    def show_drone_arena(self, x, y, z, initial_point, final_point):
        self.plot_arena()
        self.plot_obstacles()
        # self.plot_gates()
        self.plot_trajectory(x, y, z, initial_point, final_point)
        self.plot_drone(x, y, z)
        plt.show(block=False)
