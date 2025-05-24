import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from obstacles import Obstacles
from drone import Drone

class States_Plot(Obstacles, Drone):

    def __init__(self):
        Obstacles.__init__(self)
        Drone.__init__(self)
        super().obstacle_points_for_visualisation()
    
    def plot_states_figures(self, tgrid, x1_opt, x2_opt, x3_opt, x4_opt, x5_opt, x6_opt, x7_opt, x8_opt, x9_opt, x10_opt, x11_opt, x12_opt, u1_opt, u2_opt, u3_opt, u4_opt):
        fig2, axs2 = plt.subplots(2, 3, figsize=(12, 6))
        font_size=10
        
        axs2[0, 0].plot(tgrid, x1_opt, label = 'x')
        axs2[0, 0].plot(tgrid, x2_opt, label = 'y')
        axs2[0, 0].plot(tgrid, x3_opt, label = 'z')
        axs2[0, 0].grid()
        axs2[0, 0].legend(fontsize=font_size)
        axs2[0, 0].set_xlabel('t / seconds', fontsize=font_size)
        axs2[0, 0].set_ylabel('Linear Translations (m)', fontsize=font_size)
        axs2[0, 0].set_title('Linear Translations (x, y, z)', fontsize=font_size)
        
        axs2[0, 1].plot(tgrid, x4_opt, label = 'u')
        axs2[0, 1].plot(tgrid, x5_opt, label = 'v')
        axs2[0, 1].plot(tgrid, x6_opt, label = 'w')
        axs2[0, 1].grid()
        axs2[0, 1].legend(fontsize=font_size)
        axs2[0, 1].set_xlabel('t / seconds', fontsize=font_size)
        axs2[0, 1].set_ylabel('Linear Velocities (m/s)', fontsize=font_size)
        axs2[0, 1].set_title('Linear Velocities (u, v, w)', fontsize=font_size)
        
        axs2[0, 2].plot(tgrid, [u1 for u1 in u1_opt], label = 'u1')
        axs2[0, 2].plot(tgrid, [u2 for u2 in u2_opt], label = 'u2')
        axs2[0, 2].plot(tgrid, [u3 for u3 in u3_opt], label = 'u3')
        axs2[0, 2].plot(tgrid, [u4 for u4 in u4_opt], label = 'u4')
        axs2[0, 2].grid()
        axs2[0, 2].legend(fontsize=font_size)
        axs2[0, 2].set_xlabel('t / seconds', fontsize=font_size)
        axs2[0, 2].set_ylabel('Forces (N)', fontsize=font_size)
        axs2[0, 2].set_title('Input Control Forces (u1, u2, u3, u4)', fontsize=font_size)
        
        axs2[1, 0].plot(tgrid, x7_opt*180/np.pi, label = 'phi')
        axs2[1, 0].plot(tgrid, x8_opt*180/np.pi, label = 'theta')
        axs2[1, 0].plot(tgrid, x9_opt*180/np.pi, label = 'psi')
        axs2[1, 0].grid()
        axs2[1, 0].legend(fontsize=font_size)
        axs2[1, 0].set_xlabel('t / seconds', fontsize=font_size)
        axs2[1, 0].set_ylabel('Angular Translations (deg)', fontsize=font_size)
        axs2[1, 0].set_title('Angular Translations (phi, theta, psi)', fontsize=font_size)
        
        axs2[1, 1].plot(tgrid, x10_opt*180/np.pi, label = 'p')
        axs2[1, 1].plot(tgrid, x11_opt*180/np.pi, label = 'q')
        axs2[1, 1].plot(tgrid, x12_opt*180/np.pi, label = 'r')
        axs2[1, 1].grid()
        axs2[1, 1].legend(fontsize=font_size)
        axs2[1, 1].set_xlabel('t / seconds', fontsize=font_size)
        axs2[1, 1].set_ylabel('Angular Velocities (deg/s)', fontsize=font_size)
        axs2[1, 1].set_title('Angular Velocities (p, q, r)', fontsize=font_size)
        
        
        super().drone_points_for_visualisation(x1_opt, x2_opt, x3_opt)
        for x,y in zip(self.obstacle_x_points, self.obstacle_y_points):
            axs2[1, 2].plot(x, y, color='Orange')
        for x,y in zip(self.drone_x_points, self.drone_y_points):
            axs2[1, 2].plot(x, y, color='Blue')
        axs2[1, 2].set_xlim(-5-2, 5+2)
        axs2[1, 2].set_ylim(-3-2, 3+2)
        axs2[1, 2].set_aspect('equal', adjustable='box')
        axs2[1, 2].grid()
        axs2[1, 2].set_xlabel('x / meters', fontsize=font_size)
        axs2[1, 2].set_ylabel('y / meters', fontsize=font_size)
        axs2[1, 2].set_title('2D Displacement', fontsize=font_size)
        
        plt.tight_layout()
        plt.show(block=False)

        input("Press Enter to close all plots...")
        plt.close('all')
