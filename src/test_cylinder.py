import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Obstacle parameters (cylinder)
obs = {
    'type': 'cylinder',
    'cx': 1.0,       # Center x
    'cy': 3.0,       # Center y
    'r': 1.5,        # Radius
    'z_min': -0.5,    # Minimum z (for visualization, though cost depends on z_max)
    'z_max': 2.5     # Maximum z
}
buffer_xy = 0.5  # Buffer in the XY plane (added to radius)
buffer_z = 0.5   # Buffer for Z (how far above z_max the cost extends)
beta = 10.0      # Smoothness parameter: higher beta means a sharper transition

# ---
## Smooth Cost Function for Cylinder
def smooth_cylinder_cost(x, y, z):
    """
    Calculates a smooth cost for a 3D cylindrical obstacle.
    The cost is high inside the buffered cylinder (up to z_max) and
    smoothly decreases as distance from cylinder or z increases.
    """
    cost_multiplier = 2000.0  # Maximum cost when inside the buffered cylinder

    # Calculate radial distance from cylinder's center
    dx = x - obs['cx']
    dy = y - obs['cy']
    dist_xy_sq = dx**2 + dy**2
    # Using a smooth step for the radial distance.
    # It transitions from 0 to 1 as dist_xy_sq moves past (r + buffer_xy)^2
    # This acts like an "inside/outside" indicator for the buffered cylinder projection.
    # For a radial step, it's often more intuitive to use (r_buffered)^2 - dist_xy_sq
    # to make it positive when inside and negative when outside.
    # We want cost to be high when dist_xy_sq is *less* than (obs['r'] + buffer_xy)**2
    # So, the sigmoid input should be (obs['r'] + buffer_xy)**2 - dist_xy_sq
    
    # Smooth step for radial component: close to 1 inside buffered radius, close to 0 outside
    # using (R_buffered)^2 - (dx^2 + dy^2) as input
    s_radial = 1 / (1 + np.exp(-beta * ((obs['r'] + buffer_xy)**2 - dist_xy_sq)))

    # Smooth step for Z-component: close to 1 below z_max, close to 0 above z_max + buffer_z
    # The sigmoid in the original CasADi snippet was 1.0 / (1.0 + ca.exp(k_sigmoid * dz))
    # where dz = pred_z - obs['z_max'].
    # This means:
    # If pred_z < obs['z_max'], dz is negative, k_sigmoid * dz is negative, exp goes to 0, sigmoid to 1.
    # If pred_z > obs['z_max'], dz is positive, k_sigmoid * dz is positive, exp goes to infinity, sigmoid to 0.
    # So, this is a "cost when below z_max" indicator.
    
    # We want the cost to be high when z is below z_max.
    # So, for the sigmoid: beta * (z_max - z)
    # This will be large positive if z << z_max, large negative if z >> z_max.
    s_z_upper = 1 / (1 + np.exp(-beta * (obs['z_max'] - z)))
    
    # Optionally, if you also want a lower bound on Z (like the box):
    # s_z_lower = 1 / (1 + np.exp(-beta * (z - (obs['z_min'] - buffer_z))))
    # Then total_z_component = s_z_lower * s_z_upper

    # Combine radial and z components. The product is high inside the buffered cylinder
    # and low outside.
    cost_value = cost_multiplier * s_radial * s_z_upper
    
    # Adjust for desired cost output (e.g., negative high cost, positive low cost)
    return -cost_value + 1


# ---
## 3D Grid and Cost Calculation
# Create a 3D grid of points to evaluate the cost function.
x = np.linspace(0, 8, 30)
y = np.linspace(0, 8, 30)
z = np.linspace(-1, 4, 30)
X, Y, Z = np.meshgrid(x, y, z)

# Compute the smooth cost across the entire grid
cost = smooth_cylinder_cost(X, Y, Z)

# ---
## 3D Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the cost as a semi-transparent scatter plot.
scatter = ax.scatter(X, Y, Z, c=cost, cmap='viridis', alpha=0.5, s=20)

# Add a color bar to show the cost mapping
fig.colorbar(scatter, ax=ax, label='Smooth Cost (High inside cylinder, Low outside)')

# --- Plot the buffered cylinder (red lines) ---
# For a cylinder, we can plot circles at different heights and connect them.
theta = np.linspace(0, 2*np.pi, 100)
x_circ_buf = obs['cx'] + (obs['r'] + buffer_xy) * np.cos(theta)
y_circ_buf = obs['cy'] + (obs['r'] + buffer_xy) * np.sin(theta)

# Top buffered circle (at z_max, or slightly above if buffer_z applies to height)
# For the cost function, the "effective" top of the high-cost region is around z_max.
ax.plot(x_circ_buf, y_circ_buf, obs['z_max'], 'r-', linewidth=5)
# Bottom buffered circle (using obs['z_min'] for visualization)
ax.plot(x_circ_buf, y_circ_buf, obs['z_min'] - buffer_z, 'r-', linewidth=5) # Extend lower bound for visual

# Vertical lines for buffered cylinder
ax.plot([obs['cx'] + (obs['r'] + buffer_xy), obs['cx'] + (obs['r'] + buffer_xy)],
        [obs['cy'], obs['cy']],
        [obs['z_min'] - buffer_z, obs['z_max']], 'r-', linewidth=5)
ax.plot([obs['cx'] - (obs['r'] + buffer_xy), obs['cx'] - (obs['r'] + buffer_xy)],
        [obs['cy'], obs['cy']],
        [obs['z_min'] - buffer_z, obs['z_max']], 'r-', linewidth=5)
ax.plot([obs['cx'], obs['cx']],
        [obs['cy'] + (obs['r'] + buffer_xy), obs['cy'] + (obs['r'] + buffer_xy)],
        [obs['z_min'] - buffer_z, obs['z_max']], 'r-', linewidth=5)
ax.plot([obs['cx'], obs['cx']],
        [obs['cy'] - (obs['r'] + buffer_xy), obs['cy'] - (obs['r'] + buffer_xy)],
        [obs['z_min'] - buffer_z, obs['z_max']], 'r-', linewidth=5)


# --- Plot the original cylinder (blue dashed lines) ---
x_circ_orig = obs['cx'] + obs['r'] * np.cos(theta)
y_circ_orig = obs['cy'] + obs['r'] * np.sin(theta)

# Top original circle
ax.plot(x_circ_orig, y_circ_orig, obs['z_max'], 'b--', linewidth=1)
# Bottom original circle
ax.plot(x_circ_orig, y_circ_orig, obs['z_min'], 'b--', linewidth=1)

# Vertical lines for original cylinder
ax.plot([obs['cx'] + obs['r'], obs['cx'] + obs['r']],
        [obs['cy'], obs['cy']],
        [obs['z_min'], obs['z_max']], 'b--', linewidth=1)
ax.plot([obs['cx'] - obs['r'], obs['cx'] - obs['r']],
        [obs['cy'], obs['cy']],
        [obs['z_min'], obs['z_max']], 'b--', linewidth=1)
ax.plot([obs['cx'], obs['cx']],
        [obs['cy'] + obs['r'], obs['cy'] + obs['r']],
        [obs['z_min'], obs['z_max']], 'b--', linewidth=1)
ax.plot([obs['cx'], obs['cx']],
        [obs['cy'] - obs['r'], obs['cy'] - obs['r']],
        [obs['z_min'], obs['z_max']], 'b--', linewidth=1)


# Set plot labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Cylinder Obstacle with Smooth Cost')

# Set axis limits
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
ax.set_zlim(-1, 4)

# Adjust the viewing angle
ax.view_init(elev=30, azim=45)

plt.show()