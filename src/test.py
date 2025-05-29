import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Obstacle parameters
obs = {
    'x_min': 1.0, 'x_max': 8.0,
    'y_min': 2.0, 'y_max': 6.0,
    'z_min': 0.0, 'z_max': 3.0
}
buffer_xy = 0.5  # Buffer in the XY plane
buffer_z = 0.2   # Buffer in the Z direction
beta = 10.0      # Smoothness parameter: higher beta means a sharper transition

# ---
## Smooth Cost Function
# This function calculates a smooth cost that is high inside the buffered box
# and approaches zero outside the buffered box.
# It uses a product of sigmoid-like smooth step functions for each dimension.
def smooth_box_cost(x, y, z):
    # Maximum cost when inside the box
    cost_multiplier = 2000.0

    # Smooth step for the lower bound of each dimension: transitions from 0 to 1
    # as the coordinate moves past the buffered minimum.
    s_x_lower = 1 / (1 + np.exp(-beta * (x - (obs['x_min'] - buffer_xy))))
    s_y_lower = 1 / (1 + np.exp(-beta * (y - (obs['y_min'] - buffer_xy))))
    s_z_lower = 1 / (1 + np.exp(-beta * (z - (obs['z_min'] - buffer_z))))

    # Smooth step for the upper bound of each dimension: transitions from 0 to 1
    # as the coordinate moves from above the buffered maximum towards it.
    s_x_upper = 1 / (1 + np.exp(-beta * ((obs['x_max'] + buffer_xy) - x)))
    s_y_upper = 1 / (1 + np.exp(-beta * ((obs['y_max'] + buffer_xy) - y)))
    s_z_upper = 1 / (1 + np.exp(-beta * ((obs['z_max'] + buffer_z) - z)))

    # The product of these terms will be close to 1 when 'x,y,z' is inside the buffered box,
    # and close to 0 when it's outside. Multiplying by cost_multiplier sets the peak cost.
    cost_value = cost_multiplier * s_x_lower * s_x_upper * \
                 s_y_lower * s_y_upper * s_z_lower * s_z_upper
    
    return - cost_value + 1
def smooth_box_cost_2(x, y, z, buffer_xy=buffer_xy, buffer_z=buffer_z, beta_val=beta):
    """
    Calculates a smooth cost for a 3D box obstacle.
    - Inside the buffered box: The cost is a fixed low (negative) value.
    - Outside the buffered box: The cost is higher at lower 'z' values
      and smoothly decreases as 'z' increases.

    Args:
        x, y, z (casadi.SX or casadi.MX): Current position coordinates.
        obs (dict): Dictionary defining the obstacle's min/max x, y, z.
        buffer_xy (float): Buffer distance for x and y dimensions.
        buffer_z (float): Buffer distance for z dimension.
        beta_val (float): Smoothness parameter for the sigmoid-like transitions.
                          Higher values lead to sharper transitions.

    Returns:
        casadi.SX or casadi.MX: The calculated smooth cost.
    """
    # Maximum base cost when deeply inside the buffered box.
    # This value is used to scale the 'inside' part of the cost.
    cost_multiplier = 2000.0

    # --- 1. Calculate the 'base_box_shape_cost' ---
    # This component is high (cost_multiplier) when inside the buffered box
    # and approaches 0 when outside. It uses sigmoid-like smooth step functions.

    # Smooth step for the lower bound of each dimension: transitions from 0 to 1
    # as the coordinate moves past the buffered minimum.
    s_x_lower = 1 / (1 + np.exp(-beta_val * (x - (obs['x_min'] - buffer_xy))))
    s_y_lower = 1 / (1 + np.exp(-beta_val * (y - (obs['y_min'] - buffer_xy))))
    s_z_lower = 1 / (1 + np.exp(-beta_val * (z - (obs['z_min'] - buffer_z))))

    # Smooth step for the upper bound of each dimension: transitions from 0 to 1
    # as the coordinate moves from above the buffered maximum towards it.
    s_x_upper = 1 / (1 + np.exp(-beta_val * ((obs['x_max'] + buffer_xy) - x)))
    s_y_upper = 1 / (1 + np.exp(-beta_val * ((obs['y_max'] + buffer_xy) - y)))
    s_z_upper = 1 / (1 + np.exp(-beta_val * ((obs['z_max'] + buffer_z) - z)))

    # The product of these terms will be close to 1 when 'x,y,z' is inside the buffered box,
    # and close to 0 when it's outside. Multiplying by cost_multiplier sets the peak cost.
    base_box_shape_cost = cost_multiplier * s_x_lower * s_x_upper * \
                          s_y_lower * s_y_upper * s_z_lower * s_z_upper

    # --- 2. Define the Z-dependent cost for regions *outside* the box ---
    # This component will be higher for low 'z' values and lower for high 'z' values.
    # Tunable parameters for the z-dependent cost's range and decay rate.
    z_max_outside_cost = 10.0  # Maximum cost at z=0 when outside the box
    z_decay_rate = -1        # Controls how quickly the cost decreases with z

    # Exponential decay function for z-dependent cost.
    # This ensures the cost is higher at lower altitudes and decreases as z increases.
    z_dependent_outside_cost = z_max_outside_cost * np.exp(-z_decay_rate * z)

    # --- 3. Combine the costs using a smooth blending function ---
    # 'indicator_inside_box' smoothly transitions from ~0 (when outside the box)
    # to ~1 (when deeply inside the box).
    indicator_inside_box = base_box_shape_cost / cost_multiplier

    # This is the fixed cost value desired when fully inside the box,
    # matching your current function's output for the inside region.
    desired_cost_inside = -cost_multiplier + 1 # Which is -2000 + 1 = -1999

    # The final cost is a smooth blend:
    # - When 'indicator_inside_box' is ~1 (inside), the 'desired_cost_inside' term dominates.
    # - When 'indicator_inside_box' is ~0 (outside), the 'z_dependent_outside_cost' term dominates.
    final_cost = (1 - indicator_inside_box) * z_dependent_outside_cost + \
                 indicator_inside_box * desired_cost_inside

    return final_cost + 1



# ---
## 3D Grid and Cost Calculation
# Create a 3D grid of points to evaluate the cost function.
# Increasing the number of points (e.g., 30 instead of 20) provides a smoother visualization.
x = np.linspace(-1, 7, 30)
y = np.linspace(0, 8, 30)
z = np.linspace(-1, 4, 30)
X, Y, Z = np.meshgrid(x, y, z)

# Compute the smooth cost across the entire grid
cost = smooth_box_cost(X, Y, Z)

# ---
## 3D Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the cost as a semi-transparent scatter plot.
# The color of each point represents its calculated cost.
scatter = ax.scatter(X, Y, Z, c=cost, cmap='viridis', alpha=0.5, s=20)

# Add a color bar to show the cost mapping
fig.colorbar(scatter, ax=ax, label='Smooth Cost (High inside, Zero outside)')

# Define the buffered box vertices for plotting (red lines)
x_min_buf, x_max_buf = obs['x_min'] - buffer_xy, obs['x_max'] + buffer_xy
y_min_buf, y_max_buf = obs['y_min'] - buffer_xy, obs['y_max'] + buffer_xy
z_min_buf, z_max_buf = obs['z_min'] - buffer_z, obs['z_max'] + buffer_z

# Draw the buffered box (where the cost transitions) as red lines
# Bottom face
ax.plot([x_min_buf, x_max_buf], [y_min_buf, y_min_buf], [z_min_buf, z_min_buf], 'r-', linewidth=2)
ax.plot([x_max_buf, x_max_buf], [y_min_buf, y_max_buf], [z_min_buf, z_min_buf], 'r-', linewidth=2)
ax.plot([x_max_buf, x_min_buf], [y_max_buf, y_max_buf], [z_min_buf, z_min_buf], 'r-', linewidth=2)
ax.plot([x_min_buf, x_min_buf], [y_max_buf, y_min_buf], [z_min_buf, z_min_buf], 'r-', linewidth=2)
# Top face
ax.plot([x_min_buf, x_max_buf], [y_min_buf, y_min_buf], [z_max_buf, z_max_buf], 'r-', linewidth=2)
ax.plot([x_max_buf, x_max_buf], [y_min_buf, y_max_buf], [z_max_buf, z_max_buf], 'r-', linewidth=2)
ax.plot([x_max_buf, x_min_buf], [y_max_buf, y_max_buf], [z_max_buf, z_max_buf], 'r-', linewidth=2)
ax.plot([x_min_buf, x_min_buf], [y_max_buf, y_min_buf], [z_max_buf, z_max_buf], 'r-', linewidth=2)
# Vertical edges
ax.plot([x_min_buf, x_min_buf], [y_min_buf, y_min_buf], [z_min_buf, z_max_buf], 'r-', linewidth=2)
ax.plot([x_max_buf, x_max_buf], [y_min_buf, y_min_buf], [z_min_buf, z_max_buf], 'r-', linewidth=2)
ax.plot([x_max_buf, x_max_buf], [y_max_buf, y_max_buf], [z_min_buf, z_max_buf], 'r-', linewidth=2)
ax.plot([x_min_buf, x_min_buf], [y_max_buf, y_max_buf], [z_min_buf, z_max_buf], 'r-', linewidth=2)

# Define the original obstacle box vertices for plotting (blue dashed lines)
x_min_orig, x_max_orig = obs['x_min'], obs['x_max']
y_min_orig, y_max_orig = obs['y_min'], obs['y_max']
z_min_orig, z_max_orig = obs['z_min'], obs['z_max']

# Draw the original obstacle box as blue dashed lines (for reference)
# Bottom face
ax.plot([x_min_orig, x_max_orig], [y_min_orig, y_min_orig], [z_min_orig, z_min_orig], 'b--', linewidth=1)
ax.plot([x_max_orig, x_max_orig], [y_min_orig, y_max_orig], [z_min_orig, z_min_orig], 'b--', linewidth=1)
ax.plot([x_max_orig, x_min_orig], [y_max_orig, y_max_orig], [z_min_orig, z_min_orig], 'b--', linewidth=1)
ax.plot([x_min_orig, x_min_orig], [y_max_orig, y_min_orig], [z_min_orig, z_min_orig], 'b--', linewidth=1)
# Top face
ax.plot([x_min_orig, x_max_orig], [y_min_orig, y_min_orig], [z_max_orig, z_max_orig], 'b--', linewidth=1)
ax.plot([x_max_orig, x_max_orig], [y_min_orig, y_max_orig], [z_max_orig, z_max_orig], 'b--', linewidth=1)
ax.plot([x_max_orig, x_min_orig], [y_max_orig, y_max_orig], [z_max_orig, z_max_orig], 'b--', linewidth=1)
ax.plot([x_min_orig, x_min_orig], [y_max_orig, y_min_orig], [z_max_orig, z_max_orig], 'b--', linewidth=1)
# Vertical edges
ax.plot([x_min_orig, x_min_orig], [y_min_orig, y_min_orig], [z_min_orig, z_max_orig], 'b--', linewidth=1)
ax.plot([x_max_orig, x_max_orig], [y_min_orig, y_min_orig], [z_min_orig, z_max_orig], 'b--', linewidth=1)
ax.plot([x_max_orig, x_max_orig], [y_max_orig, y_max_orig], [z_min_orig, z_max_orig], 'b--', linewidth=1)
ax.plot([x_min_orig, x_min_orig], [y_max_orig, y_max_orig], [z_min_orig, z_max_orig], 'b--', linewidth=1)

# Set plot labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Box Obstacle with Smooth Cost (High Inside, Low Outside)')

# Set axis limits to ensure everything is visible
ax.set_xlim(-1, 7)
ax.set_ylim(0, 8)
ax.set_zlim(-1, 4)

# Adjust the viewing angle for better perspective
ax.view_init(elev=30, azim=45)

plt.show()