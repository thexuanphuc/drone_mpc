Drone Dynamics Explanation
Drone Configuration
A quadcopter typically has four motors arranged in a "plus" (+) configuration for this explanation. In this setup:

Motor 1: Positioned at the front (along the positive x-axis), spinning clockwise.
Motor 2: Positioned on the right (along the positive y-axis), spinning counterclockwise.
Motor 3: Positioned at the back (along the negative x-axis), spinning clockwise.
Motor 4: Positioned on the left (along the negative y-axis), spinning counterclockwise.

Each motor generates an upward thrust and a reaction torque. The clockwise and counterclockwise rotation pairing ensures stability by balancing yaw when thrusts are equal.
Motor Layout Diagram
Here’s a top-down view of the motor configuration with x and y axes:
         +y
          |
          | M2 (right, counterclockwise)
          |
+x -------(center)------- -x
          |
          | M4 (left, counterclockwise)
         -y
          |
          | M1 (front, clockwise)
          |
          | M3 (back, clockwise)


+x axis: Forward (towards Motor 1).
+y axis: Right (towards Motor 2).
Distances from the center to motors are $l_1$ (x-axis, Motors 1 and 3) and $l_2$ (y-axis, Motors 2 and 4).

Overview of the Dynamics
The drone’s motion is governed by ordinary differential equations (ODEs) for position, velocity, angle, and angular velocity.
Position Equations
$\dot{x} = u$ $ \dot{y} = v$ $ \dot{z} = w$$These define the drone’s position ($x, y, z$) in the inertial frame based on velocities ($u, v, w$).
Velocity Equations
$$\dot{u} = \frac{1}{m} ( \cos\phi \sin\theta \cos\psi + \sin\phi \sin\psi ) f_1$$$$\dot{v} = \frac{1}{m} ( \cos\phi \sin\theta \sin\psi - \sin\phi \cos\psi ) f_1$$$$\dot{w} = \frac{1}{m} ( \cos\phi \cos\theta ) f_1 - g$$These describe accelerations, where $f_1$ is total thrust, $\phi$, $\theta$, $\psi$ are roll, pitch, and yaw angles, $m$ is mass, and $g$ is gravity.
Angle Equations
$$\dot{\phi} = p,  \dot{\theta} = q,  \dot{\psi} = r$$These link angular velocities ($p, q, r$) in the body frame to the rates of change of Euler angles.
Angular Velocity Equations
The angular velocity equations govern rotational dynamics:$$\dot{p} = \frac{q r (I_y - I_z)}{I_x} + \frac{f_2 l_1}{I_x}$$$$\dot{q} = \frac{p r (I_z - I_x)}{I_y} + \frac{f_3 l_2}{I_y}$$$$\dot{r} = \frac{p q (I_x - I_y)}{I_z} + \frac{f_4 K_d}{I_z}$$
Variables

$p, q, r$: Angular velocities (roll, pitch, yaw).
$\dot{p}, \dot{q}, \dot{r}$: Angular accelerations.
$I_x, I_y, I_z$: Moments of inertia.
$f_2, f_3, f_4$: Control torques for roll, pitch, yaw.
$l_1, l_2$: Lever arms.
$K_d$: Yaw torque constant.

Breakdown

Gyroscopic Effects (first term):

Roll: $ \frac{q r (I_y - I_z)}{I_x} $ — Pitch ($q$) and yaw ($r$) affect roll if $I_y \neq I_z$.
Pitch: $ \frac{p r (I_z - I_x)}{I_y} $ — Roll ($p$) and yaw ($r$) influence pitch.
Yaw: $ \frac{p q (I_x - I_y)}{I_z} $ — Roll ($p$) and pitch ($q$) impact yaw.These terms reflect inertial coupling.


Control Torques (second term):

Roll: $ \frac{f_2 l_1}{I_x} $ — Thrust difference (e.g., Motor 4 > Motor 2) causes roll.
Pitch: $ \frac{f_3 l_2}{I_y} $ — Thrust difference (e.g., Motor 1 > Motor 3) causes pitch.
Yaw: $ \frac{f_4 K_d}{I_z} $ — Reaction torque differences drive yaw.



Physical Meaning

Roll: Increase Motor 4’s thrust over Motor 2 to roll right.
Pitch: Increase Motor 1’s thrust over Motor 3 to pitch down.
Yaw: Increase Motors 2 and 4’s thrusts for clockwise yaw.

These equations enable the drone to adjust its orientation by varying motor thrusts.
