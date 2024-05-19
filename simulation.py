# Xiqiao Zhang
# 2024/5/15
import numpy as np
import matplotlib.pyplot as plt

# initialize parameters
dt = 0.1  # timestep
time = np.arange(0, 20, dt)
x, y, theta = 0.0, 0.0, 0  # initial position and orientation
v_input, omega_input = 1.0, 0 # desire position and orientation
v, omega = 0.0, 0.0  # initial values of actual speed and angular velocity
T_v, T_omega = 1.0, 1.0  # PT1 Parameter

# Simulation
x_list, y_list = [x], [y]
for t in time:
    # update speed and angular velocity (PT1)
    v += (v_input - v) / T_v * dt
    omega += (omega_input - omega) / T_omega * dt

    # update vehicle status
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += omega * dt

    # record the trajectory
    x_list.append(x)
    y_list.append(y)

# plot the trajectory
plt.figure(figsize=(8, 8))
plt.plot(x_list, y_list, label='Trajectory')
plt.scatter(x_list[0], y_list[0], color='red', label='Start')
plt.scatter(x_list[-1], y_list[-1], color='green', label='End')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Vehicle Trajectory with PT1 Actuator Model')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()