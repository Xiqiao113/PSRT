# Xiqiao Zhang
# 2024/5/20
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import casadi as ca
import time
from casadi import *
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms

T = 0.2
N = 10
rob_length = 0.15
rob_diam = 0.1
P_T = 1

v_max = 0.6
v_min = -0.6
omega_max = pi / 4
omega_min = - omega_max

main_loop_start = time.time()

x = SX.sym('x')
y = SX.sym('y')
theta = SX.sym('theta')
states = vertcat(x, y, theta)  # Create the state vector
n_states = states.size1()  # Calculate the length of the state vector

v = SX.sym('v')
omega = SX.sym('omega')

controls = vertcat(v, omega)  # Create the control vector
n_controls = controls.size1()

rhs = vertcat(v * cos(theta), v * sin(theta), omega)

f = Function('f', [states, controls], [rhs])  # Define the dynamic equation
U = SX.sym('U', n_controls, N)  # Decision variables (2, N)
X = SX.sym('X', n_states, N+1)  # State variables（3， N+1）
P = SX.sym('P', n_states + n_states)  # Initial state and reference state parameters
X[:, 0] = P[:3]

for k in range(N):
    sta = X[:, k]
    con = U[:, k]
    k1 = f(sta, con)
    k2 = f(sta + T / 2 * k1, con)
    k3 = f(sta + T / 2 * k2, con)
    k4 = f(sta + T * k3, con)
    sta_new = sta + T / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    X[:, k+1] = sta_new

ff = Function('ff', [U, P], [X])

obj = 0

# State and control weighting matrices
Q = DM.zeros(3,3)
Q[0,0] = 1; Q[1,1] = 5; Q[2,2] = 0.1

R = DM.zeros(2,2)
R[0,0] = 0.5; R[1,1] = 0.05

# Calculate the objective function
for k in range(N):
    st = X[:,k]
    con = U[:,k]
    obj += dot(st - P[3:6], Q @ (st - P[3:6])) + dot(con, R @ con)  # Objective function

# Constraints vector
g = []

# calculate the constraints
for k in range(N+1):
    g.append(X[0,k])  # State x
    g.append(X[1,k])  # State y

g = vertcat(*g)

# Convert the decision variables into a one-dimensional vector
OPT_variables = reshape(U, 2*N, 1)

# Construct the NLP problem
nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}

opts = {
    'ipopt': {
        'max_iter': 100,
        'print_level': 0,  # Ipopt output level: 0, which means no output
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}

# Create an instance of the solver
solver = nlpsol('solver', 'ipopt', nlp_prob, opts)

# Define solver parameters and constraint boundaries
args = {
    'lbg': -3,  # Lower bounds for state variables x and y
    'ubg': 3,   # Upper bounds for state variables x and y
    'lbx': [None]*(2*N),  # Initialize as a list of size 2*N
    'ubx': [None]*(2*N)
}

# Set the boundaries for control inputs: even indices for v, odd indices for omega
for i in range(N):
    args['lbx'][2*i] = v_min       # Lower bound for v
    args['lbx'][2*i + 1] = omega_min  # Lower bound for omega
    args['ubx'][2*i] = v_max       # Upper bound for v
    args['ubx'][2*i + 1] = omega_max  # Upper bound for omega

# ===============================================================
t0 = 0
x0 = np.array([0, 0, 0.0])  # Initial state
xs = np.array([2.5, 2.5, 0])  # Reference position
sim_tim = 20  # Maximum simulation time
mpciter = 0  # MPC iteration counter
xx1 = []  # Array for storing all historical states
u_cl = []  # Array for storing all historical control values

# Initialize the state history array
xx = np.zeros((3, int(sim_tim / T) + 1))
xx[:, 0] = x0  # Set the initial state
t = [t0]  # Time array

u0 = np.full((N, 2), 0)  # Initialize control inputs to zero


def shift(T, t0, x0, u, f):
    """
    Shift the simulation state forward by one time step.

    Parameters:
    T (float): Sampling time step.
    t0 (float): Current time.
    x0 (numpy.ndarray): Current state vector.
    u (numpy.ndarray): Current control inputs.
    f (function): Function to compute the right-hand side of the system dynamics.

    Returns:
    tuple: Updated time, state, and control inputs.
    """
    st = np.copy(x0)
    con = u[0, :]  # Assuming u is a 2D array with controls in rows.
    f_value = f(st, con)
    st = st + (T * f_value)

    x0 = st
    t0 = t0 + T
    u0 = np.vstack([u[1:], u[-1, :]])  # Roll control inputs and repeat the last input

    return t0, x0, u0

while norm_2(x0 - xs) > 1e-2 and mpciter < sim_tim / T:
    p = vertcat(x0, xs)  # Set the parameter vector
    x0_reshaped = reshape(u0.T, 2 * N, 1)  # Reshape the initial optimization variables
    sol = solver(x0=x0_reshaped, lbx=args['lbx'], ubx=args['ubx'], lbg=args['lbg'], ubg=args['ubg'], p=p)

    u = np.array(reshape(sol['x'], 2, N).T)  # （N * 2）
    ff_value = ff(u.T, p)  # Compute the predicted trajectory
    xx1.append(ff_value.full().T)

    u_cl.append(u[0, :])
    t.append(t0)

    # Update the initial conditions
    t0, x0, u0 = shift(T, t0, x0, u, f)

    xx[:, mpciter + 1] = np.array(x0.full()).flatten()
    mpciter += 1


xx1 = np.array(xx1)  # (100 * 4 * 3)
print("Shape of xx1:", xx1)
print("Shape of xx:", xx.shape)
print("Shape of u:", u)


def Draw_MPC_point_stabilization_v1(t, xx, xx1, u_cl, xs, N):


    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 12})
    line_width = 1.5
    fontsize_labels = 14

    # Ensure that u_cl is a NumPy array
    if isinstance(u_cl, list):
        u_cl = np.array(u_cl)

    x_r_1 = []
    y_r_1 = []

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    fig.set_size_inches(8, 8)
    ax.set_aspect('equal', adjustable='box')

    L = rob_length  # Length
    W = rob_diam  # Width

    # Size parameters of the triangle
    triangle_size = W / 3
    triangle_height = triangle_size * np.sqrt(3) / 2

    for k in range(xx1.shape[0]):
        ax.cla()

        # Plot the reference position
        x1, y1, th1 = xs[0], xs[1], xs[2]
        ref_rect = Rectangle((x1 - L / 2, y1 - W / 2), L, W, color='green', fill=True)
        t1 = transforms.Affine2D().rotate_around(x1, y1, th1) + ax.transData
        ref_rect.set_transform(t1)
        ax.add_patch(ref_rect)

        # Plot the current position
        x1, y1, th1 = xx[0, k], xx[1, k], xx[2, k]
        x_r_1.append(x1)
        y_r_1.append(y1)
        rect = Rectangle((x1 - L / 2, y1 - W / 2), L, W, color='black', fill=True)
        t2 = transforms.Affine2D().rotate_around(x1, y1, th1) + ax.transData
        rect.set_transform(t2)
        ax.add_patch(rect)

        # Plot the direction triangle
        triangle = [
            (x1 + L / 2, y1),
            (x1 + L / 2 - triangle_height, y1 - triangle_size / 2),
            (x1 + L / 2 - triangle_height, y1 + triangle_size / 2)
        ]
        direction_triangle = Polygon(triangle, closed=True, color='red', fill=True)
        t3 = transforms.Affine2D().rotate_around(x1, y1, th1) + ax.transData
        direction_triangle.set_transform(t3)
        ax.add_patch(direction_triangle)

        # Plot the trajectory
        ax.plot(x_r_1, y_r_1, '-k', linewidth=line_width)

        # Plot the predicted trajectory
        if k < xx1.shape[0]:
            plt.plot(xx1[k, :N, 0], xx1[k, :N, 1], 'k--*')

        ax.set_xlabel('$x$-position (m)', fontsize=fontsize_labels)
        ax.set_ylabel('$y$-position (m)', fontsize=fontsize_labels)
        ax.grid(True)
        plt.pause(0.1)
        plt.draw()

    plt.close(fig)

    min_length = min(len(t), u_cl.shape[0])
    t = t[:min_length]
    u_cl = u_cl[:min_length]


    # Plot the control inputs
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.step(t, u_cl[:, 0], 'k', linewidth=line_width)  # 确保 u_cl 是二维的
    ax1.set_ylabel('v (rad/s)')
    ax1.grid(True)
    ax1.set_xlim([0, t[-1]])
    ax1.set_ylim([-0.35, 0.75])

    ax2.step(t, u_cl[:, 1], 'r', linewidth=line_width)
    ax2.set_xlabel('time (seconds)')
    ax2.set_ylabel('\omega (rad/s)')
    ax2.grid(True)
    ax2.set_xlim([0, t[-1]])
    ax2.set_ylim([-0.85, 0.85])

    plt.show()



main_loop_time = time.time() - main_loop_start
ss_error = norm_2(x0 - xs)
average_mpc_time = main_loop_time / (mpciter + 1)


Draw_MPC_point_stabilization_v1(t, xx, xx1, u_cl, xs, N)
