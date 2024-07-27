# Xiqiao Zhang
# 2024/7/21
# Xiqiao Zhang
# 2024/5/20
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pylab as p
from matplotlib.patches import Polygon
import casadi as ca
from casadi import *
from simulation_tracking import simulate
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms

loaded_times = np.load('times.npy')
loaded_states = np.load('path_cat_states.npy')
loaded_controls = np.load('path_cat_controls.npy')
# print("loaded_times shape:", loaded_times.shape)
# print("loaded_states shape:", loaded_states.shape)
# print("loaded_controls shape:", loaded_controls.shape)

x_data = loaded_states[0, 0, :].flatten()
y_data = loaded_states[1, 0, :].flatten()
theta_data = loaded_states[2, 0, :].flatten()

# 重新计算插值点，确保数量匹配
s_points = np.linspace(0, 1, len(x_data))

# 更新插值器
spline_x = ca.interpolant('spline_x', 'bspline', [s_points], x_data)
spline_y = ca.interpolant('spline_y', 'bspline', [s_points], y_data)
spline_theta = ca.interpolant('spline_theta', 'bspline', [s_points], theta_data)


adjusted_states = loaded_states[:, 0, :].T  # 转置是为了使其形状变为 (278, 3)
v_controls = loaded_controls[0::2]  # 取偶数索引
omega_controls = loaded_controls[1::2]  # 取奇数索引
adjusted_controls = np.hstack((v_controls, omega_controls))  # 横向堆叠形成 (278, 2)
trajectory = np.hstack((loaded_times, adjusted_states, adjusted_controls))

# print("Trajectory shape:", trajectory.shape)  # 预期输出: (278, 6)
# np.set_printoptions(threshold=np.inf)
#
# print("Trajectory inhalt:", trajectory)



T = 0.2
N = 11
P_T = 1

v_max = 1.5
v_min = -1.5
omega_max = pi / 4
omega_min = - omega_max

x_init = 0
y_init = 0
theta_init = 0
x_target = 6.5
y_target = 9
theta_target = 0


x = SX.sym('x')
y = SX.sym('y')
theta = SX.sym('theta')
states = vertcat(x, y, theta)  # Create the state vector
n_states = states.size1()  # Calculate the length of the state vector

v_des = SX.sym('v_des')
omega_des = SX.sym('omega_des')
controls = vertcat(v_des, omega_des)  # Create the control vector
n_controls = controls.size1()


v_actual = SX.sym('v_actual')
omega_actual = SX.sym('omega_actual')

rhs = vertcat(v_actual * cos(theta), v_actual * sin(theta), omega_actual)

f = Function('f', [states, controls, v_actual, omega_actual], [rhs])  # Define the dynamic equation
U = SX.sym('U', n_controls, N)  # Control variables (2, N)
X = SX.sym('X', n_states, N+1)  # State variables（3， N+1）
P = SX.sym('P', n_states + N*(n_states+n_controls))  # Initial state and reference state parameters

s = SX.sym('s')
v = SX.sym('v')
steps = vertcat(s, v)  # Create the state vector
n_steps = steps.size1()  # Calculate the length of the state vector

a = SX.sym('a')
change = vertcat(a)  # Create the state vector
n_change = change.size1()  # Calculate the length of the state vector

rhs2 = vertcat(1.1 * s + v, v + a)

f_virtual = ca.Function('f_virtual', [steps, change], [rhs2])
Z = SX.sym('Z', n_steps, N+1)      # include s and change step v
V = SX.sym('V', n_change, N)      # the change of v


v_current = 0
omega_current = 0
# 目标函数
obj = 0
# 约束向量
g = []

Q = DM.zeros(3,3)
Q[0,0] = 1; Q[1,1] = 5; Q[2,2] = 0.1

R = DM.zeros(2,2)
R[0,0] = 0.05; R[1,1] = 0.05
sta = X[:, 0]  # 初始状态
g.append(sta - P[:3])  # 添加初始状态约束

for k in range(N):
    st = X[:, k]
    con = U[:, k]
    steps = Z[:, k]
    change = V[:, k]
    s = steps[0]

    obj += (st - P[5 * (k+1) - 2:5 * (k+1) + 1]).T @ Q @ (st - P[5 * (k+1) - 2:5 * (k+1) + 1]) + \
           (con - P[5 * (k+1) + 1:5 * (k+1) + 3]).T @ R @ (con - P[5 * (k+1) + 1:5 * (k+1) + 3]) + \
           (s - 1)**2

    v_des_val = U[0, k]
    omega_des_val = U[1, k]

    # Update the actual output of the actuators
    v_current += T / P_T * (v_des_val - v_current)
    omega_current += T / P_T * (omega_des_val - omega_current)

    # Update the state with RK4
    k1 = f(st, vertcat(v_des_val, omega_des_val), v_current, omega_current)
    k2 = f(st + T / 2 * k1, vertcat(v_des_val, omega_des_val), v_current, omega_current)
    k3 = f(st + T / 2 * k2, vertcat(v_des_val, omega_des_val), v_current, omega_current)
    k4 = f(st + T * k3, vertcat(v_des_val, omega_des_val), v_current, omega_current)
    sta_new = X[:, k+1]
    sta_new_pre = st + T / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    g.append(sta_new - sta_new_pre)  # Equality constraints

    step_new = Z[:, k+1]
    step_pre = f_virtual(steps, change)

    g.append(step_new - step_pre)
g = ca.vertcat(*g)
print("Shape of g:", g.shape)

OPT_variables = ca.vertcat(
    X.reshape((-1, 1)),  # shape 3*(N+1)
    U.reshape((-1, 1)),  # shape 2N
    Z.reshape((-1, 1)),  # shape 2*(N+1)
    V.reshape((-1, 1))  # shape N
)
print("Shape of OPT:", OPT_variables.shape)
# 构建NLP问题
nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}

opts = {
    'ipopt': {
        'max_iter': 2000,
        'print_level': 0,  # Ipopt的输出等级，0为无输出
        'acceptable_tol': 1e-5,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0  # 是否打印求解器运行时间
}

# 创建求解器实例
solver = nlpsol('solver', 'ipopt', nlp_prob, opts)

n_params = n_states + N * (n_states + n_controls)
args = {
    'lbg': ca.DM.zeros((n_states*(N+1)+2*N, 1)),  # 状态 x 和 y 的下界
    'ubg': ca.DM.zeros((n_states*(N+1)+2*N, 1)),   # 状态 x 和 y 的上界
    'lbx': [None]*(5*(N+1) + 3*N),  # 初始化为 2*N 的列表
    'ubx': [None]*(5*(N+1) + 3*N),
    'p': np.zeros(n_params)
}

for i in range(3*(N+1)):
    if i % 3 == 0:
        args['lbx'][i] = -np.inf # x 的下界
        args['ubx'][i] = np.inf   # x 的上界
    elif i % 3 == 1:
        args['lbx'][i] = -np.inf  # y 的下界
        args['ubx'][i] = np.inf   # y 的上界
    elif i % 3 == 2:
        args['lbx'][i] = -np.inf  # theta 的下界
        args['ubx'][i] = np.inf   # theta 的上界
    print(f"Index {i}: lbx={args['lbx'][i]}, ubx={args['ubx'][i]}")

for i in range(3*(N+1), 3*(N+1) + 2*N):
    if i % 2 == 0:
        args['lbx'][i] = v_min
        args['ubx'][i] = v_max
    else:
        args['lbx'][i] = omega_min
        args['ubx'][i] = omega_max
    print(f"Index {i}: lbx={args['lbx'][i]}, ubx={args['ubx'][i]}")

for i in range(3*(N+1) + 2*N, 5*(N+1) + 2*N):  # critical value of steps
    if i % 2 == 0:  # s must be within (0, 1)
        args['lbx'][i] = 0
        args['ubx'][i] = 1
    else:
        args['lbx'][i] = 0
        args['ubx'][i] = 0.1

for i in range(5*(N+1) + 2*N, 5*(N+1) + 3*N):
    args['lbx'][i] = -0.1
    args['ubx'][i] = 0.1


def shift_timestep(step_horizon, t0, state_init, u, f, v_current, omega_current):
    v_des, omega_des = u[0, 0], u[1, 0]  # 假设 u 是 CasADi 类型的垂直堆叠控制向量

    # 更新执行器状态
    v_current += T / P_T * (v_des - v_current)
    omega_current += T / P_T * (omega_des - omega_current)

    # 构建控制向量和状态向量，确保是 CasADi 类型
    controls = ca.vertcat(v_current, omega_current)

    k1 = f(state_init, controls, v_current, omega_current)
    k2 = f(state_init + T / 2 * k1, controls, v_current, omega_current)
    k3 = f(state_init + T / 2 * k2, controls, v_current, omega_current)
    k4 = f(state_init + T * k3, controls, v_current, omega_current)
    f_value = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0


def DM2Arr(dm):
    return np.array(dm.full())

# ===============================================================
t0 = 0
sim_time = 90
v_current = 0
omega_current = 0
optimal_planned_paths = []
state_init = ca.DM([x_init, y_init, theta_init])        # initial state
state_target = ca.DM([x_target, y_target, theta_target])  # target state


t = ca.DM(t0)

u0 = ca.DM.zeros((n_controls, N))  # initial control
X0 = ca.repmat(state_init, 1, N+1)  # initial state full
Z0 = ca.DM.zeros((n_steps, N+1))  # initial control
V0 = ca.DM.zeros((n_change, N))


mpc_iter = 0
cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0[:, 0])
times = np.array([[0]])
print('shape of X0', X0.shape)

if __name__ == '__main__':
    main_loop_start = time()
    s = 0  # 初始 s 值

    while (ca.norm_2(state_init - state_target) > 1e-1) and (mpc_iter * T < sim_time):
        t1 = time()
        current_time = mpc_iter * T
        # 设置参数向量
        # 初始化机器人的姿态为初始条件 x0
        args['p'][0] = state_init[0]
        args['p'][1] = state_init[1]
        args['p'][2] = state_init[2]

        # 设置要跟踪的参考轨迹
        for k in range(1, N + 1):
            start_idx = np.searchsorted(trajectory[:, 0], current_time, side='right') - 1
            idx = start_idx + k - 1  # 从起始索引开始，计算后续索引
            print(idx)
            if idx >= len(trajectory):
                idx = len(trajectory) - 1  # 防止索引超出范围

            # 提取这一时间步的状态和控制量
            i_s = Z0[0, k]
            print("the value of s is:", i_s)

            # 使用插值函数计算目标位置
            x_ref = float(spline_x(i_s))
            y_ref = float(spline_y(i_s))
            theta_ref = float(spline_theta(i_s))

            v_ref, omega_ref = trajectory[idx, 4:6]

            print("X References:", x_ref)
            print("Y References:", y_ref)
            print("Theta References:", theta_ref)
            print("aktuell step", k)

            # 更新参数向量 P
            index_base = 5 * k
            args['p'][index_base - 2:index_base + 1] = [x_ref, y_ref, theta_ref]
            args['p'][index_base + 1:index_base + 3] = [v_ref, omega_ref]

        # 设置优化变量的初始值
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states * (N + 1), 1),
            ca.reshape(u0, n_controls * N, 1),
            ca.reshape(Z0, n_steps * (N + 1), 1),
            ca.reshape(V0, n_change * N, 1)
        )

        # 求解优化问题
        sol = solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'],
                     lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

        # 获取控制输入
        X0 = ca.reshape(sol['x'][: n_states * (N + 1)], n_states, N + 1)
        u = ca.reshape(sol['x'][n_states * (N + 1):n_states * (N + 1) + n_controls * N], n_controls, N)
        Z0 = ca.reshape(sol['x'][n_states * (N + 1) + n_controls * N:(n_states+n_steps) * (N + 1) + n_controls * N],
                        n_steps, N+1)
        V0 = ca.reshape(sol['x'][(n_states + n_steps) * (N + 1) + n_controls * N:], n_change, N)

        # print('增加步伐', Z0)
        optimal_planned_paths.append(DM2Arr(X0[:, 0]))

        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))
        # print('catstate shape', cat_states.shape)

        cat_controls = np.vstack((
            cat_controls,
            DM2Arr(u[:, 0])
        ))  # (202, 1)
        # print('cat shape', cat_controls)
        t = np.vstack((
            t,
            t0
        ))
        # print("更新前: x0 =", state_init)

        t0, state_init, u0 = shift_timestep(T, t0, state_init, u, f, v_current, omega_current)

        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )

        t2 = time()
        print(mpc_iter)
        print(t2 - t1)  # 打印出每轮迭代的时间
        times = np.vstack((
            times,
            t2 - t1
        ))

        mpc_iter = mpc_iter + 1


    main_loop_time = time()
    ss_error = ca.norm_2(state_init - state_target)
    # print("Shape of u:", u)
    # print("Shape of xx1:", xx1.shape)
    # print("Shape of xx:", xx.shape)

simulate(cat_states, cat_controls, t, T, N,
             np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]),adjusted_states,save=False)