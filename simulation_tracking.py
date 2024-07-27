# Xiqiao Zhang
# 2024/7/10
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from matplotlib import animation
from time import time
from matplotlib.patches import Rectangle


def simulate(cat_states, cat_controls, t, step_horizon, N, reference, trajectory, save=False):
    # Known obstacle coordinates
    cx1, cx2, cy1, cy2 = 2, 3, 2, 3.5
    cx3, cx4, cy3, cy4 = 0, 1, 2, 3


    def create_robot(state=[0, 0, 0], L=0.3, W=0.25, triangle_size=0.13):
        x, y, th = state
        # Create the rectangle
        rect = np.array([
            [-L / 2, -W / 2],
            [-L / 2, W / 2],
            [L / 2, W / 2],
            [L / 2, -W / 2]
        ]).T
        # Create the direction triangle
        triangle = np.array([
            [L / 2, 0],
            [L / 2 - triangle_size * np.sqrt(3) / 2, -triangle_size / 2],
            [L / 2 - triangle_size * np.sqrt(3) / 2, triangle_size / 2]
        ]).T

        rotation_matrix = np.array([
            [cos(th), -sin(th)],
            [sin(th), cos(th)]
        ])

        # Apply rotation
        rect_coords = np.array([[x, y]]) + (rotation_matrix @ rect).T
        triangle_coords = np.array([[x, y]]) + (rotation_matrix @ triangle).T

        return rect_coords, triangle_coords

    trace_interval = 7
    rectangles = []

    def init():
        ax.add_patch(current_rect)
        ax.add_patch(direction_triangle)
        ax.add_patch(target_rect)
        path_line = ax.plot(trajectory[:, 0], trajectory[:, 1], 'b--', label='Trajectory')
        ax.legend(loc='upper right')

        ax.add_patch(obstacle_rect1)  # Add the obstacle to the plot
        ax.add_patch(obstacle_rect2)  # Add the obstacle to the plot

        return path, horizon, current_rect, direction_triangle, target_rect, path_line, obstacle_rect1, obstacle_rect2

    def animate(i):
        # get variables
        x = cat_states[0, 0, i]
        y = cat_states[1, 0, i]
        th = cat_states[2, 0, i]

        # update path
        if i == 0:
            path.set_data(np.array([]), np.array([]))
        x_new = np.hstack((path.get_xdata(), x))
        y_new = np.hstack((path.get_ydata(), y))
        path.set_data(x_new, y_new)

        # update horizon
        x_new = cat_states[0, :, i]
        y_new = cat_states[1, :, i]
        horizon.set_data(x_new, y_new)

        # update current_state
        rect_coords, triangle_coords = create_robot([x, y, th])
        current_rect.set_xy(rect_coords)
        direction_triangle.set_xy(triangle_coords)

        if i % trace_interval == 0:
            ghost_rect = plt.Polygon(rect_coords, closed=True, color='black', fill=False, linestyle='--')
            ax.add_patch(ghost_rect)
            rectangles.append(ghost_rect)

        if i == len(t) - 1:
            plot_control_inputs(t, cat_controls)

        return path, horizon, current_rect, direction_triangle, target_rect, rectangles


    def plot_control_inputs(t, cat_controls):
        plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 12})
        line_width = 1.5
        fontsize_labels = 14


        # Extract v and omega from cat_controls
        v = cat_controls[0::2]
        omega = cat_controls[1::2]
        # print('v 和 omega的形状', v.shape, omega.shape)


        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.step(t, v, 'k', linewidth=line_width)
        ax1.set_ylabel('v (rad/s)')
        ax1.grid(True)
        # ax1.set_xlim([0, t[-1]])
        # ax1.set_ylim([-0.35, 0.75])

        ax2.step(t, omega, 'r', linewidth=line_width)
        ax2.set_xlabel('time (seconds)')
        ax2.set_ylabel('omega (rad/s)')
        ax2.grid(True)
        # ax2.set_xlim([0, t[-1]])
        # ax2.set_ylim([-0.85, 0.85])

        plt.show()

    # create figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    min_scale = min(reference[0], reference[1], reference[3], reference[4]) - 2
    max_scale = max(reference[0], reference[1], reference[3], reference[4]) + 2
    ax.set_xlim(left=min_scale, right=max_scale)
    ax.set_ylim(bottom=min_scale, top=max_scale)

    # create lines:
    #   path
    path, = ax.plot([], [], 'k', linewidth=2)
    #   horizon
    horizon, = ax.plot([], [], 'x-g', alpha=0.5)
    #   current_state
    rect_coords, triangle_coords = create_robot(reference[:3])
    current_rect = plt.Polygon(rect_coords, closed=True, color='black')
    direction_triangle = plt.Polygon(triangle_coords, closed=True, color='red')
    target_rect_coords, _ = create_robot(reference[3:])
    target_rect = plt.Polygon(target_rect_coords, closed=True, color='green')
    ax.add_patch(target_rect)

    # Obstacle rectangles
    obstacle_rect1 = Rectangle((cx1, cy1), cx2 - cx1, cy2 - cy1, color='yellow', alpha=0.8)
    obstacle_rect2 = Rectangle((cx3, cy3), cx4 - cx3, cy4 - cy3, color='yellow', alpha=0.8)


    anim = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=len(t),
        interval=step_horizon * 70,
        blit=False,
        repeat=False
    )

    # plot_control_inputs(t, cat_controls)

    plt.show()


    if save == True:
        anim.save('./animation' + str(time()) +'.gif', writer='ffmpeg', fps=30)

    return