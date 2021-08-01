import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.animation import FuncAnimation

trans, rots, goal = [], [], []
line, scatters, scatter_goal = [], [], []
fig, ax = plt.subplots()


def init():
    global trans, rots, goal, line, scatters, scatter_goal

    # Initialize skeleton.

    trans = np.array([[0, 0],  # Root node (shouldn't be changed).
                      [1, 0],
                      [1, 0],
                      [1, 0],
                      [1, 0],
                      ])

    rots = np.array([0.0,        # Root node.
                     0.0,
                     0.0,
                     0.0,
                     0.0,
                     ])

    goal = np.random.rand(2) * 6 - 3

    # Set canvas parameters.

    ax.axis("equal")
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)

    loc = plticker.MultipleLocator(base=1)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    ax.grid(which='both', axis='both', linestyle='-', zorder=0)
    ax.set_facecolor('#D2F8E7')

    # Set plotting objects.

    line, = ax.plot([], [], color="red", zorder=1)
    scatters = [None] * len(trans)
    for i in range(len(trans)):
        scatters[i] = ax.scatter([], [], zorder=2, s=100)

    scatter_goal = ax.scatter([], [], zorder=2, s=350, c="green", marker="*")


def update(frame):
    global trans, rots, goal, line, scatters, scatter_goal

    # Compute needed rotation to get to the goal.

    drots = get_jacobian(trans, rots, goal)
    rots = rots + drots
    pos = update_skeleton(trans, rots)

    # If the goal position is achieved, reset the goal position.

    if np.linalg.norm(pos[-1] - goal) < 0.2:
        goal = np.random.rand(2) * 6 - 3

    # Plot current state.

    line.set_data(pos[:, 0], pos[:, 1])
    for i in range(len(pos)):
        scatters[i].set_offsets(pos[i])
    scatter_goal.set_offsets(goal)


def update_skeleton(skeleton, joints):

    # Initialize homogeneous matrices.

    rot_matrix = np.array([np.identity(3)] * len(skeleton))
    trans_matrix = np.array([np.identity(3)] * len(skeleton))

    # Set the translation and rotation homogeneous matrices of each node.

    for i in range(len(skeleton)):
        rot_matrix[i][0][0] = math.cos(joints[i])
        rot_matrix[i][0][1] = -math.sin(joints[i])
        rot_matrix[i][1][0] = math.sin(joints[i])
        rot_matrix[i][1][1] = math.cos(joints[i])

        trans_matrix[i][0][2] = skeleton[i][0]
        trans_matrix[i][1][2] = skeleton[i][1]

    # Initialize the matrix that will have the position and the final vectors that will have the same position.

    pos_matrix = np.array([np.identity(3)] * len(skeleton))
    pos = np.ones(shape=(len(skeleton), 2))

    # Compute nodes positions as an accumulation of the transformations.

    for i in range(len(skeleton)):
        for j in range(i + 1):
            pos_matrix[i] = pos_matrix[i].dot(rot_matrix[j].dot(trans_matrix[j]))
        pos[i] = np.array([pos_matrix[i][0][2], pos_matrix[i][1][2]])

    return pos


def get_jacobian(skeleton, joints, final_pos):
    dt = 0.0001
    alpha = 0.1

    # Get the current position and the desired displacement.

    ini_pos = update_skeleton(skeleton, joints)
    direction = final_pos - ini_pos[-1]

    # Compute the Jacobian matrix.

    J = np.zeros(shape=(2, len(joints)))
    for i in range(len(joints)):
        djoints = joints.copy()
        djoints[i] = djoints[i] + dt
        dpos = update_skeleton(skeleton, djoints)
        tan = (dpos[-1] - ini_pos[-1]) / dt
        J[0][i] = tan[0]
        J[1][i] = tan[1]

    # Solve the system to obtain the angles displacement.

    J_inv = np.linalg.pinv(J)
    nrot = J_inv.dot(direction)

    # Normalize the result to obtain a better control.

    nrot = nrot / np.linalg.norm(nrot)

    return nrot * alpha


if __name__ == '__main__':
    init()
    ani = FuncAnimation(fig, update, interval=50)
    plt.show()
