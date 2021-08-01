import numpy as np
import math
import matplotlib.pyplot as plt


def hermite_basis():
    plt.figure(num="Hermite basis functions")

    x = np.arange(0, 1, 0.001)

    # The four Hermite basis functions.

    H0 = 2 * x ** 3 - 3 * x ** 2 + 1
    H1 = -2 * x ** 3 + 3 * x ** 2
    H2 = x ** 3 - 2 * x ** 2 + x
    H3 = x ** 3 - x ** 2

    # Plot each function.

    plt.plot(x, H0)
    plt.plot(x, H1)
    plt.plot(x, H2)
    plt.plot(x, H3)
    plt.grid()
    plt.show()


def hermite_interpolation(h0, h1, h2, h3):
    plt.figure(num="Cubic Hermite interpolation")

    x = np.arange(h0[0], h1[0], 0.001)

    # Inverse transpose Hermite basis matrix.

    H = np.array([[2, -3, 0, 1],
                  [-2, 3, 0, 0],
                  [1, -2, 1, 0],
                  [1, -1, 0, 0]])

    X = np.array([x ** 3, x ** 2, x ** 1, x ** 0])

    h = np.array([h0[1], h1[1], h2[1], h3[1]])

    # Compute Hermite basis functions and then obtain the interpolated values.

    y = np.dot(h, np.dot(H, X))

    # Plot interpolation.

    plt.scatter(h0[0], h0[1])
    plt.scatter(h1[0], h1[1])
    plt.plot(x, y)
    plt.grid()
    plt.show()


def lagrange_interpolation(x, y):
    plt.figure(num="Lagrange interpolation")

    x_out = np.arange(min(x), max(x), 0.001)
    n = len(x_out)

    # Compute Lagrange basis polynomials.

    l = np.ones((len(x), n))
    for j in range(len(x)):
        xj = x[j]
        for m in range(len(x)):
            if m != j:
                l[j] = l[j] * (x_out - x[m]) / (xj - x[m])

    # Compute the interpolation polynomial.

    y_out = np.zeros(n)
    for i in range(len(y)):
        y_out = y_out + l[i] * y[i]

    # Plot interpolation.

    plt.scatter(x, y, s=70, c='b')  # Plot points.

    for i in range(len(x)):  # Plot Lagrange basis polynomials.
        plt.plot(x_out, l[i] * y[i])

    plt.plot(x_out, y_out, linestyle='--', linewidth=1.5, c='b')  # Plot Lagrange interpolation.

    plt.grid()
    plt.show()


def spline_interpolation(x, y):
    plt.figure(num="Spline interpolation")

    i = 0
    X = None

    # Compute three curves each four points.

    while (i + 3 < len(x)):

        # Each curve is a quadratic polynomial f(x) = ax^2 + bx + c.
        # To get each a, b, c of these three curves we solve a system of 9 variables.
        # We solve these linear equations using a matrix equation (Ax = b).

        A = np.array([[x[i] ** 2, x[i], 1, 0, 0, 0, 0, 0, 0],
                      [x[i + 1] ** 2, x[i + 1], 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, x[i + 1] ** 2, x[i + 1], 1, 0, 0, 0],
                      [0, 0, 0, x[i + 2] ** 2, x[i + 2], 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, x[i + 2] ** 2, x[i + 2], 1],
                      [0, 0, 0, 0, 0, 0, x[i + 3] ** 2, x[i + 3], 1],
                      [2 * x[i + 1], 1, 0, -2 * x[i + 1], -1, 0, 0, 0, 0],
                      [0, 0, 0, 2 * x[i + 2], 1, 0, -2 * x[i + 2], -1, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0],  # Free equation
                      ])

        B = np.array([[y[i]],
                      [y[i + 1]],
                      [y[i + 1]],
                      [y[i + 2]],
                      [y[i + 2]],
                      [y[i + 3]],
                      [0],
                      [0],
                      [0],
                      ])

        # Set free equation to get C1 continuity.

        if (i > 0):
            A[8] = [2 * x[i], 1, 0, 0, 0, 0, 0, 0, 0]
            B[8] = 2 * x[i] * X[6] + X[7]

        # And now we solve the system.

        X = np.linalg.inv(A).dot(B)

        # And plot each curve.

        x_out = np.arange(x[i], x[i + 1], 0.001)
        y_out = X[0] * x_out ** 2 + x_out * X[1] + X[2]
        plt.plot(x_out, y_out)
        i = i + 1

        x_out = np.arange(x[i], x[i + 1], 0.001)
        y_out = X[3] * x_out ** 2 + x_out * X[4] + X[5]
        plt.plot(x_out, y_out)
        i = i + 1

        x_out = np.arange(x[i], x[i + 1], 0.001)
        y_out = X[6] * x_out ** 2 + x_out * X[7] + X[8]
        plt.plot(x_out, y_out)
        i = i + 1

    # Plot each point.

    plt.scatter(x, y, s=70, c='b')

    plt.grid()
    plt.show()


def quadratic_bezier(x, y):
    plt.figure(num="Quadratic Bezier curve")

    t = np.linspace(0.0, 1.0, 100)

    plt.scatter(x, y, s=70, c='b')

    # Given three points, compute quadratic Bezier polynomial.

    y_out = ((1 - t) ** 2) * y[0] + 2 * (1 - t) * t * y[1] + (t ** 2) * y[2]
    x_out = ((1 - t) ** 2) * x[0] + 2 * (1 - t) * t * x[1] + (t ** 2) * x[2]

    plt.plot(x_out, y_out)
    plt.grid()
    plt.show()


def cubic_bezier(x, y):
    plt.figure(num="Cubic Bezier curve")

    t = np.linspace(0.0, 1.0, 100)

    plt.scatter(x, y, s=70, c='b')

    # Given four points, compute cubic Bezier polynomial.

    y_out = ((1 - t) ** 3) * y[0] + 3 * t * ((1 - t) ** 2) * y[1] + 3 * (t ** 2) * (1 - t) * y[2] + (t ** 3) * y[3]
    x_out = ((1 - t) ** 3) * x[0] + 3 * t * ((1 - t) ** 2) * x[1] + 3 * (t ** 2) * (1 - t) * x[2] + (t ** 3) * x[3]

    plt.plot(x_out, y_out)
    plt.grid()
    plt.show()


def bernstein(n, i, t):
    return (math.factorial(n) * (t ** i) * ((1 - t) ** (n - i))) / (math.factorial(i) * math.factorial(n - i))


def bezier_interpolation(x, y, show_info=True):
    t = np.linspace(0.0, 1.0, 100)

    x_out = np.zeros(100)
    y_out = np.zeros(100)

    # Compute the Bezier curve using the sum of Bernstein polynomials.

    n = len(x) - 1
    for i in range(n + 1):
        x_out = x_out + x[i] * bernstein(n, i, t)
        y_out = y_out + y[i] * bernstein(n, i, t)

    # Plot points.

    if show_info:
        plt.scatter(x, y, s=20)

    plt.plot(x_out, y_out)


def draw_4_bezier_curves_C1():
    plt.figure(num="Four Bezier curves with C1 continuity")

    x = np.zeros(13)
    y = np.zeros(13)
    radius = 5
    center = 0

    # Generate random points.

    for i in range(13):
        variation = np.random.rand()
        x[i] = center + variation * radius * math.cos(2 * math.pi * (i / 12))
        y[i] = center + variation * radius * math.sin(2 * math.pi * (i / 12))

    # Modify control points to get C1 continuity.

    x[12] = x[0]
    y[12] = y[12]

    x[4] = 2 * x[3] - x[2]
    y[4] = 2 * y[3] - y[2]

    x[7] = 2 * x[6] - x[5]
    y[7] = 2 * y[6] - y[5]

    x[10] = 2 * x[9] - x[8]
    y[10] = 2 * y[9] - y[8]

    x[11] = 2 * x[0] - x[1]
    y[11] = 2 * y[0] - y[1]

    bezier_interpolation(x[0:4], y[0:4])
    bezier_interpolation(x[3:7], y[3:7])
    bezier_interpolation(x[6:10], y[6:10])
    bezier_interpolation(x[9:13], y[9:13])

    """
    scale = 0.5
    x = x * scale
    y = y * scale

    bezier_interpolation(x[0:4], y[0:4])
    bezier_interpolation(x[3:7], y[3:7])
    bezier_interpolation(x[6:10], y[6:10])
    bezier_interpolation(x[9:13], y[9:13])
    """

    plt.grid()
    plt.show()


def draw_bezier_letter(save_image=False):
    plt.figure(num="Drawing a letter using Bezier interpolations")

    see_points = False
    P = np.zeros(shape=(35, 2))

    # Set the points which with a Bezier interpolation will represent a 'g' letter.

    # External circle.

    P[0] = (1.0, 1.0)
    P[1] = (-0.3, 1.5)
    P[2] = (-0.3, -0.5)
    P[3] = (1.0, 0.0)
    bezier_interpolation(P[0:4, 0], P[0:4, 1], show_info=see_points)

    # Upper leg.

    P[4] = (1.2, -0.6)
    P[5] = (0.5, -0.7)
    P[6] = (0.10, -0.45)
    bezier_interpolation(P[3:7, 0], P[3:7, 1], show_info=see_points)

    # Lower leg.

    P[7:9] = P[3:5]
    P[7:9, 0] = P[7:9, 0] + 0.1
    P[8] = (1.3, -0.7)
    P[9:11] = P[5:7]
    P[9:11, 1] = P[9:11, 1] - 0.1
    P[9, 0] = 0.6
    bezier_interpolation(P[7:11, 0], P[7:11, 1], show_info=see_points)

    # External axis.

    P[11:15] = P[7]
    P[14, 1] = 1.1
    bezier_interpolation(P[11:15, 0], P[11:15, 1], show_info=see_points)

    # Upper inner axis.

    P[15:19] = P[0]
    P[18, 1] = 1.1
    bezier_interpolation(P[15:19, 0], P[15:19, 1], show_info=see_points)

    # Inner circle.

    P[19:23] = P[0:4]
    P[19:23] = ((P[19:23] - 0.5) * 0.8) + 0.5
    P[19, 0] = 1.0
    P[22, 0] = 1.0
    bezier_interpolation(P[19:23, 0], P[19:23, 1], show_info=see_points)

    # Inner axis.

    P[23:25] = P[19]
    P[25:27] = P[22]
    bezier_interpolation(P[23:27, 0], P[23:27, 1], show_info=see_points)

    # Closing axis.

    P[27:29] = P[18]
    P[29:31] = P[14]
    bezier_interpolation(P[27:31, 0], P[27:31, 1], show_info=see_points)

    # Closing leg.

    P[31:33] = P[6]
    P[33:35] = P[10]
    bezier_interpolation(P[31:35, 0], P[31:35, 1], show_info=see_points)

    plt.axis('off')
    plt.axis("equal")

    if save_image:
        plt.savefig("letter_g.png", dpi=500)

    plt.show()

    # Modify the points to get an artistic result.

    plt.figure(num="Playing with the letter")

    for i in range(20):
        P = P * 1.05
        P[:, 0] = P[:, 0] + 0.07

        bezier_interpolation(P[ 0: 4, 0], P[ 0: 4, 1], show_info=see_points)
        bezier_interpolation(P[ 3: 7, 0], P[ 3: 7, 1], show_info=see_points)
        bezier_interpolation(P[ 7:11, 0], P[ 7:11, 1], show_info=see_points)
        bezier_interpolation(P[11:15, 0], P[11:15, 1], show_info=see_points)
        bezier_interpolation(P[15:19, 0], P[15:19, 1], show_info=see_points)
        bezier_interpolation(P[19:23, 0], P[19:23, 1], show_info=see_points)
        bezier_interpolation(P[23:27, 0], P[23:27, 1], show_info=see_points)
        bezier_interpolation(P[27:31, 0], P[27:31, 1], show_info=see_points)
        bezier_interpolation(P[31:35, 0], P[31:35, 1], show_info=see_points)

    plt.axis('off')
    plt.axis("equal")

    if save_image:
        plt.savefig("scaled_letters_g.png", dpi=500)

    plt.show()


def draw_bezier_surface():
    fig = plt.figure(num="Bezier surface")

    m = 4  # Rows.
    n = 4  # Columns.
    resol = 100

    # Create control points.

    C = np.ones(shape=(m, n, 3))
    for i in range(m):
        for j in range(n):
            C[i][j][0] = i * (1.0 / (m - 1))
            C[i][j][1] = j * (1.0 / (n - 1))
            C[i][j][2] = np.random.rand() * 0.8 - 0.4

    # Set corner points to zero.

    C[0, 0, 2], C[m - 1, 0, 2], C[m - 1, n - 1, 2], C[0, n - 1, 2] = np.zeros(4)

    # Compute Bezier surface.

    X = np.zeros(shape=(resol, resol))
    Y = np.zeros(shape=(resol, resol))
    Z = np.zeros(shape=(resol, resol))
    for u in range(resol):
        nu = u / (resol - 1)
        for v in range(resol):
            nv = v / (resol - 1)
            for i in range(m):
                bi = bernstein(m - 1, i, nu)
                for j in range(n):
                    bj = bernstein(n - 1, j, nv)
                    X[u, v] = X[u, v] + C[i][j][0] * bj * bi
                    Y[u, v] = Y[u, v] + C[i][j][1] * bj * bi
                    Z[u, v] = Z[u, v] + C[i][j][2] * bj * bi

    # Plot the surface.

    ax = fig.add_subplot(111, projection='3d')

    ax.plot_wireframe(X, Y, Z, rcount=20, ccount=20, color='fuchsia', linewidth=1)

    ax.scatter(C[:, :, 0], C[:, :, 1], C[:, :, 2], s=10, c='red')

    for i in range(m):
        ax.plot(C[i, :, 0], C[i, :, 1], C[i, :, 2], c='black', linewidth=1)

    for j in range(n):
        ax.plot(C[:, j, 0], C[:, j, 1], C[:, j, 2], c='black', linewidth=1)

    plt.show()


if __name__ == '__main__':

    # Different interpolation techniques using random values.

    hermite_basis()

    hermite_interpolation(np.array([0, 1]),
                          np.array([1, 1]),
                          np.array([0, 0]),
                          np.array([1, 1])
                          )

    lagrange_interpolation(np.array([-9, -4, -1,  7]),
                           np.array([ 5,  2, -2,  9]))

    spline_interpolation(np.array([3.0, 4.5, 7.0, 9.0, 10.5, 13.0, 15.0, 16.0, 18.0, 20.0]),
                         np.array([2.5, 1.0, 2.5, 0.5,  2.5,  1.0,  2.5, -2.0,  5.0,  6.0])
                         )

    quadratic_bezier(np.array([0.0, 1.0, 3.0]),
                     np.array([0.0, 2.0, 0.0])
                     )

    cubic_bezier(np.array([0.0, 0.0, 2.0, 3.0]),
                 np.array([0.0, 1.0, 1.0, 0.0])
                 )

    draw_4_bezier_curves_C1()

    draw_bezier_letter()

    draw_bezier_surface()
