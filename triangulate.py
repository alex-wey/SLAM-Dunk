import numpy as np

# baseline
b = 0.120006
# focal length
f = 487.109
# principle point
cx = 320.788
cy = 245.845

# Input: 'Points_a' is nx2 matrix of 2D coordinate of points on Image A
#        'Points_b' is nx2 matrix of 2D coordinate of points on Image B
# Output: 'F_matrix' is 3x3 fundamental matrix
def calculate_F_Matrix(Points_a,Points_b):

    F = np.zeros((3, 3))

    num_Points = Points_a.shape[0]
    A = np.zeros((num_Points, 9))

    c_u_a = np.mean(Points_a[:, 0])
    c_v_a = np.mean(Points_a[:, 1])
    c_u_b = np.mean(Points_b[:, 0])
    c_v_b = np.mean(Points_b[:, 1])

    offset_a = np.array([[1, 0, -c_u_a],
                         [0, 1, -c_v_a],
                         [0, 0, 1]])
    offset_b = np.array([[1, 0, -c_u_b],
                         [0, 1, -c_v_b],
                         [0, 0, 1]])

    std_a = np.std(Points_a - np.array([c_u_a, c_v_a]))
    std_b = np.std(Points_b - np.array([c_u_b, c_v_b]))
    s_a = np.reciprocal(std_a)
    s_b = np.reciprocal(std_b)

    scale_a = np.array([[s_a, 0, 0],
                        [0, s_a, 0],
                        [0, 0, 1]])
    scale_b = np.array([[s_b, 0, 0],
                        [0, s_b, 0],
                        [0, 0, 1]])

    T_a = scale_a @ offset_a
    T_b = scale_b @ offset_b

    Points_a_2 = np.zeros((num_Points, 2))
    Points_b_2 = np.zeros((num_Points, 2))

    for i in range(num_Points):
        u_a = Points_a[i, 0]
        v_a = Points_a[i, 1]
        row_a = np.array([[u_a, v_a, 1]])

        u_b = Points_b[i, 0]
        v_b = Points_b[i, 1]
        row_b = np.array([[u_b, v_b, 1]])

        norm_a = T_a @ row_a.T
        norm_b = T_b @ row_b.T

        Points_a_2[i] = norm_a[0:2, 0]
        Points_b_2[i] = norm_b[0:2, 0]

    for i in range(num_Points):
        u_a = Points_a_2[i, 0]
        v_a = Points_a_2[i, 1]
        u_b = Points_b_2[i, 0]
        v_b = Points_b_2[i, 1]
        row = np.array([[u_a*u_b, v_a*u_b, u_b, u_a*v_b, v_a*v_b, v_b, u_a, v_a, 1]])
        A[i] = row

    U, S, Vh = np.linalg.svd(A)
    F = Vh[-1,:]
    F = np.reshape(F, (3,3))

    U, S, Vh = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diagflat(S) @ Vh

    F = T_b.T @ F @ T_a

    return F

# Input: 'Points_a' is nx2 matrix of 2D coordinate of points on Image match A
#        'Points_b' is nx2 matrix of 2D coordinate of points on Image match B
# Output: nx3 matrix of the 3D world points
def triangulate(Points_a, Points_b):

    print('Triangulating coordinates...')

    num_Points = Points_a.shape[0]
    world_points = np.zeros((num_Points, 3))

    for i in range(num_Points):
        x1 = Points_a[i, 0]
        y1 = Points_a[i, 1]
        x2 = Points_b[i, 0]

        Z = (b * f)/(x1 - x2)
        X = x1 * Z/f
        Y = y1 * Z/f

        world_points[i] = np.array([X, Y, Z])

    return world_points