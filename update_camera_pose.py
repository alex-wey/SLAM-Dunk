import numpy as np 
import triangulate
# Input: 3D points (n,3) of frame k and k+1
# Output: Transformation matrix Tk (Ck = C(k-1)*Tk)
#C (4,4) and T (4,4) both tranformation matrices
def update_camera_pose(coord1, coord2, C):

    print('Updating camera pose...')

    R = np.zeros((3,3))
    t = np.zeros((3,1))
    T = np.zeros((4,4))
    n = coord1.shape[0]
    p1 = np.sum(coord1, axis=0) / n
    p2 = np.sum(coord2, axis=0) / n
    q1 = coord1 - p1
    q2 = coord2 - p2
    
    H = np.dot(np.transpose(q1), q2)
    u, s, vh = np.linalg.svd(H)
    X = np.transpose(vh) @ np.transpose(u)
    detX = np.linalg.det(X)
    
    if(np.abs(detX - 1)  < 0.0001):
        R = X
    else:
        print("R calculation faliure")
        print(detX)
        return
    t = np.transpose(p2) - R @ np.transpose(p1)
    T[0:3, 0:3] = R
    T[0:3,3] = t
    T[3,3] = 1
    
    #update Camera Transformation with new T
    return T@C


# Input: 'matches_a' is nx2 matrix of 2D coordinate of points on Image A
#        'matches_b' is nx2 matrix of 2D coordinate of points on Image B
def ransac_F_Matrix(matches_a, matches_b):

    print('Implementing RANSAC...')

    p = 0.99
    e = 0.6
    s = 8
    N = int(np.log(1-p) / (np.log(1-np.power(1-e, s))))
    n = matches_a.shape[0]
    sigma = 0.02
    F_best = np.zeros((3,3))
    D = np.zeros(n)
    max_inliers = 0
    
    a_append = np.append(matches_a, np.ones((n, 1)), axis=1)
    b_append = np.append(matches_b, np.ones((n, 1)), axis=1)

    for i in range(N):
        samples = np.random.choice(n, s)
        sample_a = matches_a[samples]
        sample_b = matches_b[samples]
        F_est = triangulate.calculate_F_Matrix(sample_a, sample_b)
        D = np.absolute(np.sum(np.multiply((a_append @ F_est),  b_append), axis = 1))
        inliers = np.sum(D < sigma)
        if (inliers > max_inliers):
            F_best = F_est
            max_inliers = inliers

    D = np.absolute(np.sum(np.multiply((a_append @ F_best),  b_append), axis = 1))
    inliers_a = matches_a[D < sigma]
    inliers_b = matches_b[D < sigma]
    print("RANSAC Inliers:")
    print(max_inliers)
    return F_best, inliers_a, inliers_b

def main():
    #test main
    C = np.diag(np.ones(4))
    coord1 = np.array([[3,4,6],[1,6,8],[2,4,8]])
    coord2 = np.array([[3.3,4.1,6.1],[1.29,6,8.1],[2.31,4.1,8]])
    C = update_camera_pose(coord1, coord2, C)
    # thetax = np.arctan2(C[2,1], C[2,2])
    # thetay = np.arctan2(-C[2,0], np.sqrt(np.square(C[2,1])+ np.square(C[2,2])))
    # thetaz = np.arctan2(C[1,0], C[0,0])
    print(C)

if __name__ == "__main__":
    main()
