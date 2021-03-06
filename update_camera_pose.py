import numpy as np 
import triangulate
# Given 3d point sets of the matches in frame k and k+1
# find the Translation matrix that relates these point sets
# and return the updated camera pose.
# Input: coord1, coord2 - 3D points (n,3) of frame k and k+1
#        C - the current camera pose
# Output: Updated Camera Matrix C (4,4)
#C (4,4) and T (4,4) both tranformation matrices
def update_camera_pose(coord1, coord2, C):
    print('Updating camera pose...')

    R = np.zeros((3,3))
    t = np.zeros((3,1))
    T = np.zeros((4,4))
    n = coord1.shape[0]
    #find centroid of points
    p1 = np.sum(coord1, axis=0) / n
    p2 = np.sum(coord2, axis=0) / n
    q1 = coord1 - p1
    q2 = coord2 - p2
    
    H = np.zeros((3,3))
    for i in range(n):
        H = H + np.outer(q1[i,:], q2[i,:])

    u, s, vh = np.linalg.svd(H)
    X = np.transpose(vh) @ np.transpose(u)
    detX = np.linalg.det(X)
    
    if(np.abs(detX - 1)  < 0.0001):
        #Rotation matrix calculated successfully
        R = X
    else:
        #Points were probably collinear and rotation matrix was not found
        #return same camera pose as previous frame.
        print("R calculation faliure")
        return C

    #Find the translation of the centroids
    t = np.transpose(p2) - R @ np.transpose(p1)

    # set up transformation matrix
    # T =  | R t |
    #      | 0 1 |
    T[0:3, 0:3] = R
    T[0:3,3] = t
    T[3,3] = 1
    
    #update Camera Transformation with T
    return C@T


# ransac_F_Matrix applies RANSAC between the two stereo pairs of matches 
# (matches_a, matches_b) and (matches_an, matches_bn) seperately. If an outlier
# match is found in one pair, its entry in the other pair is also removed
# Input: 'matches_a' - (n,2) matrix of matches in the left image of frame k
#        'matches_b' - (n,2) matrix of matches in the right image of frame k
#        'matches_a' - (n,2) matrix of matches in the left image of frame k+1
#        'matches_bn' - (n,2) matrix of matches in the right image of frame k+1
# Ouput: 
def ransac_F_Matrix(matches_a, matches_b, matches_an, matches_bn):

    print('Implementing RANSAC...')
    sigma = 0.04
    F_best = np.zeros((3,3))
    n = matches_a.shape[0]
    D = np.zeros(n)
    s = 8
    max_inliers = 0
    
    a_append = np.append(matches_a, np.ones((n, 1)), axis=1)
    b_append = np.append(matches_b, np.ones((n, 1)), axis=1)
    an_append = np.append(matches_an, np.ones((n, 1)), axis=1)
    bn_append = np.append(matches_bn, np.ones((n, 1)), axis=1)

    #We found that 50 iterations gave us the best results while also
    #factoring speed
    N = 50
    for i in range(N):
        samples = np.random.choice(n, s)
        sample_a = matches_a[samples]
        sample_b = matches_b[samples]
        sample_an = matches_an[samples]
        sample_bn = matches_bn[samples]
        try:
            #svd may not converge in F matrix calculation 
            F_est1 = triangulate.calculate_F_Matrix(sample_a, sample_b)
            F_est2 = triangulate.calculate_F_Matrix(sample_an, sample_bn)
        except Exception as e:
            print("error in calculating F_matrix:",e)
            continue

        #Claculate the distance and find inliers
        D1 = np.absolute(np.sum(np.multiply((a_append @ F_est1),  b_append), axis = 1))
        inliers1 = np.sum(D1 < sigma)
        D2 = np.absolute(np.sum(np.multiply((an_append @ F_est2),  bn_append), axis = 1))
        inliers2 = np.sum(D2 < sigma)

        #record F matrix if inliers were maximum
        if (inliers1 > max_inliers):
            F_best = F_est1
            max_inliers = inliers1
        if (inliers2 > max_inliers):
            F_best = F_est2
            max_inliers = inliers2

    #Retrieve best F matrix and find inliers
    D1 = np.absolute(np.sum(np.multiply((a_append @ F_best),  b_append), axis = 1))
    D2 = np.absolute(np.sum(np.multiply((an_append @ F_best),  bn_append), axis = 1))
    inliers_a1 = matches_a[np.logical_and(D1 < sigma,D2 < sigma)]
    inliers_b1 = matches_b[np.logical_and(D1 < sigma,D2 < sigma)]
    inliers_a2 = matches_an[np.logical_and(D1 < sigma,D2 < sigma)]
    inliers_b2 = matches_bn[np.logical_and(D1 < sigma,D2 < sigma)]
    print("RANSAC Inliers:", max_inliers)
    return F_best, inliers_a1, inliers_b1, inliers_a2, inliers_b2