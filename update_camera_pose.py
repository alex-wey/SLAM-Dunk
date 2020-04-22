import numpy as np 

#Input: 3D points (n,3) of frame k and k+1
#Output: Transformation matrix Tk (Ck = C(k-1)*Tk)
#C (4,4) and T (4,4) both tranformation matrices
def updateCameraPose(coord1, coord2, C):
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


def main():
    #test main
    C = np.diag(np.ones(4))
    coord1 = np.array([[3,4,6],[1,6,8],[2,4,8]])
    coord2 = np.array([[3.3,4.1,6.1],[1.29,6,8.1],[2.31,4.1,8]])
    C = updateCameraPose(coord1, coord2, C)
    # thetax = np.arctan2(C[2,1], C[2,2])
    # thetay = np.arctan2(-C[2,0], np.sqrt(np.square(C[2,1])+ np.square(C[2,2])))
    # thetaz = np.arctan2(C[1,0], C[0,0])
    print(C)

if __name__ == "__main__":
    main()
    
