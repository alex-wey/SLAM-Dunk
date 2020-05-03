
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation

# Input: numpy array of x,y,z coordinates of camera
# Output: 3D representation of a camera's path as gif
def gif(matrix):
    # instantiate writer
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])

    # set up graphs
    df = pd.DataFrame(matrix, columns=["x","y","z"])
    frn = len(df)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    sc = ax.scatter([],[],[], c='#d62728', alpha=0.5)
    
    # label axis
    ax.set_title('Camera Pose')
    ax.set_xlim3d(np.min(matrix[:,0])-10, np.max(matrix[:,0])+10)
    ax.set_xlabel('x')
    ax.set_ylim3d(np.min(matrix[:,1])-10, np.max(matrix[:,1])+10)
    ax.set_ylabel('z')
    ax.set_zlim3d(np.min(matrix[:,2])-10, np.max(matrix[:,2])+10)
    ax.set_zlabel('y')

    def animate(i):
        sc._offsets3d = (df.x.values[:i], df.y.values[:i], df.z.values[:i])

    # animate and save
    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=frn, interval=70)
    ani.save('SLAM.mp4', writer=writer)
