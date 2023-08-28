import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rot_mat_z(phi):
    return np.array([[np.cos(phi), -np.sin(phi), 0],
                     [np.sin(phi), np.cos(phi), 0],
                     [0, 0, 1]])

def rot_mat_y(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def rot_mat_x(psi):
    return np.array([[1, 0, 0],
                     [0, np.cos(psi), -np.sin(psi)],
                     [0, np.sin(psi), np.cos(psi)]])

def euler_angles_from_rotation_matrix(R):
    #Rotates eigenvector1 to eigenvector2 about a fixed set of axes (extrinsic rotation) in the order of X(Psi), then Y(Theta), and then Z(phi)
    theta = -np.arcsin(R[2,0])
    cos_theta = np.cos(theta)
    psi = np.arctan2(R[2,1]/cos_theta, R[2,2]/cos_theta)
    phi = np.arctan2(R[1,0]/cos_theta, R[0,0]/cos_theta)
    return phi, theta, psi

def plot_eigenvectors(eigvecs1, eigvals1, eigvecs2, eigvals2, color1='b', color2='r'):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot eigenvectors for matrix 1
    for i in range(3):
        ax.quiver(0, 0, 0, 1/eigvals1[i]*eigvecs1[0, i], 1/eigvals1[i]*eigvecs1[1, i], 1/eigvals1[i]*eigvecs1[2, i], color=color1, linewidth=2, arrow_length_ratio=0.2)

    # Plot eigenvectors for matrix 2
    for i in range(3):
        ax.quiver(0, 0, 0, 1/eigvals2[i]*eigvecs2[0, i], 1/eigvals2[i]*eigvecs2[1, i], 1/eigvals2[i]*eigvecs2[2, i], color=color2, linewidth=2, arrow_length_ratio=0.2)

    # Calculate Euler angles
    # Euler angles are calculated in order:
    R1 = eigvecs1.T
    R2 = eigvecs2.T
    R = np.dot(R2, R1.T)
    phi, theta, psi = euler_angles_from_rotation_matrix(R)
    
    #Check Euler Angles
    RCheck = np.dot(rot_mat_y(theta),rot_mat_x(psi))
    RCheck = np.dot(rot_mat_z(phi),np.dot(rot_mat_y(theta),rot_mat_x(psi)))
    print('checking if generated rotation matrix generated from eigenvectors matches rotation matrix generated from calculated euler angles')
    print(R)
    print(RCheck)

    # Plot rotation angles
    ax.text2D(0.05, 0.95, "Phi: {:.2f}, Theta: {:.2f}, Psi: {:.2f}".format(phi*180/np.pi, theta*180/np.pi, psi*180/np.pi), transform=ax.transAxes)

    ax.set_xlim([-1/50000, 1/50000])
    ax.set_ylim([-1/50000, 1/50000])
    ax.set_zlim([-1/50000, 1/50000])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
#Printing euler angles


#Eigvec1 and eigvals1 are from the 5HT-preopen-W63-Y64-R65-W156-R169-D202 inertia tensor
#Eigvec2 and eigvals2 are from the 5HT-open-W63-Y64-R65-W156-R169-D202 inertia tensor
eigvecs1_COM = np.array([151.1299528180161,187.4744910994502,129.8946939638658])
eigvecs2_COM = np.array([151.10463273549857,187.4534979397093,129.83687232862243])

#Vector of COM displacement going from preopen to open
COM_displacement_vector = eigvecs1_COM - eigvecs2_COM
print('COM displacement vector =')
print(COM_displacement_vector)
print()

eigvecs1 = np.array([[-0.19847529,-0.97826783,0.05999683],[-0.97995553,0.19700046,-0.02963067],[-0.01716732,0.06467518,0.99775869]])
eigvals1 = np.array([26087.09098251,36833.9497312,51438.6112523])
eigvecs2 = np.array([[-0.18792152,-0.97996378,0.06600374],[-0.98154306,0.18494691,-0.04866077],[-0.0354786,0.07392992,0.99663215]])
eigvals2 = np.array([26603.61313685,37111.79264112,52130.70404578])

#plotting 
plot_eigenvectors(eigvecs1, eigvals1, eigvecs2, eigvals2, color1='#FF8000', color2='#1A9999')
