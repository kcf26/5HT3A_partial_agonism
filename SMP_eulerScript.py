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

#Eigvec1 and eigvals1 are from the SMP-preopen-W63-Y64-R65-W156-R169-D202 inertia tensor
#Eigvec2 and eigvals2 are from the SMP-open-W63-Y64-R65-W156-R169-D202 inertia tensor
eigvecs1_COM = np.array([123.71353380899932,159.97300242116708,102.88067572680998])
eigvecs2_COM = np.array([123.75763986267438,160.1223892161468,102.72668276486708])

#Vector of COM displacement going from preopen to open
COM_displacement_vector = eigvecs1_COM - eigvecs2_COM
print('COM displacement vector =')
print(COM_displacement_vector)
print()

eigvecs1 = np.array([[-0.23859437,-0.96994706,0.0477014 ],[-0.97110844,0.23853617,-0.00699234],[0.00459631,0.04799157,0.99883717]])
eigvals1 = np.array([27631.68544667,39866.03098558,54487.90367368])
eigvecs2 = np.array([[-0.15441256,-0.9847379,0.08029965],[-0.98799999,0.15360663,-0.01615611],[-0.00357498,0.08183076,0.99663983]])
eigvals2 = np.array([27103.43855405,41325.077091,56170.01706325])

#plotting
plot_eigenvectors(eigvecs1, eigvals1, eigvecs2, eigvals2, color1='#FFCC80', color2='#E2E5DE')
