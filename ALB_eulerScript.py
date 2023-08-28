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

#Eigvec1 and eigvals1 are from the ALB-preopen-W63-Y64-R65-W156-R169-D202 inertia tensor
#Eigvec2 and eigvals2 are from the ALB-open-W63-Y64-R65-W156-R169-D202 inertia tensor
eigvecs1_COM = np.array([151.42145978350828,187.6970520456348,128.7855117124184])
eigvecs2_COM = np.array([151.3614184908336,187.97046325101618,128.59879056381456])

#Vector of COM displacement going from preopen to open
COM_displacement_vector = eigvecs1_COM - eigvecs2_COM
print('COM displacement vector =')
print(COM_displacement_vector)
print()

eigvecs1 = np.array([[-0.2451166,-0.96873495,0.03834647],[-0.9692553,0.24574151,0.01246077],[0.0214945,0.03411318,0.99918681]])
eigvals1 = np.array([27652.96039144,40796.29299862,56194.91413859])
eigvecs2 = np.array([[-0.20828468,-0.97649319,0.0554846 ],[-0.97806098,0.20816634,-0.00796821],[0.00376913,0.05592698,0.99842775]])
eigvals2 = np.array([28422.13541492,41336.95463796,57050.37067951])

#plotting
plot_eigenvectors(eigvecs1, eigvals1, eigvecs2, eigvals2, color1='#C4B200', color2='#40FFBF')
