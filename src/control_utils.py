
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os, sys, argparse
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

cam_pos_world_position = np.array([0.03343094, -0.01157302, 0.15718198])
cam_pos_world_quaternion = np.array([0.18658106, 0.77200158, -0.604802, -0.05844316])

cam_pos_psm1_position = np.array([0.08880545, -0.04930693, -0.03779696])
cam_pos_psm1_quaternion = np.array([0.20070423, 0.2857976, -0.9321147, 0.0959152])
cam_pos_psm2_position = np.array([-0.07828412, 0.01814246, 0.07171638])
cam_pos_psm2_quaternion = np.array([0.11987684, -0.2255778, -0.95644951, 0.14123927])


def disparity_to_pointcloud(disparity, Q_matrix, max_points=4096, scale_factor=0.001):

    """
    disparity.shape: (H, W)
    pointcloud.shape: (N, 3)
    """
    pointcloud = cv2.reprojectImageTo3D(disparity, Q_matrix, handleMissingValues=True)

    valid_mask = pointcloud[:, :, 2] > 0 
    valid_points = pointcloud[valid_mask]

    if len(valid_points) > max_points:
        indices = np.random.choice(len(valid_points), max_points, replace=False)
        downsampled_pointcloud = valid_points[indices]
    else:
        downsampled_pointcloud = valid_points

    pointcloud = downsampled_pointcloud * scale_factor
    
    return pointcloud



def plot_pointcloud(pointcloud):
    """
    poitncloud numpy shape: (N, 3)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2], 
                c=pointcloud[:, 2], cmap='viridis', s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud (Interactive)')

    plt.show()


# def main():
#     disparity_path =(os.path.join(pwd, "../../data/0819/disparities/disparities_0001.npz"))
#     if not os.path.exists(disparity_path):
#         print(f"Disparity file not found at {disparity_path}. Please check the path.")
#         sys.exit(1)
#     Q_matrix_path = os.path.join(pwd, "../../data/0819/disparities/Q.npy")
#     disparity_data = np.load(disparity_path)
#     Q_matrix = np.load(Q_matrix_path)

#     disparity = disparity_data['pred_disp']

#     disparity_init = disparity[0,:, :]

#     print(f"Disparity shape: {disparity_init.shape}")

#     pointcloud = disparity_to_pointcloud(disparity_init, Q_matrix)


#     cam_pos_world_position = np.array([0.03343094, -0.01157302, 0.15718198])
#     cam_pos_world_quaternion = np.array([0.18658106, 0.77200158, -0.604802, -0.05844316])

#     cam_pos_psm1_position = np.array([0.08880545, -0.04930693, -0.03779696])
#     cam_pos_psm1_quaternion = np.array([0.20070423, 0.2857976, -0.9321147, 0.0959152])

#     cam_pos_psm2_position = np.array([-0.07828412, 0.01814246, 0.07171638])
#     cam_pos_psm2_quaternion = np.array([0.11987684, -0.2255778, -0.95644951, 0.14123927])

#     psm1_T_cam = pose_to_matrix(cam_pos_psm1_position, cam_pos_psm1_quaternion)
#     cam_T_psm1 = np.linalg.inv(psm1_T_cam)

#     pointcloud_psm1 = cam_T_psm1 @ np.vstack((pointcloud.T, np.ones(pointcloud.shape[0])))
#     pointcloud_psm1 = pointcloud_psm1[:3, :].T
#     print(f"Pointcloud PSM1 shape: {pointcloud_psm1.shape}")
#     pointcloud = pointcloud_psm1
#     # Create 3D plot


#     # Call the function to plot the pointcloud
#     plot_pointcloud(pointcloud)

# if __name__ == "__main__":
#     main()