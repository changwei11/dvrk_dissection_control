import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_to_T_xyzw(pos, quat_xyzw):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix().astype(np.float32)
    T[:3,  3] = np.asarray(pos, dtype=np.float32).reshape(3)
    return T

def apply_T(T, p_xyz):
    p = np.asarray(p_xyz, np.float32).reshape(3)
    return (T @ np.hstack([p, 1.0]))[:3].astype(np.float32)

def quaternion_wxyz_to_rotation_matrix(quat_wxyz):
    """
    Converts a quaternion in (w, x, y, z) format to a 3x3 rotation matrix.
    """
    quat_xyzw = np.roll(quat_wxyz, -1)  # convert to (x, y, z, w)
    R_mat = R.from_quat(quat_xyzw).as_matrix()
    return R_mat

def rotation_matrix_to_quaternion_wxyz(R_mat):
    """
    Converts a 3x3 rotation matrix to a quaternion in (w, x, y, z) format.
    """
    quat_xyzw = R.from_matrix(R_mat).as_quat()  # returns (x, y, z, w)
    quat_wxyz = np.roll(quat_xyzw, 1)  # convert to (w, x, y, z)
    return quat_wxyz


def pose_to_matrix(position, quaternion):
    """Convert position and quaternion to a 4x4 transformation matrix."""
    rotation = quaternion_wxyz_to_rotation_matrix(quaternion)
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    return T

def matrix_to_pose(T):
    """Convert a 4x4 transformation matrix to position and quaternion."""
    position = T[:3, 3]
    quaternion = rotation_matrix_to_quaternion_wxyz(T[:3, :3])

    
    return position, quaternion

