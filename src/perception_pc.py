# perception_pc.py
import numpy as np
import cv2
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


def disparity_to_pointcloud(
    disparity: np.ndarray,
    Q: np.ndarray,
    *,
    use_negative_disp: bool = True,
    max_points: int | None = 4096,
    scale: float = 1.0,
    mask: np.ndarray | None = None,
    clip_z: tuple[float, float] | None = (1e-6, np.inf),
    return_colors: bool = False,
    image: np.ndarray | None = None
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:

    disp = disparity.astype(np.float32)
    if use_negative_disp: 
        disp = -disp

    pts = cv2.reprojectImageTo3D(disp, Q, handleMissingValues=True)
    H, W = disp.shape[:2]

    m = np.isfinite(pts).all(axis=2)
    if clip_z is not None:
        z = pts[..., 2]
        m &= np.isfinite(z) & (z >= clip_z[0]) & (z <= clip_z[1])
    if mask is not None:
        m &= mask.astype(bool)
    P = pts[m]  # (N,3)

    C = None
    if return_colors and (image is not None):
        if image.ndim == 2:
            I = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            I = image
        C = I.reshape(-1, I.shape[-1])[m.reshape(-1)].astype(np.uint8)

    if max_points is not None and P.shape[0] > max_points:
        idx = np.random.choice(P.shape[0], max_points, replace=False)
        P = P[idx]
        if C is not None:
            C = C[idx]

    P = (P * float(scale)).astype(np.float32)

    return (P, C) if return_colors else P

