import numpy as np
import cv2

class CameraModel:
    """Tiny wrapper around OpenCV Q reprojection."""
    def __init__(self, Q: np.ndarray, use_negative_disp: bool = True, scale: float = 0.001):
        self.Q = Q.astype(np.float32).reshape(4,4)
        self.neg = use_negative_disp
        self.scale = float(scale)  # 0.01 if Q outputs cm, 1.0 if meters

    def point_from_pixel(self, disparity_img: np.ndarray, u: int, v: int) -> np.ndarray:
        disp = disparity_img.astype(np.float32)
        pts = cv2.reprojectImageTo3D(disp, self.Q, handleMissingValues=True) * self.scale
        print(f"Point from pixel ({u}, {v}): {pts[int(v), int(u)]}")
        return (pts[int(v), int(u)].astype(np.float32))

    def center_point(self, disparity_img: np.ndarray) -> np.ndarray:
        H, W = disparity_img.shape[:2]
        return self.point_from_pixel(disparity_img, W//2, H//2)
