#!/usr/bin/env python3

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped

try:
    from cv_bridge import CvBridge
except Exception:  # pragma: no cover
    CvBridge = None  # type: ignore

import cv2
from scipy.spatial.transform import Rotation as R


# ============================
# Camera / Geometry Utilities
# ============================
@dataclass
class CameraModel:
    """Minimal pinhole model driven by an OpenCV Q matrix.

    Q is the 4x4 reprojection matrix from cv2.stereoRectify.
    If your disparity sign is flipped, set `use_negative_disp=True`.
    """

    Q: np.ndarray  # (4,4)
    use_negative_disp: bool = True

    def reproject_to_3d(self, disparity: np.ndarray) -> np.ndarray:
        disp = disparity.astype(np.float32)
        if self.use_negative_disp:
            disp = -disp
        # Returns an (H, W, 3) float32 array in *camera* frame units
        return cv2.reprojectImageTo3D(disp, self.Q)

    def pixel_to_point(self, disparity: np.ndarray, u: int, v: int) -> np.ndarray:
        pts3d = self.reproject_to_3d(disparity)
        return pts3d[int(v), int(u)].astype(np.float32)


class RigidTransform:
    """4x4 homogeneous transform with convenience constructors."""

    def __init__(self, T: np.ndarray):
        assert T.shape == (4, 4)
        self.T = T.astype(np.float32)

    @staticmethod
    def from_pos_quat_xyzw(pos: np.ndarray, quat_xyzw: np.ndarray) -> "RigidTransform":
        pos = np.asarray(pos, dtype=np.float32).reshape(3)
        qxyzw = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
        Rm = R.from_quat(qxyzw).as_matrix().astype(np.float32)  # SciPy expects [x,y,z,w]
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = Rm
        T[:3, 3] = pos
        return RigidTransform(T)

    def apply_to_point(self, p_xyz: np.ndarray) -> np.ndarray:
        p = np.asarray(p_xyz, dtype=np.float32).reshape(3)
        ph = np.hstack([p, 1.0])
        out = (self.T @ ph)[:3]
        return out.astype(np.float32)


# ============================
# Robot Publishers (PSM1)
# ============================
class PSMController:
    """Tiny wrapper exposing move_cp(position, quat)."""

    def __init__(self, node: Node, ns: str = "PSM1"):
        self._node = node
        self._move_cp_pub = node.create_publisher(PoseStamped, f"/{ns}/move_cp", 10)
        # For keeping the current tool orientation
        self._measured_cp_sub = node.create_subscription(
            PoseStamped, f"/{ns}/measured_cp", self._on_measured_cp, QoSProfile(depth=10)
        )
        self._last_measured_cp: Optional[PoseStamped] = None

    def _on_measured_cp(self, msg: PoseStamped):
        self._last_measured_cp = msg

    def current_orientation_xyzw(self) -> Tuple[float, float, float, float]:
        if self._last_measured_cp is None:
            # Fallback: identity orientation
            return (0.0, 0.0, 0.0, 1.0)
        q = self._last_measured_cp.pose.orientation
        return (q.x, q.y, q.z, q.w)

    def move_cp(self, position_xyz: Tuple[float, float, float], quat_xyzw: Optional[Tuple[float, float, float, float]] = None,
                frame_id: str = "PSM1_base"):
        if quat_xyzw is None:
            quat_xyzw = self.current_orientation_xyzw()
        ps = PoseStamped()
        ps.header.stamp = self._node.get_clock().now().to_msg()
        ps.header.frame_id = frame_id
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, position_xyz)
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = map(float, quat_xyzw)
        self._move_cp_pub.publish(ps)
        self._node.get_logger().info(
            f"move_cp → frame={frame_id} pos=({ps.pose.position.x:.4f},{ps.pose.position.y:.4f},{ps.pose.position.z:.4f}) "
            f"quat=({ps.pose.orientation.x:.3f},{ps.pose.orientation.y:.3f},{ps.pose.orientation.z:.3f},{ps.pose.orientation.w:.3f})"
        )


# ============================
# The ROS2 Node
# ============================
class DissectionNode(Node):
    """All wiring kept here; logic delegated to small helpers above."""

    def __init__(self):
        super().__init__("dissection_control_simple")
        self.bridge = CvBridge() if CvBridge is not None else None

        # --- State ---
        self.left_img: Optional[np.ndarray] = None
        self.disparity: Optional[np.ndarray] = None
        self.cam: Optional[CameraModel] = None
        self._clicked_points_px: List[Tuple[int, int]] = []
        self._did_interaction = False

        # --- Load T_psm1_cam (pos=[m], quat=[x,y,z,w]) ---
        try:
            data = np.load("/docker-ros/ws/src/arclab_dvrk/assets/cam_pose_psm1.npz")
            pos = data["pos"].astype(np.float32).reshape(3)
            quat_xyzw = data["quat"].astype(np.float32).reshape(4)
            self.T_psm1_cam = RigidTransform.from_pos_quat_xyzw(pos, quat_xyzw)
            self.get_logger().info("Loaded cam→PSM1 transform from npz.")
        except Exception as e:
            self.get_logger().warn(f"Failed to load cam_pose_psm1.npz: {e}; using identity.")
            self.T_psm1_cam = RigidTransform(np.eye(4, dtype=np.float32))

        # --- Robot interface ---
        self.psm1 = PSMController(self)

        # --- Subscriptions ---
        sensor_qos = qos_profile_sensor_data
        self.create_subscription(Image, "/stereo/left/rectified_downscaled_image", self._on_left, sensor_qos)
        self.create_subscription(Image, "/stereo/disparity", self._on_disp, sensor_qos)
        self.create_subscription(Float32MultiArray, "/stereo/rectified/Q", self._on_Q, QoSProfile(depth=5))

        # --- Timers ---
        self.create_timer(0.2, self._maybe_interact)
        self.get_logger().info("Node ready — waiting for left image, disparity, and Q.")

    # ----- Callbacks -----
    def _on_left(self, msg: Image):
        self.left_img = self._to_cv(msg)

    def _on_disp(self, msg: Image):
        self.disparity = self._to_cv(msg)

    def _on_Q(self, msg: Float32MultiArray):
        try:
            Q = np.array(msg.data, dtype=np.float32).reshape(4, 4)
            self.cam = CameraModel(Q=Q, use_negative_disp=True)
        except Exception as e:
            self.get_logger().warn(f"Bad Q message: {e}")

    # ----- Helpers -----
    def _to_cv(self, msg: Image):
        if self.bridge is None:
            raise RuntimeError("cv_bridge is required for this script.")
        if msg.encoding in ("32FC1", "16UC1", "mono16"):
            return self.bridge.imgmsg_to_cv2(msg)
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _ready(self) -> bool:
        return self.left_img is not None and self.disparity is not None and self.cam is not None

    # ----- Interaction -----
    def _maybe_interact(self):
        if self._did_interaction or not self._ready():
            return
        try:
            p1_cam, p2_cam = self._pick_two_points_and_lift()
        except Exception as e:
            self.get_logger().warn(f"Picking failed: {e}")
            return

        # Optional scale if your Q units are cm (set to 0.01); set to 1.0 if already meters
        SCALE = 0.01
        p1_cam *= SCALE
        p2_cam *= SCALE
        self.get_logger().info(f"P1_cam(m)={p1_cam.tolist()} | P2_cam(m)={p2_cam.tolist()}")

        # Transform to PSM1 frame
        p1_psm1 = self.T_psm1_cam.apply_to_point(p1_cam)
        self.get_logger().info(f"P1_psm1(m)={p1_psm1.tolist()}")

        # Move once to the negative of P1 (gentle retract example)
        goal = (-p1_psm1).astype(np.float32)
        self.psm1.move_cp(tuple(goal))

        self._did_interaction = True

    def _pick_two_points_and_lift(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.left_img is not None and self.disparity is not None and self.cam is not None
        img = self.left_img.copy()

        picked: List[Tuple[int, int]] = []
        win = "Click 2 points (ESC to cancel)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, 720)

        def draw():
            vis = img.copy()
            for i, (u, v) in enumerate(picked, 1):
                cv2.circle(vis, (u, v), 6, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(vis, f"P{i}", (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, "Click 2 points | r=reset | ESC=cancel", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
            return vis

        def mouse(evt, x, y, *_):
            if evt == cv2.EVENT_LBUTTONDOWN and len(picked) < 2:
                picked.append((int(x), int(y)))

        cv2.setMouseCallback(win, mouse)
        while True:
            cv2.imshow(win, draw())
            k = cv2.waitKey(20) & 0xFF
            if k == 27:  # ESC
                cv2.destroyWindow(win)
                raise RuntimeError("User cancelled.")
            if k == ord("r"):
                picked.clear()
            if len(picked) >= 2:
                break
        cv2.destroyWindow(win)

        # Lift to 3D via Q+disparity
        (u1, v1), (u2, v2) = picked
        p1_cam = self.cam.pixel_to_point(self.disparity, u1, v1)
        p2_cam = self.cam.pixel_to_point(self.disparity, u2, v2)
        return p1_cam, p2_cam


# ============================
# Main
# ============================

def main():
    rclpy.init()
    node = DissectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
