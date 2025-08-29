#!/usr/bin/env python3

from __future__ import annotations

import math
import numpy as np
from typing import Optional, List
import time
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data, QoSProfile

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, PointCloud2, JointState
from geometry_msgs.msg import PoseStamped

try:
    from cv_bridge import CvBridge
except Exception:
    CvBridge = None  # type: ignore
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def pose_to_matrix(pos, quat):
    """Convert position + quaternion to 4x4 homogeneous transform"""
    T = np.eye(4)
    # rotation
    R_mat = R.from_quat(quat).as_matrix()   # quat format = [x,y,z,w]
    T[:3, :3] = R_mat
    # translation
    T[:3, 3] = pos
    return T

class ros_utils:

    @staticmethod
    def wait_for_message(node: Node, topic: str, msg_type, timeout_sec: Optional[float] = 5.0):
        from rclpy.task import Future

        future: Future = Future()
        holder = {"sub": None}

        def _once(msg):
            if not future.done():
                future.set_result(msg)
                try:
                    if holder["sub"] is not None:
                        node.destroy_subscription(holder["sub"])  # type: ignore[arg-type]
                except Exception:
                    pass

        holder["sub"] = node.create_subscription(msg_type, topic, _once, 10)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
        if not future.done():
            raise TimeoutError(f"Timeout waiting for message on {topic}")
        return future.result()




class DissectionControl(Node):
    def __init__(self):
        super().__init__('dissection_control')


        self.bridge = CvBridge() if CvBridge is not None else None
        if self.bridge is None:
            self.get_logger().warn('cv_bridge not available; image callbacks will store raw ROS Image messages only.')

        try:
            q_msg: Float32MultiArray = ros_utils.wait_for_message(self, '/stereo/rectified/Q', Float32MultiArray, timeout_sec=10.0)
            self.Q = np.array(q_msg.data, dtype=np.float32).reshape(4, 4)
            self.get_logger().info('Camera parameters (Q) received.')
        except Exception as e:
            self.Q = np.eye(4, dtype=np.float32)
            self.get_logger().warn(f'Failed to receive /stereo/rectified/Q in time. Using identity Q. Error: {e}')

        self.left_img = None
        self.right_img = None
        self.disparity = None
        self.point_cloud = None

        self.point_cloud = None

        self._want_clicks = True                # trigger once at startup
        self._picked_pixels: list[tuple[int,int]] = []
        self._picked_points3d: list[np.ndarray] = []
        self._did_quick_test_move = False

        self.retractor_js: Optional[JointState] = None
        self.retractor_jaw_js: Optional[JointState] = None
        self.retractor_cp: Optional[PoseStamped] = None

        self.cutter_js: Optional[JointState] = None
        self.cutter_jaw_js: Optional[JointState] = None
        self.cutter_cp: Optional[PoseStamped] = None

        sensor_qos = qos_profile_sensor_data
        default_qos = QoSProfile(depth=10)
        self.target_point_1 = None
        self._sent_retractor_goal = False
        self.target_point_2 = None
        self.cam_pos_psm1 = np.load("/docker-ros/ws/src/arclab_dvrk/assets/cam_pose_psm1.npz")

        self.T_psm1_cam = np.eye(4, dtype=np.float32)
        try:
            data = self.cam_pos_psm1
            pos  = data["pos"].astype(np.float32).reshape(3)
            quat = data["quat"].astype(np.float32).reshape(4)  # assumed (w,x,y,z)
            self.T_psm1_cam = pose_to_matrix(pos, quat)

            self.get_logger().info(f"Loaded T_psm1_cam:\n{self.T_psm1_cam}")
        except Exception as e:
            self.get_logger().warn(f"Failed to load cam_pos_psm1.npz: {e} (using identity)")



        self.cam_pos_psm2 = np.load("/docker-ros/ws/src/arclab_dvrk/assets/cam_pose_psm2.npz")
        self.cam_pose_world = np.load("/docker-ros/ws/src/arclab_dvrk/assets/cam_pose_world.npz")

        self.img_left_sub = self.create_subscription(Image, '/stereo/left/rectified_downscaled_image', self._callback_left_img, sensor_qos)
        self.img_right_sub = self.create_subscription(Image, '/stereo/right/rectified_downscaled_image', self._callback_right_img, sensor_qos)
        self.disp_sub = self.create_subscription(Image, '/stereo/disparity', self._callback_disp, sensor_qos)

        # Retractors (PSM1)
        self.retractor_joint_state_sub = self.create_subscription(JointState, '/PSM1/measured_js', self._callback_retractor_joint_state, default_qos)
        self.retractor_jaw_state_sub = self.create_subscription(JointState, '/PSM1/jaw/measured_js', self._callback_retractor_jaw_state, default_qos)
        self.retractor_ee_frame_sub = self.create_subscription(PoseStamped, '/PSM1/measured_cp', self._callback_retractor_ee_frame, default_qos)

        # Cutters (PSM2)
        self.cutter_joint_state_sub = self.create_subscription(JointState, '/PSM2/measured_js', self._callback_cutter_joint_state, default_qos)
        self.cutter_jaw_state_sub = self.create_subscription(JointState, '/PSM2/jaw/measured_js', self._callback_cutter_jaw_state, default_qos)
        self.cutter_ee_frame_sub = self.create_subscription(PoseStamped, '/PSM2/measured_cp', self._callback_cutter_ee_frame, default_qos)


        self.set_retractor_gripperServo_pub = self.create_publisher(JointState, '/PSM1/jaw/servo_jp', 10)
        # self.set_gripperMove_pub = self.create_publisher(JointState, '/PSM1/jaw/move_jp', 10)
        self.retractorServo_pub = self.create_publisher(JointState, '/PSM1/servo_jp', 10)
        self.retractorMoveCP_pub = self.create_publisher(PoseStamped, '/PSM1/move_cp', 10)
        self.retractorMove_pub = self.create_publisher(JointState, '/PSM1/move_jp', 10)

        self.cutterMove_pub = self.create_publisher(JointState, '/PSM2/move_jp', 10)
        self.cutterServo_pub = self.create_publisher(JointState, '/PSM2/servo_jp', 10)
        self.cutterMoveCP_pub = self.create_publisher(PoseStamped, '/PSM2/move_cp', 10)
        self.set_cutter_gripperServo_pub = self.create_publisher(JointState, '/PSM2/jaw/servo_jp', 10)


        self._preview_once = True

        # self.create_timer(2.0, self._heartbeat)
        # self.create_timer(0.1, self._callback_disp)
        self.create_timer(0.1, self.test_perception)
        self.get_logger().info('dissection_control node initialized.')

    def _to_cv(self, msg: Image):
        if self.bridge is None:
            return msg
        try:
            if msg.encoding in ('32FC1', '16UC1', 'mono16'):
                return self.bridge.imgmsg_to_cv2(msg)
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge convert failed: {e}')
            return msg

    def _callback_left_img(self, msg: Image):
        self.left_img = self._to_cv(msg)

    def _callback_right_img(self, msg: Image):
        self.right_img = self._to_cv(msg)


    def _callback_disp(self, msg: Image):
        self.disparity = self._to_cv(msg)


    def _callback_retractor_joint_state(self, msg: JointState):
        self.retractor_js = msg

    def _callback_retractor_jaw_state(self, msg: JointState):
        self.retractor_jaw_js = msg

    def _callback_retractor_ee_frame(self, msg: PoseStamped):
        self.retractor_cp = msg

    def _callback_cutter_joint_state(self, msg: JointState):
        self.cutter_js = msg

    def _callback_cutter_jaw_state(self, msg: JointState):
        self.cutter_jaw_js = msg

    def _callback_cutter_ee_frame(self, msg: PoseStamped):
        self.cutter_cp = msg

    def send_retractor_jaw_servo(self, jaw_position_rad: float):
        """Command PSM1 jaw via /PSM1/jaw/servo_jp (JointState.position[0])."""
        msg = JointState()
        msg.name = ['PSM1_jaw']
        msg.position = [jaw_position_rad]
        self.set_retractor_gripperServo_pub.publish(msg)

    def send_cutter_jaw_servo(self, jaw_position_rad: float):
        msg = JointState()
        msg.name = ['PSM2_jaw']
        msg.position = [jaw_position_rad]
        self.set_cutter_gripperServo_pub.publish(msg)

    def send_retractor_servo_jp(self, joint_positions: List[float]):
        msg = JointState()
        msg.name = [f'PSM1_joint_{i}' for i in range(len(joint_positions))]
        msg.position = list(joint_positions)
        self.retractorServo_pub.publish(msg)

    def send_retractor_move_jp(self, joint_positions: List[float]):
        msg = JointState()
        msg.name = [f'PSM1_joint_{i}' for i in range(len(joint_positions))]
        msg.position = list(joint_positions)
        self.retractorMove_pub.publish(msg)

    def send_cutter_servo_jp(self, joint_positions: List[float]):
        msg = JointState()
        msg.name = [f'PSM2_joint_{i}' for i in range(len(joint_positions))]
        msg.position = list(joint_positions)
        self.cutterServo_pub.publish(msg) 

    def send_cutter_move_jp(self, joint_positions: List[float]):
        msg = JointState()
        msg.name = [f'PSM2_joint_{i}' for i in range(len(joint_positions))]
        msg.position = list(joint_positions)
        self.cutterMove_pub.publish(msg)

    def send_retractor_servo_cp(self, pose: PoseStamped):
        """Directly publish a target PoseStamped for /PSM1/servo_cp."""
        self.retractorMoveCP_pub.publish(pose)


    def _PSMMoveCP(
        self,
        pub,
        goal_state: float,
        sleep_time: float = 3.0,
    ):
        """

        Set a retractor's Pose in Cartesian space
        args:
            pub: ROS publisher of the retractor's joint angle
            goal_state: target joint angle in radians
            sleep_time: time to wait for the retractor to respond
        """

        pose_msg = PoseStamped()

        pose_msg.pose.position.x = float(goal_state[0])
        pose_msg.pose.position.y = float(goal_state[1])
        pose_msg.pose.position.z = float(goal_state[2])

        pose_msg.pose.orientation.w = float(goal_state[3])  # ⇨ keep quaternion unit‑norm
        pose_msg.pose.orientation.x = float(goal_state[4])
        pose_msg.pose.orientation.y = float(goal_state[5])
        pose_msg.pose.orientation.z = float(goal_state[6])

        # -------- publish --------
        pub.publish(pose_msg)

        # Optional: wait until the low‑level trajectory finishes
        # You can monitor /PSM1/goal_reached or the 'is_busy' flag instead of sleep.
        time.sleep(sleep_time)


    def uniform_sampling_numpy(self, point_cloud, colors, num_points, reject_outliers: bool = True):
        if reject_outliers:
            point_cloud, _, _ = self.reject_outliers_by_radius(point_cloud.squeeze(0), method="sigma")  # Remove outliers first
            point_cloud = point_cloud[None, ...]  # Ensure shape is (B, N, C)
        B, N, C = point_cloud.shape
        # padd if num_points > N
        if num_points > N:
            return self.pad_point_numpy(point_cloud, num_points)
        
        # random sampling
        indices = np.random.permutation(N)[:num_points]
        sampled_points = point_cloud[:, indices]
        return sampled_points, colors[:, indices]
        
    def reject_outliers_by_radius(
        self,
        pts: np.ndarray,                  # (N, 3) float32 / float64
        method: str       = "sigma",      # "abs" | "sigma" | "percentile"
        abs_thresh: float = None,         # used when method == "abs"
        k_sigma: float    = 3.0,          # used when method == "sigma"
        pct: float        = 99.5          # used when method == "percentile"
        ):
        """
        Returns
        -------
        inliers  : (M, 3)   points that pass the test
        mask     : (N,)     boolean mask of kept rows
        thresh   : float    radius threshold actually used
        """
        # drop NaN / ±inf first — keeps logic simple
        finite_mask = np.isfinite(pts).all(axis=1)
        pts         = pts[finite_mask]

        dists = np.linalg.norm(pts, axis=1)   # (N,) Euclidean radii

        # -------- choose threshold ----------
        if method == "abs":
            if abs_thresh is None:
                raise ValueError("abs_thresh must be set for method='abs'")
            thresh = abs_thresh

        elif method == "sigma":
            mu, sigma = dists.mean(), dists.std()
            thresh = mu + k_sigma * sigma      # μ + kσ

        elif method == "percentile":
            thresh = np.percentile(dists, pct)

        else:
            raise ValueError(f"Unknown method '{method}'")

        # -------- build mask & slice ----------
        mask = dists <= thresh                 # keep pts inside radius
        inliers = pts[mask]

        # If you need a mask that matches the *original* array length:
        full_mask = np.zeros(finite_mask.shape, dtype=bool)
        full_mask[finite_mask] = mask          # re-insert into original indexing

        return inliers, full_mask, thresh
    
    def _quat_wxyz_to_R(self, q):
        q = np.asarray(q, dtype=np.float64)
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
        ], dtype=np.float32)
        return R

    def _Rt_to_T(self, R, t):
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.astype(np.float32)
        T[:3,  3] = np.asarray(t, dtype=np.float32).reshape(3)
        return T

    def transform_point(T, p_cam):
        p_h = np.hstack([p_cam, 1.0])          # make homogeneous
        p_psm1_h = T @ p_h                     # transform
        return p_psm1_h[:3]                    # back to xyz

    def transform_orientation(T, quat_cam):
        R_cam = R.from_quat(quat_cam).as_matrix()
        R_psm1_cam = T[:3, :3]
        R_psm1 = R_psm1_cam @ R_cam
        return R.from_matrix(R_psm1).as_quat()


    def _cv2_plot_pointcloud(self, pts: np.ndarray, colors_bgr: np.ndarray):

        if not hasattr(self, "_pcv_init"):
            self._pcv_W, self._pcv_H = 960, 720
            self._pcv_f = 900.0          # focal length in pixels
            self._pcv_yaw_deg   = 30.0   # left/right orbit
            self._pcv_pitch_deg = -20.0  # up/down tilt
            self._pcv_roll_deg  = 0.0
            cv2.namedWindow("PointCloud", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("PointCloud", self._pcv_W, self._pcv_H)
            self._pcv_init = True

        P = pts.astype(np.float32)
        good = np.isfinite(P).all(1)
        if not np.any(good):
            return
        P = P[good]
        C = np.median(P, axis=0)                    # center of mass
        P = P - C
        r = np.percentile(np.linalg.norm(P, axis=1), 95)
        r = float(r if r > 1e-6 else 1.0)           # avoid degenerate scales


        yd, pd, rd = map(np.deg2rad, (self._pcv_yaw_deg, self._pcv_pitch_deg, self._pcv_roll_deg))
        cy, sy = np.cos(yd), np.sin(yd)
        cp, sp = np.cos(pd), np.sin(pd)
        cr, sr = np.cos(rd), np.sin(rd)
        Ry = np.array([[ cy, 0, sy],[ 0, 1, 0],[-sy, 0, cy]], np.float32)
        Rx = np.array([[ 1, 0, 0],[ 0, cp,-sp],[ 0, sp, cp]], np.float32)
        Rz = np.array([[ cr,-sr, 0],[ sr, cr, 0],[ 0, 0, 1]], np.float32)
        R = Rz @ Rx @ Ry                              # (3,3)

        Pv = (P @ R.T)                                # rotate
        Pv[:, 2] += 2.5 * r                           # push in front of camera

        f = self._pcv_f
        cx, cy_ = self._pcv_W // 2, self._pcv_H // 2
        z = Pv[:, 2]
        vis = z > 1e-6
        if not np.any(vis):
            return
        x, y, z = Pv[vis, 0], Pv[vis, 1], z[vis]
        u = (f * x / z + cx).astype(np.int32)
        v = (cy_ - f * y / z).astype(np.int32)

        canvas = np.zeros((self._pcv_H, self._pcv_W, 3), np.uint8)
        inb = (u >= 0) & (u < self._pcv_W) & (v >= 0) & (v < self._pcv_H)
        if np.any(inb):
            # align colors with current mask chain
            cols = colors_bgr[good][vis][inb]
            canvas[v[inb], u[inb]] = cols

        def _proj(Xw):
            Xc = (Xw @ R.T)
            Xc[:, 2] += 2.5 * r
            z = Xc[:, 2]
            u = (f * Xc[:, 0] / z + cx).astype(np.int32)
            v = (cy_ - f * Xc[:, 1] / z).astype(np.int32)
            return u, v

        axis_len = 1.0 * r
        A = np.array([[0, 0, 0],
                    [axis_len, 0, 0],
                    [0, axis_len, 0],
                    [0, 0, axis_len]], np.float32)
        uA, vA = _proj(A)
        # BGR colors: X=red, Y=green, Z=blue
        # --- highlight the two picked 3D points, if available ---
        if getattr(self, "_picked_points3d", None):
            Pp = np.array(self._picked_points3d, dtype=np.float32)
            # match centering/scaling of the cloud
            Pp = Pp - C
            Pp = (Pp @ R.T)
            Pp[:, 2] += 2.5 * r
            z2 = Pp[:, 2]
            good2 = z2 > 1e-6
            if np.any(good2):
                u2 = (f * Pp[good2, 0] / z2[good2] + cx).astype(np.int32)
                v2 = (cy_ - f * Pp[good2, 1] / z2[good2]).astype(np.int32)
                for uu, vv in zip(u2, v2):
                    if 0 <= uu < self._pcv_W and 0 <= vv < self._pcv_H:
                        cv2.circle(canvas, (uu, vv), 6, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(canvas, (uu, vv), 10, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.line(canvas, (uA[0], vA[0]), (uA[1], vA[1]), (0, 0, 255), 2, cv2.LINE_AA)  # X
        cv2.line(canvas, (uA[0], vA[0]), (uA[2], vA[2]), (0, 255, 0), 2, cv2.LINE_AA)  # Y
        cv2.line(canvas, (uA[0], vA[0]), (uA[3], vA[3]), (255, 0, 0), 2, cv2.LINE_AA)  # Z
        cv2.putText(canvas, "X", (uA[1]+6, vA[1]+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Y", (uA[2]+6, vA[2]+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Z", (uA[3]+6, vA[3]+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('a'): self._pcv_yaw_deg   -= 3
        if k == ord('d'): self._pcv_yaw_deg   += 3
        if k == ord('w'): self._pcv_pitch_deg -= 3
        if k == ord('s'): self._pcv_pitch_deg += 3
        if k == ord('q'): self._pcv_roll_deg  -= 3
        if k == ord('e'): self._pcv_roll_deg  += 3
        if k == ord('-'): self._pcv_f = max(100.0, self._pcv_f * 0.9)
        if k == ord('=') or k == ord('+'): self._pcv_f = min(5000.0, self._pcv_f * 1.1)

        cv2.imshow("PointCloud", canvas)
    # ---------------------------------------------------------------------------
    def pick_two_points_on_left(self, window_name: str = "Left Image — click 2 points"):
        """
        Shows the current left image, lets you click exactly 2 pixels, and saves:
        - self._picked_pixels   : [(u1,v1), (u2,v2)]
        - self._picked_points3d : [XYZ1, XYZ2] in the left-camera frame from Q+disparity
        Also writes both to 'picked_points.npz'.
        Keys:
        r = reset,  ESC = cancel
        """
        if self.left_img is None or self.disparity is None:
            raise RuntimeError("Left image or disparity not available yet.")

        img = self.left_img
        disp = self.disparity

        if disp.ndim != 2:
            # If disparity came in as a 3-channel viz image by mistake, convert to gray/float
            if disp.ndim == 3:
                disp = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
            disp = disp.astype(np.float32)

        H_img, W_img = img.shape[:2]
        H_disp, W_disp = disp.shape[:2]
        if (H_img, W_img) != (H_disp, W_disp):
            self.get_logger().warn(
                f"Left image size {W_img}x{H_img} != disparity size {W_disp}x{H_disp}. "
                "Clicks assume same resolution — results may be off."
            )

        # Local state for this interaction
        picked: list[tuple[int,int]] = []
        vis = img.copy()

        def draw_overlay():
            tmp = vis.copy()
            for i, (u, v) in enumerate(picked, 1):
                cv2.circle(tmp, (u, v), 6, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(tmp, f"P{i}", (u+8, v-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(tmp, "Click 2 points (r=reset, ESC=cancel)",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            return tmp

        def on_mouse(event, x, y, flags, param):
            nonlocal picked, vis
            if event == cv2.EVENT_LBUTTONDOWN and len(picked) < 2:
                x = int(np.clip(x, 0, W_img-1))
                y = int(np.clip(y, 0, H_img-1))
                picked.append((x, y))

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)
        cv2.setMouseCallback(window_name, on_mouse)

        # Interaction loop (blocks until 2 clicks or ESC)
        while True:
            cv2.imshow(window_name, draw_overlay())
            k = cv2.waitKey(20) & 0xFF
            if k == 27:  # ESC
                cv2.destroyWindow(window_name)
                self.get_logger().info("Point picking cancelled.")
                return
            if k == ord('r'):
                picked.clear()
            if len(picked) >= 2:
                break

        cv2.destroyWindow(window_name)

        # Compute per-pixel 3D via Q and disparity
        # NOTE: matches your usage: reprojectImageTo3D(-disp, Q)
        pts3d_img = cv2.reprojectImageTo3D(-disp.astype(np.float32), self.Q)  # (H,W,3)

        pts3d_list: list[np.ndarray] = []
        for (u, v) in picked:
            p = pts3d_img[int(v), int(u)].astype(np.float32)
            if not np.isfinite(p).all():
                self.get_logger().warn(f"Picked pixel ({u},{v}) has invalid 3D ({p}).")
            pts3d_list.append(p)


        # Save on the instance
        self._picked_pixels = picked
        self._picked_points3d = pts3d_list

        pts_cam = np.array(pts3d_list, dtype=np.float32)
        pts_cam[0] = pts_cam[0] * 0.01
        pts_cam[1] = pts_cam[1] * 0.01
        print("Points in camera frame (m):", pts_cam[0])
        self.target_point_1 =  self.transform_point(self.T_psm1_cam, pts_cam[0])
        self.target_point_2 =  self.transform_point(self.T_psm1_cam, pts_cam[1])
        print("Point in PSM1 frame:", self.target_point_1)



        # # Persist to disk with both frames
        # out_path = "picked_points.npz"
        # np.savez(out_path,
        #         pixels=np.array(picked, dtype=np.int32),
        #         points3d_cam=pts_cam,
        #         points3d_psm1=np.array(pts3d_list, dtype=np.float32) * 0.01,)
        # self.get_logger().info(
        #     "Targets set:\n"
        #     f"  P1 PSM1 (cm): {self.target_point_1.tolist()}\n"
        #     f"  P2 PSM1 (cm): {self.target_point_2.tolist()}\n"
        #     f"Saved -> {out_path}"
        # )
        try:
            self.control_retractor()
        except Exception as e:
            self.get_logger().warn(f"control_retractor failed: {e}")


        # Persist to disk
        out_path = "picked_points.npz"
        np.savez(out_path,
                pixels=np.array(picked, dtype=np.int32),
                points3d=np.array(pts3d_list, dtype=np.float32))
        self.get_logger().info(
            f"Picked pixels: {picked} | 3D (m): {[p.tolist() for p in pts3d_list]} | saved -> {out_path}"
        )

    def test_perception(self):
            if self.disparity is None:
                print("Waiting for pc, disp or images…")
                return

            now = self.get_clock().now()

            import math
            if hasattr(self, "_last_call_time"):
                dt   = (now - self._last_call_time).nanoseconds * 1e-9   # seconds
                freq = 1.0 / dt if dt > 0 else math.inf
                self._call_count += 1
            else:
                self._call_count = 1
            self._last_call_time = now
        

            if self._want_clicks and (self.left_img is not None) and (self.disparity is not None):
                try:
                    self.pick_two_points_on_left()
                except Exception as e:
                    self.get_logger().warn(f"Point picking failed: {e}")
                finally:
                    self._want_clicks = False

            # Observations 
            left_img = self.left_img
            right_img = self.right_img
            disp = self.disparity

            disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
            disp_vis = disp_vis.astype("uint8")
            disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

            points_3d = cv2.reprojectImageTo3D(disp, self.Q).reshape(-1, 3)
            pts, rgb = self.uniform_sampling_numpy(point_cloud = points_3d[None, ...], 
                                                colors=disp_vis.reshape(-1, 3)[None, ...], 
                                                num_points=4096, 
                                                reject_outliers=True) # sample some points
            pts, rgb = pts.squeeze(0), rgb.squeeze(0)
            # print("pointcloud shape:", pts.shape)
            self._cv2_plot_pointcloud(pts, rgb)


    def _heartbeat(self):
        # Lightweight status indicator
        left = 'received' if self.left_img is not None else 'miss'
        right = 'received' if self.right_img is not None else 'miss'
        disp = 'received' if self.disparity is not None else 'miss'
        self.get_logger().info(f'Streams - L:{left} R:{right} Disp:{disp}')


    def control_retractor(self):
        if self.target_point_1 is None:
            self.get_logger().warn("control_retractor: target_point_1 is not set yet.")
            return

        print("target 1", self.target_point_1)
        # Compute target = -1 * target_point_1
        p = (-1.0) * np.asarray(self.target_point_1, dtype=np.float32)
        # [0.6197879  0.47835505 1.0093435 ]
        # target_pose = np.array([0.025467625682240702,0.03458231270745048, -0.12948347516958353, 0.66870057119975, 0.30563288554600837, 0.6631421889044691, -0.14025164043581886])
        target_pose = np.array([0.025467625682240702,0.03458231270745048, -0.1048347516958353, 0.66870057119975, 0.30563288554600837, 0.6631421889044691, -0.14025164043581886])
        self._PSMMoveCP(
            pub=self.retractorMoveCP_pub,
            goal_state=target_pose,
            sleep_time=0.01
        )

    def quick_test_move_psm1(self):
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()

        ps.header.frame_id = (self.retractor_cp.header.frame_id
                            if self.retractor_cp is not None else "PSM1_base")

        ps.pose.position.x = 0.025467625682240702
        ps.pose.position.y = 0.03458231270745048
        ps.pose.position.z = -0.12948347516958353

        ox = 0.66870057119975
        oy = 0.30563288554600837
        oz = 0.6631421889044691
        ow = -0.14025164043581886

        import math
        n = math.sqrt(ox*ox + oy*oy + oz*oz + ow*ow)
        if n > 1e-9:
            ox, oy, oz, ow = ox/n, oy/n, oz/n, ow/n
        ps.pose.orientation.x = ox
        ps.pose.orientation.y = oy
        ps.pose.orientation.z = oz
        ps.pose.orientation.w = ow

        self.retractorMoveCP_pub.publish(ps)
        self.get_logger().info(
            f"Quick test: sent PSM1 servo_cp in frame '{ps.header.frame_id}' "
            f"pos=({ps.pose.position.x:.4f},{ps.pose.position.y:.4f},{ps.pose.position.z:.4f}) "
            f"quat(xyz w)=({ox:.4f},{oy:.4f},{oz:.4f} {ow:.4f})"
        )
        self._did_quick_test_move = True

def main(args=None):
    rclpy.init(args=args)
    node = DissectionControl()
    try:
        # Multi-threaded to keep callbacks responsive while using wait_for_message helper
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
