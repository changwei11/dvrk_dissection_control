#!/usr/bin/env python3
from typing import Optional
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image

from controller import PSM1Controller, PSM2Controller
from planner import PSM1PickPlanner
from cutter import PSM2CutPlanner
from revealing_control import PSM1RevealPlanner
from perception import CameraModel


from transforms import *
from perception_pc import disparity_to_pointcloud
import open3d as o3d
import cv2
try:
    from cv_bridge import CvBridge
except Exception:
    CvBridge = None
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lite_tracker', 'scripts', 'lite_tracker'))
from tracking_module_new import LiteTrackingModule, TrackerConfig
import time

class DissectionNode(Node):

    def __init__(self):
        super().__init__("robot_app")
        self.bridge = CvBridge() if CvBridge else None

        # State
        self.left: Optional[np.ndarray] = None
        self.disp: Optional[np.ndarray] = None
        self.cam: Optional[CameraModel] = None
        self._did_pick_points = False

        self.target_1 = None
        self.target_2 = None
        self.picked_pixels: Optional[list] = None
        self.picked_points_cam: Optional[np.ndarray] = None
        self.picked_point_psm1: Optional[np.ndarray] = None

        self._cut_targets_cam = []
        self._cut_targets_base = []
        self._cut_idx = 0
        self._cut_total = 0

        self.trk_cfg = TrackerConfig(
            checkpoint="./weights/scaled_online.pth",
            grid_query_frame=0,
            use_autocast=True,
            save_debug_plots=False,
        )
        self.trk = LiteTrackingModule(self.trk_cfg)
        self._trk_ready = False
        self._tracked_uv = None    
        self._tracking_active = False
        self._tracked_pixels = None
        self.psm1 = PSM1Controller(self, ns="PSM1", frame_id="PSM1_base")
        self.create_subscription(Image, "/stereo/left/rectified_downscaled_image", self.callback_left, qos_profile_sensor_data)
        self.create_subscription(Image, "/stereo/disparity", self.callback_disp, qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, "/stereo/rectified/Q", self.callback_Q, QoSProfile(depth=5))

        self.psm2 = PSM2Controller(self, ns="PSM2", frame_id="PSM2_base")
        self.psm2_cut = PSM2CutPlanner(
            self.psm2,
            move_cmd_s=2.0,
            deep_cmd_s=2.0,
            close_cmd_s=2.0,
            jaw_hold_close=-0.3
        )


        self.reveal = PSM1RevealPlanner(
            self.psm1,
            retract_cmd_s=5.0,
            retract_hz=20.0,
            retract_dist_m=0.020,   # 2 cm along tool -Z
            jaw_hold_close=-1.0,
            keep_orientation=True,
        )
        self._cutter_started= False

        self.planner = PSM1PickPlanner(self.psm1)
        self._plan_started = False
        self.create_timer(0.1, self._tick)   # 5 Hz

        self.get_logger().info("RobotApp ready.")

        self._phase = "PSM1"
        self._plan_started = False
        self._cutter_started = False
        self._handoff_wait_s = 0.5
        self._handoff_t0 = None

        # after other state vars
        self.dissection_uvs: Optional[list[tuple[int,int]]] = None  # final 3 cutting UVs
        self._final_vis_done = False


        self.reveal_cfg = TrackerConfig(
            checkpoint="./weights/scaled_online.pth",
            grid_query_frame=0,
            use_autocast=True,
            save_debug_plots=False,
            pruner_threshold=2.0,
        )
        self.reveal_trk = LiteTrackingModule(self.reveal_cfg)
        self._reveal_ready = False
        self._reveal_active = False
        self._reveal_t0 = None
        self._reveal_stop_after_s = 8.0  # ← your termination condition (time); change as you like
        self.reveal_result = None        # ← where we store the final result dict

        # dissection/cutting targets also kept in UV for REVEAL init
        self._cut_targets_uv = []        # [(u,v), (u,v), (u,v)] captured after PSM2 init
        self._last_cut_poly4_uv = None   # 4-pt polyline of the most recent cut (for suggestion)

        # Live overlay window (OpenCV)
        self._viz_enable = True
        self._viz_win = "Live Tracker Overlay"
        self._viz_window_created = False
        self._viz_every = 1   # draw every frame; make 2/3/etc. to throttle
        self._viz_counter = 0


    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def callback_left(self, msg: Image):
        bgr = self._to_cv(msg)
        self.left = bgr

        if self._trk_ready and self._tracking_active:
        # if self._trk_ready:
            try:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                self.trk.step(rgb)
                self._tracked_uv = self.trk.last_keypoints_int()
                self._tracked_pixels = self._tracked_uv
            except Exception as e:
                self.get_logger().warn(f"Tracker step failed: {e}")

        # Update region tracker during REVEAL
        if self._reveal_active and self._reveal_ready:
            try:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                self.reveal_trk.step(rgb)

                self._cut_targets_uv = self._get_live_cut_targets_uv()
                self._last_cut_poly4_uv = self._poly4_from_targets(self._cut_targets_uv)

                tracks_np, vis_np = self.reveal_trk.tracks_numpy()  # [T,N,2], [T,N,?]
                t = tracks_np.shape[0] - 1
                if t >= 0 and self.reveal_trk.graph is not None:
                    self.reveal_trk.graph.update_from_prediction(
                        tracks_np, vis_np, t=t, threshold=self.reveal_cfg.pruner_threshold
                    )
            except Exception as e:
                self.get_logger().warn(f"REVEAL tracker step failed: {e}")
        try:
            if self._viz_enable:
                self._viz_counter += 1
                if (self._viz_counter % self._viz_every) == 0:
                    self._draw_live_overlay()
        except Exception as e:
            self.get_logger().warn(f"live overlay failed: {e}")

    def callback_disp(self, msg: Image):
        self.disp = self._to_cv(msg)

    def callback_Q(self, msg: Float32MultiArray):
        try:
            Q = np.array(msg.data, np.float32).reshape(4,4)
            # set scale=1.0 if your Q already yields meters
            self.cam = CameraModel(Q, use_negative_disp=True, scale=0.001)
        except Exception as e:
            self.get_logger().warn(f"Bad Q: {e}")

    def _to_cv(self, msg: Image):
        if not self.bridge:
            raise RuntimeError("cv_bridge not installed.")
        if msg.encoding in ("32FC1", "16UC1", "mono16"):
            return self.bridge.imgmsg_to_cv2(msg)
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _ready(self) -> bool:
        return self.disp is not None and self.cam is not None


    def _tick(self):
        if not self._ready():
            return
        
        if not self._did_pick_points:
            try:
                self.pick_n_points_on_left(n=4)
                self._did_pick_points = True
            except Exception as e:
                self.get_logger().warn(f"Point picking aborted: {e}")
                self._did_pick_points = True
            return


        if not hasattr(self, "_t_prev"):
            self._t_prev = self.get_clock().now()
        now = self.get_clock().now()
        dt = (now - self._t_prev).nanoseconds * 1e-9
        self._t_prev = now

        self.psm1.hold_jaw_goal(dt)
        self.psm2.hold_jaw_goal(dt)
        self.target_1 = self.picked_points_cam[0]
        self.psm1.offset = np.array([-0.06, -0.03, 0.0], dtype=np.float32) 
        

        if self._phase == "PSM1":
            # start once
            if not self._plan_started:
                self.target_1 = self.picked_points_cam[0]

                self.psm1.offset = np.array([-0.06, -0.03, 0.0], dtype=np.float32)  
                self.get_logger().info("Starting PSM1 plan…")
                self._tracking_active = True 
                self.planner.start(self.target_1)
                self._plan_started = True
                return

            st1 = self.planner.step()
            if st1.done:
                self.get_logger().info("PSM1 plan DONE ✔")
                time.sleep(2)
                # self._tracking_active = False 
                try:
                    self._tracked_pixels = self.trk.last_keypoints_int() if self._trk_ready else None
                except Exception as e:
                    self.get_logger().warn(f"get tracked UVs failed: {e}")
                    self._tracked_pixels = None
                if not self._tracked_pixels:
                    self._tracked_pixels = self.picked_pixels
                self.get_logger().info(f"Tracked UVs at handoff: {self._tracked_pixels}")

                self._cut_targets_cam = self._choose_cut_targets_cam()
                self._cut_targets_base = [ self.psm2.cam_pos_psm2(p) for p in self._cut_targets_cam ]

                # after you set:
                #   self._cut_targets_cam
                #   self._cut_targets_base
                # also compute UVs and directions:
                uv_all = self._tracked_pixels or self.picked_pixels or []
                self._cut_targets_uv = [uv_all[i] for i in (1,2,3) if i < len(uv_all)]
                self._cut_dirs_base = self._compute_cut_dirs_base_from_uv(self._cut_targets_uv)

                self._cut_total = len(self._cut_targets_base)
                self._cut_idx = 0
                if self._cut_total == 0:
                    self.get_logger().warn("No valid cutting targets; skipping PSM2.")
                    self._phase = "DONE"
                    return

                self._phase = "HANDOFF"
                self._handoff_t0 = self._now_s()

        elif self._phase == "HANDOFF":
            if self._now_s() - self._handoff_t0 >= self._handoff_wait_s:
                self.get_logger().info(f"Starting PSM2 cut sequence: {self._cut_total} targets")
                self._cutter_started = False
                self._phase = "PSM2"
                # self.reveal.start()
                # self._phase = "REVEAL_INIT"

        elif self._phase == "PSM2":
            if not self._cutter_started and self._cut_idx < self._cut_total:
                tgt = self._cut_targets_base[self._cut_idx]
                # choose a direction for this target
                if hasattr(self, "_cut_dirs_base") and len(self._cut_dirs_base) > self._cut_idx:
                    dir_y = self._cut_dirs_base[self._cut_idx]
                else:
                    dir_y = np.array([0.0, 1.0, 0.0], np.float32)  # safe default

                self.get_logger().info(f"PSM2 target {self._cut_idx+1}/{self._cut_total}")
                self.psm2_cut.start(target_pos_base=tgt, target_dir_y_base=dir_y)  # <-- pass direction here
                self._cutter_started = True
                return

            if self._cutter_started:
                st2 = self.psm2_cut.step()
                if st2.done:
                    self._cutter_started = False
                    self._cut_idx += 1
                    if self._cut_idx >= self._cut_total:
                        self.get_logger().info("PSM2: all cut targets complete ✔")
                        self.reveal.start()
                        self._phase = "REVEAL_INIT"


        elif self._phase == "REVEAL_INIT":

            self._cut_targets_uv = self._get_live_cut_targets_uv()
            self._last_cut_poly4_uv = self._poly4_from_targets(self._cut_targets_uv)

            W, H = self.left.shape[1], self.left.shape[0]
            endpoints = self._endpoints_from_points(self._cut_targets_uv, W, H)
            if self.left is None:
                self.get_logger().warn("REVEAL_INIT: no left image yet")
                return

            # endpoints = self._endpoints_from_points(self._cut_targets_uv, self.left.shape[1], self.left.shape[0])
            try:
                first_rgb = cv2.cvtColor(self.left, cv2.COLOR_BGR2RGB)
                self.reveal_trk.initialize(
                    first_frame_rgb=first_rgb,
                    mode="line_pairs",
                    points=endpoints,
                    n_pairs=100,
                    width_factor=2.0,
                )
                self._reveal_ready  = True
                self._reveal_active = True
                self._reveal_t0 = self._now_s()
                self.get_logger().info(f"REVEAL: initialized line_pairs with endpoints {endpoints}")
                self._phase = "REVEAL_RUN"
            except Exception as e:
                self.get_logger().warn(f"REVEAL_INIT failed: {e}")
                self._phase = "DONE"

        elif self._phase == "REVEAL_RUN":
            stR = self.reveal.step()
            if stR.done:
                # if (self._now_s() - self._reveal_t0) >= self._reveal_stop_after_s:
                self._finish_reveal()  # compute graph result + next suggestion
                self._reveal_active = False
                self._tracking_active = False 
                self._phase = "DONE"


        elif self._phase == "DONE":

            pass







    def _finish_reveal(self):
        # refresh from the very latest tracked targets
        self._cut_targets_uv = self._get_live_cut_targets_uv()
        self._last_cut_poly4_uv = self._poly4_from_targets(self._cut_targets_uv)

        try:
            tracks_np, vis_np = self.reveal_trk.tracks_numpy()
            if tracks_np.shape[0] == 0:
                self.get_logger().warn("REVEAL: no tracks to finalize")
                self.reveal_result = {"kept_edges": [], "next_poly4": None, "score": 0, "vis_path": None}
                return

            # Full graph pruning pass
            for t in range(tracks_np.shape[0]):
                self.reveal_trk.graph.update_from_prediction(
                    tracks_np, vis_np, t=t, threshold=self.reveal_cfg.pruner_threshold
                )

            # Use your last cut polyline (4-pt) to propose the next one
            if self._last_cut_poly4_uv is None:
                self._last_cut_poly4_uv = self._poly4_from_targets(self._cut_targets_uv)

            prop = self.reveal_trk.suggest_next_from_last_cut(
                last_poly4=self._last_cut_poly4_uv,
                visualize=True,            # returns a nice overlay
                tol_px=1.0,
            )

            vis_path = None
            if prop.get("vis_rgb") is not None:
                vis_path = "reveal_result_overlay.png"
                cv2.imwrite(vis_path, cv2.cvtColor(prop["vis_rgb"], cv2.COLOR_RGB2BGR))
                cv2.imshow("REVEAL result overlay", prop["vis_rgb"])

                self.get_logger().info(f"REVEAL: saved overlay → {vis_path}")

            self.reveal_result = {
                "kept_edges": self.reveal_trk.graph.kept_edges(),
                "next_poly4": prop.get("next_poly"),
                "score": prop.get("score", 0),
                "total_score": prop.get("total_score", 0.0),
                "vis_path": vis_path,
            }
            self.get_logger().info(f"REVEAL done. Next suggestion (poly4): {self.reveal_result['next_poly4']}, score={self.reveal_result['score']}")
        except Exception as e:
            self.get_logger().warn(f"_finish_reveal failed: {e}")
            self.reveal_result = {"kept_edges": [], "next_poly4": None, "score": 0, "vis_path": None}



    def _choose_cut_targets_cam(self) -> list:

        out = []

        candidates_uv = self._tracked_pixels or self.picked_pixels or []
        cam_pts_all = self._tracked_uvs_to_cam_points(candidates_uv)

        for idx in (1, 2, 3):
            if cam_pts_all.shape[0] > idx:
                out.append(cam_pts_all[idx])
            elif self.picked_points_cam is not None and len(self.picked_points_cam) > idx:
                out.append(self.picked_points_cam[idx])
            else:
                self.get_logger().warn(f"Cut target index {idx} missing; skipping.")
        return out


    def pick_n_points_on_left(self, n: int = 2, window_name: str = "Pick points"):
        if self.left is None:
            raise RuntimeError("Left image not available yet.")

        picked = self._ui_pick_points(n, window_name)
        
        self._process_picked_points(picked)

    def _ui_pick_points(self, n: int, window_name: str):
        img = self.left.copy()
        picked = []

        def on_mouse(evt, x, y, *_):
            if evt == cv2.EVENT_LBUTTONDOWN and len(picked) < n:
                picked.append((int(x), int(y)))

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)
        cv2.setMouseCallback(window_name, on_mouse)

        while True:
            vis = img.copy()
            for i, (u, v) in enumerate(picked, 1):
                cv2.circle(vis, (u, v), 6, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(vis, f"P{i}", (u+8, v-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, f"Click {n} points  |  r=reset  |  ESC=cancel", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,20,20), 2, cv2.LINE_AA)
            cv2.imshow(window_name, vis)
            
            k = cv2.waitKey(20) & 0xFF
            if k == 27:  # ESC
                cv2.destroyWindow(window_name)
                raise RuntimeError("User cancelled point picking.")
            if k == ord('r'):
                picked.clear()
            if len(picked) >= n:
                break

        cv2.destroyWindow(window_name)
        return picked

    def _process_picked_points(self, picked_pixels):
        self.picked_pixels = picked_pixels

        P, C = disparity_to_pointcloud(self.disp, self.cam.Q, use_negative_disp=False, max_points=4096, scale=0.001, return_colors=True, image=self.left)

        picked_3d = []
        for u, v in picked_pixels:
            if 0 <= u < self.disp.shape[1] and 0 <= v < self.disp.shape[0]:
                d = self.disp[v, u]
                if d > 0:
                    point_3d = self.cam.point_from_pixel(self.disp, u, v)
                    picked_3d.append(point_3d)
                else:
                    self.get_logger().warn(f"Invalid disparity at pixel ({u}, {v})")

        self.picked_points_cam = np.array(picked_3d) if picked_3d else None
        self.picked_points_psm1 = None

        self._visualize_with_picked_points(P, C)

        try:
            if self.picked_pixels and not self._trk_ready:
                first_rgb = cv2.cvtColor(self.left, cv2.COLOR_BGR2RGB)
                self.trk.set_keypoints_edges(keypoints_xy=self.picked_pixels, edges=[])
                self.trk.initialize(first_frame_rgb=first_rgb, mode="manual")
                self._trk_ready = True
                self.get_logger().info(f"LiteTrackingModule initialized with {len(self.picked_pixels)} keypoints.")
        except Exception as e:
            self.get_logger().warn(f"Tracker init failed: {e}")

    def _visualize_with_picked_points(self, P, C):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(P)
        if C is not None:
            point_cloud.colors = o3d.utility.Vector3dVector(C / 255.0)

        geometries = [point_cloud, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])]
        
        if self.picked_points_cam is not None and len(self.picked_points_cam) > 0:
            for point in self.picked_points_cam:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                sphere.translate(point)
                sphere.paint_uniform_color([1.0, 0.0, 0.0])
                geometries.append(sphere)
            window_name = f"Point Cloud with {len(self.picked_points_cam)} Picked Points"
        else:
            window_name = "Point Cloud"
            
        o3d.visualization.draw_geometries(geometries, window_name=window_name)

    def visualize_pointcloud(self, P, C):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(P)
        if C is not None:
            point_cloud.colors = o3d.utility.Vector3dVector(C / 255.0)

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([point_cloud, axis], window_name="Point Cloud with Axis")

    def _tracked_uvs_to_cam_points(self, uv_list):
        if uv_list is None or self.disp is None or self.cam is None:
            return np.zeros((0,3), dtype=np.float32)
        H, W = self.disp.shape[:2]
        out = []
        for (u,v) in uv_list:
            if 0 <= u < W and 0 <= v < H:
                d = float(self.disp[v, u])
                if np.isfinite(d) and d > 0:
                    out.append(self.cam.point_from_pixel(self.disp, u, v))
        return np.asarray(out, dtype=np.float32) if out else np.zeros((0,3), dtype=np.float32)


    def _visualize_handoff_pointcloud_tracked(self, max_points: int = 20000):
        if self.disp is None or self.cam is None or self.left is None:
            self.get_logger().warn("Cannot visualize: missing disp/cam/left.")
            return
        P, C = disparity_to_pointcloud(
            self.disp, self.cam.Q,
            use_negative_disp=False,
            max_points=max_points,
            scale=0.001,
            return_colors=True,
            image=self.left
        )
        uvs = self._tracked_pixels or self.picked_pixels or []
        Pk = self._tracked_uvs_to_cam_points(uvs)

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(P)
        if C is not None:
            cloud.colors = o3d.utility.Vector3dVector(C / 255.0)

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

        geoms = [cloud, axis]
        for p in Pk:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
            s.translate(p.astype(float))
            s.paint_uniform_color([0.0, 1.0, 0.0])
            geoms.append(s)

        if self.picked_points_cam is not None and len(self.picked_points_cam) > 0:
            for p in self.picked_points_cam:
                s = o3d.geometry.TriangleMesh.create_sphere(radius=0.0035)
                s.translate(p.astype(float))
                s.paint_uniform_color([1.0, 0.0, 0.0])
                geoms.append(s)

        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"Handoff: Point Cloud + {len(Pk)} tracked kpts"
        )

    def _visualize_tracked_on_image(
        self,
        window_name: str = "Tracked keypoints (overlay)",
        save_path: Optional[str] = None,
        draw_indices: bool = True,
        draw_drift_line: bool = True,
        r: int = 5,
    ):

        if self.left is None:
            self.get_logger().warn("No left image available for overlay.")
            return

        img = self.left.copy()

        tracked = self._tracked_pixels or []
        original = self.picked_pixels or []

        for (u, v) in original:
            cv2.circle(img, (u, v), r, (0, 0, 255), 2, cv2.LINE_AA)

        for i, (u, v) in enumerate(tracked):
            cv2.circle(img, (u, v), r, (0, 255, 0), -1, cv2.LINE_AA)
            if draw_indices:
                cv2.putText(img, str(i), (u + 6, v - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
            if draw_drift_line and i < len(original):
                u0, v0 = original[i]
                cv2.line(img, (u0, v0), (u, v), (0, 255, 255), 1, cv2.LINE_AA)

        if save_path:
            try:
                cv2.imwrite(save_path, img)
                self.get_logger().info(f"Tracked overlay saved → {save_path}")
            except Exception as e:
                self.get_logger().warn(f"Failed to save overlay: {e}")

        cv2.imshow(window_name, img)
        cv2.waitKey(1)

    def _final_three_cut_uvs(self) -> list[tuple[int,int]]:
        """
        Return three cutting targets as UVs = indices [1,2,3].
        Prefer latest tracked UVs; fall back to originally picked pixels.
        """
        src = self.trk.last_keypoints_int() if self._trk_ready else self.picked_pixels
        if not src or len(src) < 4:
            return []
        return [src[1], src[2], src[3]]
    

    def _visualize_points_on_left(
        self,
        points_uv: list[tuple[int,int]],
        *,
        window_name: str = "Final dissection targets",
        save_path: Optional[str] = "final_dissection_targets.png",
        draw_indices: bool = True,
        radius: int = 6
    ):
        if self.left is None or not points_uv:
            self.get_logger().warn("No left image or points to visualize.")
            return
        img = self.left.copy()

        # magenta for final targets
        for i, (u, v) in enumerate(points_uv):
            cv2.circle(img, (u, v), radius, (255, 0, 255), -1, cv2.LINE_AA)
            if draw_indices:
                cv2.putText(img, f"{i+1}", (u+8, v-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        # also show the original picked points in red (optional)
        if self.picked_pixels:
            for (u0, v0) in self.picked_pixels:
                cv2.circle(img, (u0, v0), radius, (0, 0, 255), 2, cv2.LINE_AA)

        if save_path:
            try:
                cv2.imwrite(save_path, img)
                self.get_logger().info(f"Saved final overlay → {save_path}")
            except Exception as e:
                self.get_logger().warn(f"Failed to save overlay: {e}")

        cv2.imshow(window_name, img)
        cv2.waitKey(1)


    def _poly4_from_targets(self, uvs3: list[tuple[int,int]]):
        """
        Turn ~3 target points into a straight 4-pt polyline P1..P4
        along their principal direction. Works even if we only have 2 points.
        """
        if not uvs3:
            return None
        pts = np.array(uvs3, float)
        # principal direction
        c = pts.mean(axis=0)
        X = pts - c
        if len(pts) >= 2:
            C = (X.T @ X) / max(1, len(X)-1)
            w, V = np.linalg.eig(C)
            u = V[:, np.argmax(w)]
        else:
            u = np.array([1.0, 0.0])

        # project to get min/max endpoints
        t = X @ u
        tmin, tmax = t.min(), t.max()
        p1 = c + u * tmin
        p4 = c + u * tmax
        # 4 points equally spaced
        p2 = p1 + (p4 - p1) / 3.0
        p3 = p1 + 2.0 * (p4 - p1) / 3.0
        poly4 = [tuple(map(int, np.round(p))) for p in (p1, p2, p3, p4)]
        return poly4

    def _endpoints_from_points(self, pts_uv: list[tuple[int,int]], W: int, H: int, margin_px: int = 30):
        """
        Two endpoints spanning the 3 targets, expanded by a small margin along their principal axis.
        Used to seed line_pairs.
        """
        if not pts_uv:
            # fallback: image center horizontal
            return [(int(W*0.25), int(H*0.5)), (int(W*0.75), int(H*0.5))]
        pts = np.array(pts_uv, float)
        c = pts.mean(axis=0)
        X = pts - c
        if len(pts) >= 2:
            C = (X.T @ X) / max(1, len(X)-1)
            w, V = np.linalg.eig(C)
            u = V[:, np.argmax(w)]
        else:
            u = np.array([1.0, 0.0])

        t = X @ u
        tmin, tmax = t.min() - margin_px, t.max() + margin_px
        p1 = c + u * tmin
        p2 = c + u * tmax
        p1 = np.array([np.clip(p1[0], 0, W-1), np.clip(p1[1], 0, H-1)], int)
        p2 = np.array([np.clip(p2[0], 0, W-1), np.clip(p2[1], 0, H-1)], int)
        return [tuple(p1), tuple(p2)]



    def _draw_live_overlay(self):
        """Show a live window with tracked keypoints, graph edges (if any), dissection targets,
        and an arrowed direction for each target (last reuses previous direction)."""
        if self.left is None:
            return

        # --- tiny helpers (local to keep this function self-contained) ---
        def _clip_xy(x: float, y: float, W: int, H: int) -> tuple[int, int]:
            return int(np.clip(round(x), 0, W - 1)), int(np.clip(round(y), 0, H - 1))

        def _compute_target_dirs(uvs: list[tuple[int, int]]) -> list[tuple[float, float]]:
            """direction[i] = unit(uvs[i+1] - uvs[i]); last reuses the previous (or [1,0] if only one)"""
            n = len(uvs)
            if n == 0:
                return []
            dirs = []
            for i in range(n - 1):
                p = np.array(uvs[i], dtype=float)
                q = np.array(uvs[i + 1], dtype=float)
                v = q - p
                nrm = float(np.linalg.norm(v))
                if nrm < 1e-6:
                    dirs.append((1.0, 0.0))
                else:
                    dirs.append((v[0] / nrm, v[1] / nrm))
            dirs.append(dirs[-1] if n > 1 else (1.0, 0.0))
            return dirs
        # -----------------------------------------------------------------

        img = self.left.copy()  # BGR frame
        H, W = img.shape[:2]

        # ---- draw original clicked points (red, hollow) ----
        for (u, v) in (self.picked_pixels or []):
            cv2.circle(img, (u, v), 5, (0, 0, 255), 2, cv2.LINE_AA)

        # ---- choose which tracker to visualize ----
        kpts_np = None          # Nx2 int
        edges = None            # list[(i,j)]

        # REVEAL region tracker (has graph)
        if getattr(self, "_reveal_active", False) and getattr(self, "_reveal_ready", False):
            tracks_np, _ = self.reveal_trk.tracks_numpy()  # [T,N,2]
            if tracks_np.shape[0] > 0:
                kpts_np = tracks_np[-1].astype(int)
                edges = self.reveal_trk.graph.kept_edges() if self.reveal_trk.graph is not None else []
        # Otherwise, show the simple keypoint tracker (no graph)
        elif getattr(self, "_trk_ready", False):
            k = self.trk.last_keypoints_int() or []
            if k:
                kpts_np = np.asarray(k, dtype=int)
            edges = None

        # ---- draw graph edges (semi-transparent black) ----
        if edges and kpts_np is not None and len(kpts_np) > 0:
            overlay = img.copy()
            for (i, j) in edges:
                if 0 <= i < len(kpts_np) and 0 <= j < len(kpts_np):
                    x1, y1 = kpts_np[i]
                    x2, y2 = kpts_np[j]
                    if 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                        cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 0), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, dst=img)

        # ---- draw tracked keypoints (green, filled) ----
        if kpts_np is not None and len(kpts_np) > 0:
            for (x, y) in kpts_np:
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(img, (x, y), 3, (0, 255, 0), -1, cv2.LINE_AA)

        # ---- draw dissection targets (yellow, thick) + direction arrows ----
        targets = (getattr(self, "_cut_targets_uv", None) or [])
        for (u, v) in targets:
            cv2.circle(img, (u, v), 6, (0, 255, 255), 2, cv2.LINE_AA)
        if targets:
            dirs = _compute_target_dirs(targets)
            arrow_len = max(20, int(0.05 * max(W, H)))  # ≈5% of image diag
            for (u, v), (dx, dy) in zip(targets, dirs):
                x0, y0 = _clip_xy(u, v, W, H)
                x1, y1 = _clip_xy(u + arrow_len * dx, v + arrow_len * dy, W, H)
                cv2.arrowedLine(
                    img, (x0, y0), (x1, y1),
                    (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.25
                )

        # ---- draw last cut (cyan) ----
        if getattr(self, "_last_cut_poly4_uv", None):
            pts = np.array(self._last_cut_poly4_uv, dtype=int).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], isClosed=False, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # ---- draw proposed next cut (magenta) if available (after REVEAL finalize) ----
        if getattr(self, "reveal_result", None) and self.reveal_result.get("next_poly4"):
            pts2 = np.array(self.reveal_result["next_poly4"], dtype=int).reshape(-1, 1, 2)
            cv2.polylines(img, [pts2], isClosed=False, color=(255, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        # ---- show window ----
        if not self._viz_window_created:
            cv2.namedWindow(self._viz_win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._viz_win, 960, 720)
            self._viz_window_created = True
        cv2.imshow(self._viz_win, img)
        cv2.waitKey(1)

    def _compute_cut_dirs_base_from_uv(self, uvs: list[tuple[int,int]]) -> list[np.ndarray]:
        """
        Build a forward unit direction per dissection target in PSM2 base:
        dir[i] = unit( T_{i+1} - T_i ), last reuses previous.
        If fewer than 2 valid 3D points, returns [].
        """
        if self.disp is None or self.cam is None or len(uvs) == 0:
            return []
        # UV -> CAM 3D (skip invalid)
        pts_cam = []
        H, W = self.disp.shape[:2]
        for (u, v) in uvs:
            if 0 <= u < W and 0 <= v < H:
                d = float(self.disp[v, u])
                if np.isfinite(d) and d > 0:
                    pts_cam.append(self.cam.point_from_pixel(self.disp, u, v))
        if len(pts_cam) < 1:
            return []

        # CAM -> PSM2 BASE
        pts_base = [ self.psm2.cam_pos_psm2(np.asarray(p, np.float32)) for p in pts_cam ]
        pts_base = [ np.asarray(p, np.float32).reshape(3) for p in pts_base ]

        # directions
        dirs = []
        def _unit(v):
            n = float(np.linalg.norm(v))
            return (v / (n + 1e-8)).astype(np.float32)
        for i in range(len(pts_base) - 1):
            dirs.append(_unit(pts_base[i+1] - pts_base[i]))
        if len(dirs) == 0:
            return []
        dirs.append(dirs[-1].copy())  # last = previous
        return dirs[:len(pts_base)]   # clamp to number of targets we actually have


    def _get_live_cut_targets_uv(self) -> list[tuple[int,int]]:
        """
        Return the *current* 3 dissection targets (indices 1,2,3) from the keypoint tracker.
        Fallbacks to last snapshot and finally to clicked pixels.
        """
        uv_all = None
        if self._trk_ready:
            try:
                uv_all = self.trk.last_keypoints_int()
            except Exception:
                uv_all = None
        if not uv_all:
            uv_all = self._tracked_pixels or self.picked_pixels or []
        return [uv_all[i] for i in (1, 2, 3) if i < len(uv_all)]

    def _clip_xy_to_image(self, x: float, y: float, W: int, H: int) -> tuple[int,int]:
        return int(max(0, min(W-1, round(x)))), int(max(0, min(H-1, round(y))))

    def _compute_target_directions(self, targets_uv: list[tuple[int,int]]) -> list[tuple[float,float]]:
        """
        For each target i, direction = unit(next_i - this_i).
        For the last target, reuse the previous direction (or [1,0] if only one).
        Returns a list of unit vectors (dx, dy) aligned with targets_uv.
        """
        n = len(targets_uv)
        if n == 0:
            return []
        dirs: list[tuple[float,float]] = []
        # forward differences
        for i in range(n - 1):
            p = np.array(targets_uv[i],   dtype=float)
            q = np.array(targets_uv[i+1], dtype=float)
            v = q - p
            norm = float(np.linalg.norm(v))
            if norm < 1e-6:
                d = (1.0, 0.0)  # fallback if two points coincide
            else:
                d = (v[0]/norm, v[1]/norm)
            dirs.append(d)
        # tail rule
        if n >= 2:
            dirs.append(dirs[-1])
        else:
            dirs.append((1.0, 0.0))
        return dirs

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
