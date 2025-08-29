from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

@dataclass
class PlanStatus:
    state: str
    pos_err: Optional[float]
    jaw_rad: Optional[float]
    done: bool

class PSM1PickPlanner:

    def __init__(
        self,
        ctrl,
        *,
        move_cmd_s: float = 3.0,
        open_cmd_s: float = 1.0,
        open_wait_s: float = 1.0,
        deep_cmd_s: float = 2.0,
        close_cmd_s: float = 2.0,
        resend_period_s: float = 0.3,
        retract_cmd_s: float = 3.0,
        retract_hz: float = 20.0,
        retract_dist_m: float = 0.0005,
        open_rad: float = 0.5,
        deepen_delta_base = (0.0, 0.0, -0.002),
        rotate_deg: float = 80.0, 
        deep_hz: float = 10.0,
    ):
        self.ctrl = ctrl
        self.move_cmd_s    = float(move_cmd_s)
        self.open_cmd_s    = float(open_cmd_s)
        self.open_wait_s   = float(open_wait_s)
        self.deep_cmd_s    = float(deep_cmd_s)
        self.close_cmd_s   = float(close_cmd_s)
        self.retract_cmd_s = float(retract_cmd_s)
        self.retract_hz    = float(retract_hz)
        self.retract_dist_m = float(retract_dist_m)
        self.resend_period = float(resend_period_s)
        self.open_rad      = float(open_rad)
        self.deepen_delta_base = np.asarray(deepen_delta_base, dtype=np.float32)
        self.rotate_deg    = float(rotate_deg)
        self._state = "IDLE"
        self._target_cam: Optional[np.ndarray] = None
        self._target_psm1: Optional[np.ndarray] = None
        self._deep_target_cam: Optional[np.ndarray] = None   # NEW
        self._t_enter = time.time()
        self._t_last_send = 0.0
        self.quat_tool: Optional[np.ndarray] = None
        self._retract_psm1: Optional[np.ndarray] = None   # goal pos in CAMERA frame
        self._retract_quat: Optional[tuple] = None 

        self._retract_target_base: Optional[np.ndarray] = None
        self._ret_steps: int = 0
        self._ret_i: int = 0
        self._ret_dt: float = 1.0 / max(1.0, self.retract_hz)
        self._ret_t_last: float = 0.0
        self._ret_start_base: Optional[np.ndarray] = None
        self._ret_delta_base_step: Optional[np.ndarray] = None
        self._ret_quat_hold: Optional[tuple] = None


    def start(self, target_cam_xyz: np.ndarray):
        self._target_cam = np.asarray(target_cam_xyz, dtype=np.float32).reshape(3)
        # Orientation: X->up, Z->planar heading toward target (in base)
        self.quat_tool = self.ctrl.quat_x_up_z_heading(self._target_cam) if self._target_cam is not None else None

        # For error display and for deepening target
        try:
            self._target_psm1 = self.ctrl.cam_pos_psm1(self._target_cam)
        except Exception:
            self._target_psm1 = None


        self._goto("MOVE_CMD")


    def is_active(self) -> bool:
        return self._state not in ("IDLE", "DONE")

    def is_done(self) -> bool:
        return self._state == "DONE"

    def _goto(self, new_state: str):
        self._state = new_state
        self._t_enter = time.time()
        self._t_last_send = 0.0
        if new_state == "DEEPEN_CMD":
            # self._quat_rot = self._rotate_about_local_y(self.quat_tool, np.deg2rad(self.rotate_deg))
            self.deepen_delta_base = np.array([0.0, 0.0, -0.002])
            self.deepen_delta_base = np.asarray(self.deepen_delta_base, dtype=np.float32)
        elif new_state == "RETRACT_CMD":
            p_base = np.asarray(self.ctrl.current_pos(), np.float32)
            q_xyzw = np.asarray(self.ctrl.current_orientation(), np.float32)
            R_base_tool = R.from_quat(q_xyzw).as_matrix().astype(np.float32)

            R_base_tool = R.from_quat(q_xyzw).as_matrix().astype(np.float32)
            default_dist = 0.005
            goal_base = p_base + R_base_tool @ np.array([0,0,-default_dist], np.float32)


            retract_offset = np.array([-0.02, -0.01, 0.015], dtype=np.float32)
            goal_base += retract_offset 
            self._retract_psm1 = goal_base.astype(np.float32)


            N = max(1, int(round(self.retract_cmd_s * self.retract_hz)))
            self._ret_steps = N
            self._ret_i = 0
            self._ret_dt = 1.0 / max(1.0, self.retract_hz)
            self._ret_t_last = 0.0 
            self._ret_start_base = p_base
            self._ret_delta_base_step = (goal_base - p_base) / float(N)

            # delta_base_step reduce
            self._ret_delta_base_step = self._ret_delta_base_step.astype(np.float32) / 2.0


            # map to camera frame for send_pose()
            if not hasattr(self.ctrl, "psm1_pos_to_cam"):
                raise RuntimeError("Controller needs psm1_pos_to_cam() for retract stage.")

            self._retract_quat = tuple(q_xyzw.tolist())

    def _elapsed(self) -> float:
        return time.time() - self._t_enter

    def _pos_err(self) -> Optional[float]:
        if self._target_psm1 is None:
            return None
        cur = np.asarray(self.ctrl.current_pos(), dtype=np.float32)
        return float(np.linalg.norm(cur - self._target_psm1))

    def step(self) -> PlanStatus:
        now = time.time()
        jaw = self.ctrl.current_jaw()
        err = self._pos_err()

        if self._state == "IDLE":
            return PlanStatus(self._state, err, jaw, False)

        if self._state == "MOVE_CMD":
            if (now - self._t_last_send) >= self.resend_period:
                self.ctrl.send_pose(self._target_cam, quat=None if self.quat_tool is None else tuple(self.quat_tool))
                self._t_last_send = now
            if self._elapsed() >= self.move_cmd_s:
                self._goto("DEEPEN_CMD")

        elif self._state == "DEEPEN_CMD":
            if (now - self._t_last_send) >= self.resend_period:
                current_base = np.asarray(self.ctrl.current_pos(), dtype=np.float32)
                next_pos = current_base + self.deepen_delta_base
                self.ctrl.set_jaw_servo(0.5)  # hold jaw open
                self.ctrl.send_servo_pose(next_pos)
                self._t_last_send = now
            if self._elapsed() >= self.deep_cmd_s:
                self._goto("CLOSE_CMD")

        elif self._state == "CLOSE_CMD":
            if (now - self._t_last_send) >= self.resend_period:
                self.ctrl.close_jaw()
                self._t_last_send = now
            if self._elapsed() >= self.close_cmd_s:
                self._goto("RETRACT_CMD")


        elif self._state == "RETRACT_CMD":

            if self._ret_start_base is not None and self._ret_delta_base_step is not None:
                if (self._ret_t_last == 0.0) or (now - self._ret_t_last >= self._ret_dt):
                    self._ret_t_last = now
                    self.ctrl.set_jaw_servo(-1.0)
                    self._ret_i += 1

                    pos_base = self._ret_start_base + self._ret_i * self._ret_delta_base_step
                    # print("pos_base:", pos_base)    
                    # print("retract_target_base:", self._retract_psm1)
                    # print("error:", np.linalg.norm(pos_base - self._retract_psm1))                
                    self.ctrl.send_servo_pose(pos_base)

            # stop either by steps or by time
            if (self._ret_i >= self._ret_steps) or (self._elapsed() >= self.retract_cmd_s):
                self._goto("DONE")

        return PlanStatus(self._state, err, jaw, self.is_done())
