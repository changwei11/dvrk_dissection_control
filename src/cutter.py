# psm2_cut_planner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

@dataclass
class PlanStatus:
    state: str
    done: bool

class PSM2CutPlanner:
    """
    Stages (all in PSM2 base frame):
      MOVE_CMD            : Go to target pose (pos + quat). (move_cp or servo)
      OPEN_DEEPEN_SERVO   : Open jaw and push deeper with servo steps (fixed Hz, duration).
      CLOSE_SERVO         : Close jaw in phases (e.g., 0.0 then -0.2) while holding pose.
      MOVE_AWAY_SERVO     : Step the EE away by a fixed delta per tick (servo).
      DONE
    """

    def __init__(
        self,
        ctrl,
        *,
        # MOVE
        move_cmd_s: float = 2.0,
        move_use_servo: bool = False,
        move_hz: float = 100.0,
        # OPEN + DEEPEN (servo)
        deep_cmd_s: float = 2.0,
        deep_hz: float = 100.0,
        deep_delta: Tuple[float, float, float] = (-0.0005, 0.0, -0.002),
        jaw_open_value: float = 0.5,
        close_cmd_s: float = 1.0,
        close_hz: float = 100.0,
        close_targets: Tuple[float, float] = (0.0, -0.2),  # stream 1st half to 0.0, 2nd half to -0.2
        # MOVE AWAY (servo)
        away_cmd_s: float = 2.0,
        away_hz: float = 100.0,
        away_delta_base: Tuple[float, float, float] = (0.001, 0.0, +0.002),  # per tick in BASE (m/tick)
        jaw_hold_close: float = -0.2,  # rad to keep jaw closed while moving away
        # Common
        resend_period_s: float = 0.3,
    ):
        self.ctrl = ctrl

        # MOVE
        self.move_cmd_s = float(move_cmd_s)
        self.move_use_servo = bool(move_use_servo)
        self.move_hz = float(move_hz)

        self.deep_cmd_s = float(deep_cmd_s)
        self.deep_hz = float(deep_hz)
        self.deep_delta = np.asarray(deep_delta, np.float32)
        self.jaw_open_value = float(jaw_open_value)

        self.close_cmd_s = float(close_cmd_s)
        self.close_hz = float(close_hz)
        self.close_targets = tuple(close_targets)

        # MOVE AWAY
        self.away_cmd_s = float(away_cmd_s)
        self.away_hz = float(away_hz)
        self.away_delta_base = np.asarray(away_delta_base, np.float32)
        self.jaw_hold_close = float(jaw_hold_close)

        self.resend_period = float(resend_period_s)

        # Internal
        self._state = "IDLE"
        self._t_enter = time.time()
        self._t_last_send = 0.0

        # Targets / pose
        self._target_pos_base: Optional[np.ndarray] = None
        self._target_quat: Optional[np.ndarray] = None

        # MOVE (servo) bookkeeping
        self._move_dt = 1.0 / max(1.0, self.move_hz)
        self._move_t_last = 0.0

        # DEEPEN bookkeeping
        self._deep_steps = 0
        self._deep_i = 0
        self._deep_dt = 1.0 / max(1.0, self.deep_hz)
        self._deep_t_last = 0.0
        self._deep_pos_base: Optional[np.ndarray] = None
        self._deep_quat_hold: Optional[tuple] = None
        self._deep_delta_base_step: Optional[np.ndarray] = None

        # CLOSE bookkeeping
        self._close_steps = 0
        self._close_i = 0
        self._close_dt = 1.0 / max(1.0, self.close_hz)
        self._close_t_last = 0.0
        self._close_pos_hold: Optional[np.ndarray] = None
        self._close_quat_hold: Optional[tuple] = None
        self._close_phase_cut = 0  # 0 or 1

        # AWAY bookkeeping
        self._away_steps = 0
        self._away_i = 0
        self._away_dt = 1.0 / max(1.0, self.away_hz)
        self._away_t_last = 0.0
        self._away_pos_base: Optional[np.ndarray] = None
        self._away_quat_hold: Optional[tuple] = None

    # ---------- Public API ----------
    # def start(self, target_pos_base: np.ndarray, target_quat_xyzw: Optional[np.ndarray] = None):
    #     """Start a full cut plan from current pose to target, then open+deepen, close, move away."""
    #     self._target_pos_base = np.asarray(target_pos_base, np.float32).reshape(3)
    #     # self._target_quat = np.arrray([0.88, -0.08, -0.45, 0.037])
    #     # self._target_quat = self._normalize(np.array([0.97, 0.04, -0.17, 0.14]))
    #     self._target_quat = self._normalize(np.array([0.83, -0.32, -0.32, 0.3]))
    #     # self._target_quat = self._normalize(np.array([0.97, -0.14, -0.18, 0.024]))


    #     self._goto("MOVE_CMD")

    def start(
        self,
        target_pos_base: np.ndarray,
        target_dir_y_base: Optional[np.ndarray] = None,
        target_quat_xyzw: Optional[np.ndarray] = None,
    ):
        """
        Start a full cut plan.
        If target_quat_xyzw is given, it wins. Otherwise if target_dir_y_base is given,
        orientation is computed with Y -> dir and Z ~ (-1,0,-2).
        """
        self._target_pos_base = np.asarray(target_pos_base, np.float32).reshape(3)

        if target_quat_xyzw is not None:
            self._target_quat = self._normalize(np.asarray(target_quat_xyzw, np.float32).reshape(4))
        elif target_dir_y_base is not None:
            self._target_quat = self._normalize(self.quat_from_yz(target_dir_y_base))
        else:
            # last-resort fallback (keep whatever the robot currently has)
            self._target_quat = self._normalize(np.asarray(self.ctrl.current_orientation(), np.float32).reshape(4))

        self._goto("MOVE_CMD")



    def is_active(self) -> bool:
        return self._state not in ("IDLE", "DONE")

    def is_done(self) -> bool:
        return self._state == "DONE"

    # ---------- State machinery ----------
    def _goto(self, new_state: str):
        self._state = new_state
        self._t_enter = time.time()
        self._t_last_send = 0.0

        if new_state == "MOVE_CMD":
            self._move_t_last = 0.0
            # move_offset = np.array([-0.09, 0.015, 0.012], dtype=np.float32)
            move_offset = np.array([-0.095, 0.013, 0.012], dtype=np.float32)
            self._target_pos_base = self._target_pos_base + move_offset

            current_pos = np.asarray(self.ctrl.current_pos(), np.float32)

            print("Target pos (base):", self._target_pos_base)

        elif new_state == "OPEN_DEEPEN_SERVO":
            # Start from current measured pose; keep orientation fixed during push
            p_base = np.asarray(self.ctrl.current_pos(), np.float32)
            q_xyzw = np.asarray(self.ctrl.current_orientation(), np.float32)
            # R_base_tool = R.from_quat(q_xyzw).as_matrix().astype(np.float32)
            # delta_base_per_tick = R_base_tool @ self.deep_delta_local  # map local -> base

            self._deep_pos_base = p_base.copy()
            self._deep_quat_hold = tuple(q_xyzw.tolist())
            # self._deep_delta_base_step = delta_base_per_tick
            self._deep_steps = max(1, int(round(self.deep_cmd_s * self.deep_hz)))
            self._deep_i = 0
            self._deep_dt = 1.0 / max(1.0, self.deep_hz)
            self._deep_t_last = 0.0

        elif new_state == "CLOSE_SERVO":
            self._close_pos_hold = np.asarray(self.ctrl.current_pos(), np.float32)
            self._close_quat_hold = tuple(np.asarray(self.ctrl.current_orientation(), np.float32).tolist())
            self._close_steps = max(2, int(round(self.close_cmd_s * self.close_hz)))
            self._close_i = 0
            self._close_phase_cut = 0
            self._close_dt = 1.0 / max(1.0, self.close_hz)
            self._close_t_last = 0.0

        elif new_state == "MOVE_AWAY_SERVO":
            self._away_pos_base = np.asarray(self.ctrl.current_pos(), np.float32)
            self._away_quat_hold = tuple(np.asarray(self.ctrl.current_orientation(), np.float32).tolist())
            self._away_steps = max(1, int(round(self.away_cmd_s * self.away_hz)))
            self._away_i = 0
            self._away_dt = 1.0 / max(1.0, self.away_hz)
            self._away_t_last = 0.0

    def _elapsed(self) -> float:
        return time.time() - self._t_enter

    @staticmethod
    def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        v = np.asarray(v, np.float32); n = float(np.linalg.norm(v)); return v / (n + eps)

    def _quat_x_up_z_heading(self, cur_base: np.ndarray, tgt_base: np.ndarray) -> np.ndarray:
        x_axis = np.array([0.0, 0.0, 1.0], np.float32)                # tool +X = world up
        z_dir = tgt_base - cur_base; z_dir[2] = 0.0                   # planar heading
        z_axis = self._normalize(z_dir if np.linalg.norm(z_dir) > 1e-6 else np.array([1.0, 0.0, 0.0], np.float32))
        y_axis = self._normalize(np.cross(z_axis, x_axis))            # right-handed
        R_world_tool = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)
        q_xyzw = R.from_matrix(R_world_tool).as_quat().astype(np.float32)
        return q_xyzw / (np.linalg.norm(q_xyzw) + 1e-9)

    def step(self) -> PlanStatus:
        now = time.time()

        if self._state == "IDLE":
            return PlanStatus(self._state, False)

        if self._state == "MOVE_CMD":

            if (now - self._t_last_send) >= self.resend_period:
                self.ctrl.send_pose_psm2(self._target_pos_base, quat=tuple(self._target_quat))
                print("Target pos (base):", self._target_pos_base)
                self._t_last_send = now

            if self._elapsed() >= self.move_cmd_s:
                self._goto("OPEN_DEEPEN_SERVO")
                # self._goto("DONE")

        # 2) OPEN and DEEPEN (servo)
        elif self._state == "OPEN_DEEPEN_SERVO":
            if (self._deep_t_last == 0.0) or (now - self._deep_t_last >= self._deep_dt):
                self._deep_t_last = now
                self._deep_i += 1
                self.ctrl.set_jaw_servo(0.5)
                current_pose = np.asarray(self.ctrl.current_pos(), np.float32)
                deep_pose = current_pose + self.deep_delta if self.deep_delta is not None else current_pose
                self.ctrl.send_servo_pose(deep_pose)

            if (self._deep_i >= self._deep_steps) or (self._elapsed() >= self.deep_cmd_s):
                self._goto("CLOSE_SERVO")

        # 3) CLOSE (servo) in phases: first -> close_targets[0], then -> close_targets[1]
        elif self._state == "CLOSE_SERVO":
            if (self._close_t_last == 0.0) or (now - self._close_t_last >= self._close_dt):
                self.ctrl.set_jaw_servo(-0.2)

            if (self._close_i >= self._close_steps) or (self._elapsed() >= self.close_cmd_s):
                self._goto("MOVE_AWAY_SERVO")

        # 4) MOVE AWAY (servo)
        elif self._state == "MOVE_AWAY_SERVO":
            if (self._away_t_last == 0.0) or (now - self._away_t_last >= self._away_dt):
                self._away_t_last = now
                self._away_i += 1

                # jaw: keep tight
                self.ctrl.set_jaw_servo(self.jaw_hold_close)

                current_pose = np.asarray(self.ctrl.current_pos(), np.float32)
                away_pose = current_pose + self.away_delta_base if self.away_delta_base is not None else current_pose
                self.ctrl.send_servo_pose(away_pose)

            if (self._away_i >= self._away_steps) or (self._elapsed() >= self.away_cmd_s):
                self._goto("DONE")

        return PlanStatus(self._state, self._state == "DONE")


    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, np.float32).reshape(-1)
        n = float(np.linalg.norm(v))
        return v / (n + 1e-8)

    def quat_from_yz(
        self,
        y_dir_base: np.ndarray,
        z_toward_base: np.ndarray = np.array([-1.0, 0.0, -2.0], np.float32),
    ) -> np.ndarray:
        """
        Build a right-handed frame with:
        tool Y-axis -> y_dir_base (exact)
        tool Z-axis -> as close as possible to z_toward_base but orthogonal to Y
        tool X-axis -> Y x Z
        Returns quaternion [x,y,z,w] in base.
        """
        y = self._unit(y_dir_base)
        z_pref = self._unit(z_toward_base)

        # x = y × z_pref; if nearly collinear, choose a fallback z_pref
        x = np.cross(y, z_pref).astype(np.float32)
        if float(np.linalg.norm(x)) < 1e-6:
            # pick a fallback that isn't collinear with y
            z_pref = np.array([0.0, 0.0, -1.0], np.float32)
            x = np.cross(y, z_pref).astype(np.float32)
            if float(np.linalg.norm(x)) < 1e-6:
                z_pref = np.array([1.0, 0.0, 0.0], np.float32)
                x = np.cross(y, z_pref).astype(np.float32)

        x = self._unit(x)
        z = self._unit(np.cross(x, y))  # orthonormalize; z close to z_pref, orthogonal to y

        R_base_tool = np.column_stack([x, y, z]).astype(np.float32)  # columns are tool axes
        q_xyzw = R.from_matrix(R_base_tool).as_quat().astype(np.float32)
        return q_xyzw / (np.linalg.norm(q_xyzw) + 1e-9)
