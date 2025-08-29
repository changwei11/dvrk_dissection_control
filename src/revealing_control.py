# reveal_planner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import time
from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R

@dataclass
class PlanStatus:
    state: str
    pos_err: Optional[float]
    jaw_rad: Optional[float]
    done: bool

class PSM1RevealPlanner:
    """
    Retract along tool -Z in the PSM1 base frame for a fixed duration while
    continuously commanding a closed jaw via servo.
    """
    def __init__(
        self,
        ctrl,
        *,
        retract_cmd_s: float = 1.0,      # total time of retract
        retract_hz: float = 20.0,        # stream rate for servo cp
        retract_dist_m: float = 0.020,   # distance to move along tool -Z (positive number)
        resend_period_s: float = 0.2,    # not critical; used for safety/info
        jaw_hold_close: float = -1.0,    # servo value to keep closing
        offset_base = (0.0, 0.0, 0.0),   # extra base-frame offset if you need it
        keep_orientation: bool = True,   # hold current orientation during retract
    ):
        self.ctrl = ctrl
        self.retract_cmd_s   = float(retract_cmd_s)
        self.retract_hz      = float(retract_hz)
        self.retract_dist_m  = float(retract_dist_m)
        self.resend_period_s = float(resend_period_s)
        self.jaw_hold_close  = float(jaw_hold_close)
        self.offset_base     = np.asarray(offset_base, np.float32)
        self.keep_orientation= bool(keep_orientation)
        self.deep_delta: Tuple[float, float, float] = (-0.001, -0.0002, 0.0005),
        self._state = "IDLE"
        self._t_enter = 0.0
        self._t_last_step = 0.0
        self._dt = 1.0 / max(1.0, self.retract_hz)

        # step interpolation
        self._start_p: Optional[np.ndarray] = None
        self._goal_p:  Optional[np.ndarray] = None
        self._N: int = 0
        self._i: int = 0
        self._delta: Optional[np.ndarray] = None
        self._hold_quat_xyzw: Optional[tuple] = None

    # --- API ---
    def start(self):
        """Compute base-frame retract path from current pose."""
        p_base = np.asarray(self.ctrl.current_pos(), np.float32)
        q_xyzw = np.asarray(self.ctrl.current_orientation(), np.float32)  # (x,y,z,w)
        R_base_tool = R.from_quat(q_xyzw).as_matrix().astype(np.float32)

        # move along tool -Z by retract_dist_m, then add optional base offset
        goal = p_base + R_base_tool @ np.array([0.0, 0.0, -self.retract_dist_m], np.float32)
        goal = goal + self.offset_base

        self._start_p = p_base
        self._goal_p  = goal
        self._N       = max(1, int(round(self.retract_cmd_s * self.retract_hz)))
        self._i       = 0
        self._delta   = (goal - p_base) / float(self._N)
        self._hold_quat_xyzw = tuple(q_xyzw.tolist()) if self.keep_orientation else None

        self._state = "RETRACT"
        self._t_enter = time.time()
        self._t_last_step = 0.0  # force immediate first send

    def is_active(self) -> bool:
        return self._state == "RETRACT"

    def is_done(self) -> bool:
        return self._state == "DONE"

    def step(self) -> PlanStatus:
        if self._state == "IDLE":
            return PlanStatus(self._state, None, self.ctrl.current_jaw(), False)

        now = time.time()

        # stream multiple sub-steps if your ROS timer is slower than retract_hz
        while self._state == "RETRACT" and (
            self._t_last_step == 0.0 or (now - self._t_last_step) >= self._dt
        ):
            self._t_last_step = now
            self._i += 1

            # keep jaw squeezing
            self.ctrl.set_jaw_servo(self.jaw_hold_close)


            current_pose = np.asarray(self.ctrl.current_pos(), np.float32)
            deep_pose = current_pose + self.deep_delta if self.deep_delta is not None else current_pose
            deep_pose = deep_pose.astype(np.float32)
            deep_pose = deep_pose[0]
            print("Deepening to: ", deep_pose)
            self.ctrl.send_servo_pose(deep_pose)
            # # next position in base frame
            # pos_base = self._start_p + self._i * self._delta
            # if self._hold_quat_xyzw is None:
            #     # use your convenience method that holds orientation internally
            #     self.ctrl.send_servo_pose(pos_base)
            # else:
            #     # or explicitly send pose+quat if you prefer to lock orientation
            #     self.ctrl.send_servo_pose(pos_base)  # replace with send_pose_psm1(pos_base, quat=self._hold_quat_xyzw) if you have it

            # finish by steps or by time
            if self._i >= self._N or (now - self._t_enter) >= self.retract_cmd_s:
                self._state = "DONE"
                break

        # simple status for logging/UI
        pos_err = None
        if self._goal_p is not None:
            cur = np.asarray(self.ctrl.current_pos(), np.float32)
            pos_err = float(np.linalg.norm(cur - self._goal_p))
        return PlanStatus(self._state, pos_err, self.ctrl.current_jaw(), self.is_done())
