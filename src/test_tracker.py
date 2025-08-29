
    # ap.add_argument("--video", type=str, default="/docker-ros/ws/src/lite_tracker/assets/stir-5-seq-01.mp4", help="path to video or webcam index (e.g., 0)")
    # ap.add_argument("--checkpoint", type=str, default= "./weights/scaled_online.pth", help="path to scaled_online.pth")
    # ap.add_argument("--num_points", type=int, default=4)
    # ap.add_argument("--no_norm", action="store_true", help="do not normalize image to [0,1]")

# demo_tracking_module_stream.py
import argparse, cv2
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lite_tracker', 'scripts', 'lite_tracker'))
from tracking_module_new import LiteTrackingModule, TrackerConfig

def pick_points(img, n=4, win="pick points"):
    pts = []
    def cb(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((x, y))
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)
    cv2.setMouseCallback(win, cb)
    while True:
        vis = img.copy()
        for i,(u,v) in enumerate(pts, 1):
            cv2.circle(vis, (u,v), 5, (0,255,255), -1, cv2.LINE_AA)
            cv2.putText(vis, f"P{i}", (u+8,v-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Click {n} points  |  ENTER=OK  r=reset  q=quit", (16,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,20,20), 2, cv2.LINE_AA)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (13,10):  # Enter
            if len(pts) == n: break
        elif k == ord('r'): pts.clear()
        elif k in (27, ord('q')): raise SystemExit("Cancelled")
    cv2.destroyWindow(win)
    return pts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/docker-ros/ws/src/lite_tracker/assets/stir-5-seq-01.mp4", help="webcam index (e.g. 0) or video path")
    ap.add_argument("--checkpoint", default="./weights/scaled_online.pth", help="path to scaled_online.pth")
    ap.add_argument("--num_points", type=int, default=4)
    args = ap.parse_args()

    src = 0 if (args.video.isdigit() and len(args.video) < 5) else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open {args.video}")

    ok, frame = cap.read()
    if not ok: raise RuntimeError("Could not read first frame")
    # OpenCV is BGR; module wants RGB:
    first_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pts = pick_points(frame, n=args.num_points)

    # Build tracker (no ROI/graph—manual points only)
    cfg = TrackerConfig(checkpoint=args.checkpoint, grid_query_frame=0, use_autocast=True)
    trk = LiteTrackingModule(cfg)
    trk.set_keypoints_edges(keypoints_xy=pts, edges=[])     # edges not needed for plain tracking
    trk.initialize(first_frame_rgb=first_rgb, mode="manual")

    while True:
        ok, bgr = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        coords_cpu, _, _ = trk.step(rgb)  # one step

        # get last (u,v) as ints
        uv = trk.last_keypoints_int()

        # draw overlay
        disp = bgr.copy()
        for (u0,v0) in pts:
            cv2.circle(disp, (u0,v0), 5, (0,0,255), 2, cv2.LINE_AA)  # original (red)
        for i,(u,v) in enumerate(uv):
            cv2.circle(disp, (u,v), 5, (0,255,0), -1, cv2.LINE_AA)   # tracked (green)
            if i < len(pts):
                cv2.line(disp, pts[i], (u,v), (0,255,255), 1, cv2.LINE_AA)
        cv2.imshow("LiteTrackingModule demo", disp)
        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()