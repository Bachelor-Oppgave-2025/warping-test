"""Main entry point for the ball tracking system."""

import os
import time
from collections import deque

import ctypes
import cv2
import numpy as np

from detection import (
    detect_balls,
    detect_pockets_warp,
    detect_table_corners,
    detect_table_orientation,
)
from display import display_combined, draw_pockets, draw_tracked_balls, is_window_open
from tracking import KalmanBallTracker
from warping import TableWarper

SHORT_SIDE_W = 280
SHORT_SIDE_H = 560
INPUT_DOWNSCALE = 0.85  # Slightly reduce input resolution for faster processing.
PLAYBACK_FRAME_STEP = 1  # Process every frame for real-time playback.
DISPLAY_WAIT_MS = 1
PAUSED_SCRUB_INTERVAL_MS = 90  # Small debounce to keep held-arrow scrubbing stable.
LEFT_ARROW_KEYS = {2424832, 81}
RIGHT_ARROW_KEYS = {2555904, 83}
POCKET_CONFIRM_FRAMES = 1
POCKET_CLEAR_FRAMES = 5
POCKET_HIT_COOLDOWN_SECONDS = 5
VK_LEFT = 0x25
VK_RIGHT = 0x27


def _get_held_arrow_direction():
    """Return -1 for held left, +1 for held right, 0 otherwise (Windows only)."""
    try:
        user32 = ctypes.windll.user32
        left_down = bool(user32.GetAsyncKeyState(VK_LEFT) & 0x8000)
        right_down = bool(user32.GetAsyncKeyState(VK_RIGHT) & 0x8000)
        if left_down and not right_down:
            return -1
        if right_down and not left_down:
            return 1
    except Exception:  # pylint: disable=broad-exception-caught
        return 0
    return 0


def _init_pocket_states(pocket_count):
    """Initialize per-pocket confirmation state."""
    return [
        {
            "candidate": "none",
            "candidate_streak": 0,
            "clear_streak": 0,
            "confirmed": "none",
            "cooldown_until_s": 0.0,
            "armed": True,
        }
        for _ in range(pocket_count)
    ]


def _raw_pocket_candidate(pocket, tracked):
    """Return tracked ball color when a sphere overlaps the pocket rectangle."""
    px = pocket["x"]
    py = pocket["y"]
    ps = pocket["size"]
    if not tracked:
        return "none"

    # Slightly expand rectangle by ball radius to count edge overlaps.
    best_label = "none"
    best_score = -1.0
    cx = px + (ps / 2.0)
    cy = py + (ps / 2.0)
    for track_label, pos in tracked.items():
        if len(pos) == 3:
            bx, by, r = pos
        else:
            bx, by = pos
            r = 0

        expand = max(0, int(r))
        x1 = px - expand
        y1 = py - expand
        x2 = px + ps + expand
        y2 = py + ps + expand
        if not (x1 <= bx <= x2 and y1 <= by <= y2):
            continue

        # Prefer ball centers closer to pocket center when multiple overlap.
        dist_sq = float((bx - cx) ** 2 + (by - cy) ** 2)
        score = max(0.0, 1.0 - (dist_sq / max(1.0, float(ps * ps))))
        score += min(1.0, float(r) / 10.0)
        if score > best_score:
            best_score = score
            best_label = str(track_label).split("_", 1)[0].lower()

    return best_label if best_score >= 0.0 else "none"


def _pocket_color_reads(pockets, tracked, pocket_states, video_time_s):
    """Return stable per-pocket tracked-ball colors using temporal rules."""
    if len(pocket_states) != len(pockets):
        pocket_states = _init_pocket_states(len(pockets))

    reads = []
    for idx, pocket in enumerate(pockets):
        state = pocket_states[idx]
        if video_time_s < float(state.get("cooldown_until_s", 0.0)):
            # During cooldown, hide state and keep pocket disarmed.
            state["confirmed"] = "none"
            reads.append("none")
            continue

        candidate = _raw_pocket_candidate(pocket, tracked)

        if candidate == "none":
            state["candidate"] = "none"
            state["candidate_streak"] = 0
            state["clear_streak"] += 1
            if state["clear_streak"] >= POCKET_CLEAR_FRAMES:
                state["confirmed"] = "none"
                state["armed"] = True
        else:
            state["clear_streak"] = 0
            if not state.get("armed", True):
                reads.append("none")
                continue
            if state["candidate"] == candidate:
                state["candidate_streak"] += 1
            else:
                state["candidate"] = candidate
                state["candidate_streak"] = 1
            if (
                state["candidate_streak"] >= POCKET_CONFIRM_FRAMES
                and state["confirmed"] != candidate
            ):
                state["confirmed"] = candidate
                state["cooldown_until_s"] = video_time_s + POCKET_HIT_COOLDOWN_SECONDS
                state["armed"] = False

        reads.append(state["confirmed"])

    return reads, pocket_states


def _current_video_time_seconds(cap, frame_count, source_fps):
    """Return current video timestamp in seconds, with frame-based fallback."""
    pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
    if pos_msec > 0:
        return pos_msec / 1000.0
    if source_fps and source_fps > 1e-6:
        return float(frame_count) / float(source_fps)
    return 0.0


def _update_pocket_history(pocket_reads, prev_reads, history, frame_count):
    """Append new pocket events and return current reads as the next previous state."""
    normalized = [str(label).lower() for label in pocket_reads]
    if len(prev_reads) != len(normalized):
        prev_reads = ["none"] * len(normalized)

    for idx, label in enumerate(normalized):
        prev = prev_reads[idx]
        if label not in ("none", prev):
            history.appendleft(f"F{frame_count} P{idx + 1} {label.capitalize()}")

    return normalized


def select_video():
    """Ask user to select a video file."""
    print("\n=== Ball Tracking System ===")
    print("Available videos:")
    video_dir = "videos"

    if os.path.exists(video_dir):
        videos = [
            f
            for f in os.listdir(video_dir)
            if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
        for i, v in enumerate(videos, 1):
            print(f"  {i}. {v}")
    else:
        videos = []

    video_path = input("\nEnter video path (or video number): ").strip()

    # If user entered a number, select from available videos
    if video_path.isdigit():
        video_num = int(video_path) - 1
        if 0 <= video_num < len(videos):
            video_path = os.path.join(video_dir, videos[video_num])
        else:
            print("ERROR: Invalid video number")
            return None

    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        return None

    return video_path


def _downscale_frame(frame):
    """Apply a light input downscale to speed up the pipeline."""
    if frame is None or INPUT_DOWNSCALE >= 1.0:
        return frame

    h, w = frame.shape[:2]
    new_w = max(1, int(w * INPUT_DOWNSCALE))
    new_h = max(1, int(h * INPUT_DOWNSCALE))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _collect_stable_table_corners(cap, max_scan_frames=24):
    """Estimate stable table corners from several startup frames."""
    candidates = []

    for _ in range(max_scan_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = _downscale_frame(frame)
        corners = detect_table_corners(frame)
        if corners is not None:
            candidates.append(corners.astype(np.float32))

    if not candidates:
        return None

    stacked = np.stack(candidates, axis=0)
    median = np.median(stacked, axis=0)
    best_idx = int(np.argmin([np.linalg.norm(c - median) for c in candidates]))
    return candidates[best_idx]


def _process_frame(frame, warper, tracker, current_pockets, pocket_states):
    """Run the processing pipeline for a single frame."""
    frame = _downscale_frame(frame)
    warped = warper.warp(frame)
    detections = detect_balls(warped)
    tracked = tracker.update(detections)

    if not current_pockets:
        current_pockets = detect_pockets_warp(warped)
        pocket_states = _init_pocket_states(len(current_pockets))

    return frame, warped, tracked, current_pockets, pocket_states


def _initialize_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return None, None, None

    table_pts = _collect_stable_table_corners(cap, max_scan_frames=24)
    if table_pts is None:
        print("ERROR: Table not detected in startup frames")
        cap.release()
        return None, None, None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Detect table orientation
    is_long_side = detect_table_orientation(table_pts)
    if is_long_side:
        table_w, table_h = SHORT_SIDE_H, SHORT_SIDE_W  # Swap for landscape
        print("Detected: Filming from LONG side")
    else:
        table_w, table_h = SHORT_SIDE_W, SHORT_SIDE_H
        print("Detected: Filming from SHORT side")

    print(f"Detected table corners: {table_pts}")
    print(f"Warp dimensions: {table_w}x{table_h}")
    print("Press SPACE to pause/resume, LEFT/RIGHT to scrub while paused")
    print("Press P to jump to a frame, R to restart, ESC to quit")

    warper = TableWarper((table_w, table_h))
    warper.compute(table_pts)

    return cap, warper, is_long_side


def _video_frame_interval_ms(cap):
    """Return frame interval in milliseconds from source FPS."""
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6:
        return 0.0
    return 1000.0 / fps


def main():
    """Run the ball tracking pipeline for a selected video."""
    video_path = select_video()
    if video_path is None:
        return

    while True:
        cap, warper, is_long_side = _initialize_video(video_path)
        if cap is None:
            return

        tracker = KalmanBallTracker(
            max_missing=12, gating_distance=60, max_red_tracks=15
        )
        tracker_config = {
            "max_missing": 12,
            "gating_distance": 60,
            "max_red_tracks": 15,
        }
        frame_count = 0
        paused = False
        current_frame = None
        current_warped = None
        current_tracked = {}
        current_pockets = []
        pocket_states = []
        pocket_history = deque(maxlen=20)
        prev_pocket_reads = []

        # FPS tracking
        fps = 0
        prev_time = time.time()
        last_scrub_time = 0.0
        restart_requested = False
        frame_interval_ms = _video_frame_interval_ms(cap)
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        while True:
            loop_start = time.perf_counter()
            if not paused:
                step = max(1, int(PLAYBACK_FRAME_STEP))
                frames_advanced = 0
                for _ in range(step - 1):
                    if not cap.grab():
                        frames_advanced = -1
                        break
                    frames_advanced += 1
                if frames_advanced < 0:
                    break

                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += frames_advanced + 1
                (
                    current_frame,
                    current_warped,
                    current_tracked,
                    current_pockets,
                    pocket_states,
                ) = _process_frame(
                    frame,
                    warper,
                    tracker,
                    current_pockets,
                    pocket_states,
                )

            frame = current_frame
            warped = current_warped.copy()
            tracked = current_tracked

            # Draw tracked balls
            warped = draw_tracked_balls(warped, tracked)
            warped = draw_pockets(warped, current_pockets)
            video_time_s = _current_video_time_seconds(cap, frame_count, source_fps)
            pocket_reads, pocket_states = _pocket_color_reads(
                current_pockets, current_tracked, pocket_states, video_time_s
            )
            prev_pocket_reads = _update_pocket_history(
                pocket_reads, prev_pocket_reads, pocket_history, frame_count
            )

            # Calculate FPS
            elapsed = time.time() - prev_time
            if elapsed > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / elapsed)
            prev_time = time.time()

            # Display combined view with pause status
            pause_text = " [PAUSED]" if paused else ""
            info_text = (
                f"Frame: {frame_count} | FPS: {fps:.1f} | Balls: {len(tracked)} "
                f"{pause_text}"
            )
            panel_data = {
                "pocket_info": pocket_reads,
                "pocket_log": list(pocket_history),
            }
            display_combined(
                frame,
                warped,
                info_text,
                is_landscape=is_long_side,
                panel_data=panel_data,
            )

            # Handle key presses and window close events
            wait_ms = DISPLAY_WAIT_MS
            if not paused and frame_interval_ms > 0:
                elapsed_ms = (time.perf_counter() - loop_start) * 1000.0
                wait_ms = max(1, int(round(frame_interval_ms - elapsed_ms)))
            key = cv2.waitKeyEx(wait_ms)
            if not is_window_open():
                restart_requested = False
                break
            if key in (27, 1048603):  # ESC to quit
                restart_requested = False
                break
            if key in (ord("r"), ord("R"), ord("r") + 0x100000, ord("R") + 0x100000):
                restart_requested = True
                break
            if key in (ord("p"), ord("P"), ord("p") + 0x100000, ord("P") + 0x100000):
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                max_frame = max(1, total_frames)
                prompt = f"Jump to frame (1-{max_frame}): "
                try:
                    target_str = input(prompt).strip()
                except EOFError:
                    target_str = ""
                if not target_str:
                    continue
                if not target_str.isdigit():
                    print("Invalid frame number.")
                    continue

                target_frame = int(target_str)
                target_frame = max(1, min(max_frame, target_frame))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
                ret, frame = cap.read()
                if not ret:
                    print(f"Could not jump to frame {target_frame}.")
                    continue

                tracker = KalmanBallTracker(**tracker_config)
                current_pockets = []
                pocket_states = []
                current_tracked = {}
                (
                    current_frame,
                    current_warped,
                    current_tracked,
                    current_pockets,
                    pocket_states,
                ) = _process_frame(
                    frame,
                    warper,
                    tracker,
                    current_pockets,
                    pocket_states,
                )
                frame_count = target_frame
                print(f"Jumped to frame {target_frame}.")
                continue
            if key in (32, 1048608):  # SPACE to pause/resume
                paused = not paused
                continue
            scrub_direction = 0
            if paused:
                if key in LEFT_ARROW_KEYS:
                    scrub_direction = -1
                elif key in RIGHT_ARROW_KEYS:
                    scrub_direction = 1
                else:
                    scrub_direction = _get_held_arrow_direction()

            if paused and scrub_direction != 0:
                now = time.time()
                if (now - last_scrub_time) * 1000.0 < PAUSED_SCRUB_INTERVAL_MS:
                    continue
                last_scrub_time = now

                next_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if scrub_direction < 0:
                    target_pos = max(0, next_pos - 2)
                else:
                    target_pos = max(0, next_pos)

                cap.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
                ret, frame = cap.read()
                if not ret:
                    continue

                (
                    current_frame,
                    current_warped,
                    current_tracked,
                    current_pockets,
                    pocket_states,
                ) = _process_frame(
                    frame,
                    warper,
                    tracker,
                    current_pockets,
                    pocket_states,
                )
                frame_count = max(1, int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

        cap.release()
        if not restart_requested:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
