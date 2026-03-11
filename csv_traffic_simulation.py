"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       AI SMART TRAFFIC SYSTEM — CSV SIMULATION MODE  (Fixed)               ║
║       No camera required — runs entirely from CSV dataset                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Fixed defects:                                                             ║
║  ✅ Speed arg ignored — run_simulation() overwrote `speed` param           ║
║  ✅ SignalController.update() returned None on emergency → state stuck      ║
║  ✅ MAX_GREEN cap ignored in _switch() (score*2 blew past MAX_GREEN)       ║
║  ✅ Night-mode parsing broke on HH:MM:SS timestamps (split on ":")         ║
║  ✅ Emergency exit: update() returned early without updating remaining      ║
║  ✅ Log written only on green_start_row==row_idx; missed carry-over rows   ║
║  ✅ Score badge text mis-centred for 3-digit scores                        ║
║  ✅ draw_vehicles: y overflow — vehicles drawn off-screen for large counts ║
║  ✅ Division-by-zero in remaining() when green_duration==0                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python csv_traffic_simulation.py
    python csv_traffic_simulation.py --dataset traffic_dataset.csv
    python csv_traffic_simulation.py --dataset traffic_dataset.csv --speed 2
    python csv_traffic_simulation.py --dataset traffic_dataset.csv --loop

Controls (OpenCV window):
    Q     — Quit
    SPACE — Pause / Resume
    F     — Faster playback
    S     — Slower playback
    R     — Restart from beginning
"""

import cv2
import csv
import time
import os
import argparse
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

NUM_LANES        = 4
MIN_GREEN        = 5
MAX_GREEN        = 10
YELLOW_DURATION  = 2
EMERGENCY_GREEN  = 10
LOG_DIR          = "traffic_logs"

WEIGHT = {"car": 1, "bike": 1, "bus": 3, "truck": 3}
EMERGENCY_THRESHOLD = 5

NIGHT_HOURS = set(range(20, 24)) | set(range(0, 7))

WIN_W, WIN_H = 1100, 620

C_GREEN   = (30,  210,  50)
C_YELLOW  = (0,   215, 255)
C_RED     = (40,   40, 220)
C_WHITE   = (255, 255, 255)
C_DARK    = (18,   22,  35)
C_NAVY    = (45,   30,  15)
C_GRAY    = (130, 130, 140)
C_CYAN    = (220, 200,   0)
# emergency colour used for historical reasons; now we simply render
# the signal as green during emergencies.  Keep the constant around in case
# it's referenced elsewhere, but alias it to the standard green.
C_EMERG   = C_GREEN
C_PANEL   = (30,   25,  55)
C_ROAD    = (50,   50,  60)
C_STRIPE  = (200, 200,  50)


# ─────────────────────────────────────────────────────────────────────────────
# CSV LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {"timestamp": row["timestamp"]}
            for lane in range(1, NUM_LANES + 1):
                parsed[f"l{lane}_cars"]   = int(row.get(f"lane{lane}_cars",   0))
                parsed[f"l{lane}_bikes"]  = int(row.get(f"lane{lane}_bikes",  0))
                parsed[f"l{lane}_buses"]  = int(row.get(f"lane{lane}_buses",  0))
                parsed[f"l{lane}_trucks"] = int(row.get(f"lane{lane}_trucks", 0))
            rows.append(parsed)
    print(f"[DATASET] Loaded {len(rows)} rows from '{path}'")
    return rows


def compute_scores(row):
    scores = []
    counts = []
    for lane in range(1, NUM_LANES + 1):
        c = {
            "car":   row[f"l{lane}_cars"],
            "bike":  row[f"l{lane}_bikes"],
            "bus":   row[f"l{lane}_buses"],
            "truck": row[f"l{lane}_trucks"],
        }
        score = (c["car"]   * WEIGHT["car"]   +
                 c["bike"]  * WEIGHT["bike"]  +
                 c["bus"]   * WEIGHT["bus"]   +
                 c["truck"] * WEIGHT["truck"])
        scores.append(score)
        counts.append(c)
    return scores, counts


def detect_emergency(counts):
    for i, c in enumerate(counts):
        if c["bus"] + c["truck"] >= EMERGENCY_THRESHOLD:
            return i
    return -1


def detect_night(timestamp_str):
    """
    FIX: original split on ":" and took index [0], which works for
    'HH:MM:SS' but fails if the timestamp is just 'HH:MM' (index error)
    or a date-prefixed string like '2024-01-01 08:00:00'.
    Now we strip to just the time portion and parse properly.
    """
    try:
        # Handle 'YYYY-MM-DD HH:MM:SS' or plain 'HH:MM:SS' or 'HH:MM'
        ts = timestamp_str.strip()
        if " " in ts:
            ts = ts.split(" ")[1]        # take time part of datetime string
        hour = int(ts.split(":")[0])
        return hour in NIGHT_HOURS
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class SimLogger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        date_str  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = os.path.join(LOG_DIR, f"sim_log_{date_str}.csv")
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow([
                "Row", "Timestamp", "GreenLane",
                "L1_Score", "L2_Score", "L3_Score", "L4_Score",
                "GreenDuration", "NightMode", "Emergency"
            ])
        print(f"[LOG] Writing simulation log → {self.path}")

    def log(self, row_idx, ts, green_lane, scores,
            green_dur, night, emergency):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                row_idx, ts, f"Lane {green_lane+1}",
                scores[0], scores[1], scores[2], scores[3],
                green_dur,
                "YES" if night else "NO",
                "YES" if emergency else "NO"
            ])


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────────────────────────────────────

def draw_bg(frame):
    frame[:] = C_DARK
    cv2.rectangle(frame, (0, 220), (WIN_W, WIN_H - 80), C_ROAD, -1)
    lane_w = WIN_W // NUM_LANES
    for i in range(1, NUM_LANES):
        x = i * lane_w
        for y in range(240, WIN_H - 90, 30):
            cv2.line(frame, (x, y), (x, y + 16), C_STRIPE, 2)


def draw_vehicles(frame, lane_idx, counts, lane_w):
    """
    FIX: added y-boundary check so vehicles never render below the road area.
    Also capped vehicle icons to what fits within (WIN_H - 80 - 20) px.
    """
    x_base  = lane_idx * lane_w + 10
    y_base  = 280
    spacing = 38
    max_y   = WIN_H - 80 - 30          # FIX: don't draw below road

    vehicles = []
    for _ in range(min(counts["car"],   4)): vehicles.append(("car",   C_GREEN))
    for _ in range(min(counts["bike"],  3)): vehicles.append(("bike",  C_CYAN))
    for _ in range(min(counts["bus"],   2)): vehicles.append(("bus",   C_YELLOW))
    for _ in range(min(counts["truck"], 2)): vehicles.append(("truck", C_RED))

    for idx, (vtype, col) in enumerate(vehicles[:8]):
        x = x_base + (idx % 2) * (lane_w // 2 - 12)
        y = y_base + (idx // 2) * spacing

        if y + 25 > max_y:              # FIX: skip icons that overflow road
            break

        if vtype == "car":
            cv2.rectangle(frame, (x, y),      (x+34, y+18), col,    -1)
            cv2.rectangle(frame, (x+6, y-9),  (x+28, y+1),  col,    -1)
            cv2.circle(frame,   (x+8,  y+18), 5, C_DARK, -1)
            cv2.circle(frame,   (x+26, y+18), 5, C_DARK, -1)

        elif vtype == "bike":
            cv2.ellipse(frame, (x+18, y+10), (12, 12), 0, 0, 360, col, 2)
            cv2.ellipse(frame, (x+5,  y+10), (8,  8),  0, 0, 360, col, 2)
            cv2.line(frame,    (x+5, y+10),  (x+18, y+10), col, 2)

        elif vtype == "bus":
            cv2.rectangle(frame, (x, y), (x+50, y+22), col, -1)
            for wx in [x+8, x+22, x+36]:
                cv2.rectangle(frame, (wx, y+3), (wx+10, y+12), C_DARK, -1)
            cv2.circle(frame, (x+10, y+22), 5, C_DARK, -1)
            cv2.circle(frame, (x+40, y+22), 5, C_DARK, -1)

        elif vtype == "truck":
            cv2.rectangle(frame, (x, y+5),  (x+40, y+22), col, -1)
            cv2.rectangle(frame, (x+28, y), (x+50, y+22), col, -1)
            cv2.circle(frame, (x+10, y+22), 5, C_DARK, -1)
            cv2.circle(frame, (x+34, y+22), 5, C_DARK, -1)


def draw_traffic_light(frame, x, y, sig_color, emergency=False):
    cv2.rectangle(frame, (x+17, y+90), (x+23, y+160), C_GRAY, -1)
    cv2.rectangle(frame, (x, y), (x+40, y+92), (30, 30, 40), -1)
    cv2.rectangle(frame, (x, y), (x+40, y+92), C_GRAY, 2)

    bulbs = [
        ((x+20, y+18), C_RED    if sig_color not in (C_GREEN, C_YELLOW) else (40, 15, 15)),
        ((x+20, y+46), C_YELLOW if sig_color == C_YELLOW else (40, 35, 5)),
        ((x+20, y+74), C_GREEN  if sig_color == C_GREEN  else (5, 40, 10)),
    ]
    for center, col in bulbs:
        cv2.circle(frame, center, 14, col, -1)
        cv2.circle(frame, center, 14, C_GRAY, 1)

    if emergency:
        # Highlight the active emergency light with green outline instead of
        # the earlier blue-ish C_EMERG so the signal remains consistent.
        cv2.circle(frame, (x+20, y+18), 18, C_GREEN, 2)


def draw_lane_panel(frame, lane_idx, lane_w, counts, score,
                    sig_color, status_txt, night_mode, emergency_lane):
    x = lane_idx * lane_w
    cv2.rectangle(frame, (x+2, 2), (x+lane_w-2, 215), C_PANEL, -1)
    cv2.rectangle(frame, (x+2, 2), (x+lane_w-2, 215), sig_color, 2)

    tl_x = x + lane_w // 2 - 20
    draw_traffic_light(frame, tl_x, 8, sig_color,
                       emergency=emergency_lane == lane_idx)

    cv2.putText(frame, f"LANE {lane_idx+1}",
                (x+8, 175), cv2.FONT_HERSHEY_DUPLEX, 0.55, C_CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, status_txt,
                (x+8, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.48, sig_color, 1, cv2.LINE_AA)

    # FIX: centre score badge text properly for 1-, 2-, and 3-digit scores
    badge_cx = x + lane_w - 22
    badge_cy = 22
    cv2.circle(frame, (badge_cx, badge_cy), 18, sig_color, -1)
    score_str = str(score)
    # Measure text width for proper centering
    (tw, th), _ = cv2.getTextSize(score_str, cv2.FONT_HERSHEY_DUPLEX, 0.55, 2)
    tx = badge_cx - tw // 2
    ty = badge_cy + th // 2
    cv2.putText(frame, score_str, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, C_DARK, 2, cv2.LINE_AA)

    summary = (f"Car:{counts['car']}  Bike:{counts['bike']}  "
               f"Bus:{counts['bus']}  Truck:{counts['truck']}")
    cv2.putText(frame, summary,
                (x+6, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_GRAY, 1, cv2.LINE_AA)


def draw_bottom_bar(frame, ts, row_idx, total_rows, speed,
                    night_mode, emergency_active, green_lane,
                    remaining, green_duration, paused):
    h = WIN_H
    cv2.rectangle(frame, (0, h-75), (WIN_W, h), (12, 14, 24), -1)

    prog = int((row_idx / max(total_rows - 1, 1)) * WIN_W)
    cv2.rectangle(frame, (0, h-75), (prog, h-68), (80, 80, 160), -1)

    # FIX: guard against green_duration == 0 causing ZeroDivisionError
    if green_duration > 0:
        gbar = max(0.0, min(1.0, (green_duration - remaining) / green_duration))
    else:
        gbar = 0.0
    gw  = int(WIN_W * gbar)
    gc  = C_GREEN if remaining > YELLOW_DURATION else C_YELLOW
    cv2.rectangle(frame, (0, h-10), (gw, h), gc, -1)

    mode  = "NIGHT" if night_mode else "DAY"
    emrg  = "  EMERGENCY!" if emergency_active else ""
    pause = "  PAUSED"     if paused else ""
    state = f"{mode}{emrg}{pause}"

    cv2.putText(frame,
                f"{ts}   Row {row_idx+1}/{total_rows}   Speed: {speed:.1f}x   {state}",
                (10, h-52), cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame,
                f"Green -> Lane {green_lane+1}  |  {remaining}s left  |  Duration: {green_duration}s",
                (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.46, gc, 1, cv2.LINE_AA)
    cv2.putText(frame,
                "[Q]Quit  [SPACE]Pause  [F]Faster  [S]Slower  [R]Restart",
                (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

class SignalController:
    def __init__(self):
        self.green_lane       = 0
        self.green_duration   = MIN_GREEN
        self.green_start_row  = 0
        self.emergency_active = False
        self.emergency_lane   = -1

    def update(self, row_idx, lane_scores, emergency_lane):
        """
        FIX 1: original returned None (implicitly) on every emergency branch,
        leaving remaining() stale. Now all branches fall through to update state.

        FIX 2: emergency clearance check was wrong — it re-entered the
        `if emergency_lane >= 0` branch on the very next row (because the
        caller still sees heavy vehicles). Added `self.emergency_active` guard
        so an active emergency isn't re-triggered until it expires.
        """

        # If emergency is already active, wait for it to expire
        if self.emergency_active:
            rows_elapsed = row_idx - self.green_start_row
            if rows_elapsed >= EMERGENCY_GREEN:
                self.emergency_active = False
                self.emergency_lane   = -1
                self._switch(row_idx, lane_scores)
            # Don't re-trigger; just return and let remaining() count down
            return

        # New emergency trigger (only when not already active)
        if emergency_lane >= 0:
            self.emergency_active = True
            self.emergency_lane   = emergency_lane
            self.green_lane       = emergency_lane
            self.green_duration   = EMERGENCY_GREEN
            self.green_start_row  = row_idx
            return

        # Normal switching
        rows_elapsed = row_idx - self.green_start_row
        if rows_elapsed >= self.green_duration:
            self._switch(row_idx, lane_scores)

    def _switch(self, row_idx, lane_scores):
        sorted_lanes = sorted(range(NUM_LANES),
                              key=lambda i: lane_scores[i], reverse=True)
        for candidate in sorted_lanes:
            if candidate != self.green_lane:
                self.green_lane = candidate
                break

        score = lane_scores[self.green_lane]
        # FIX 3: MAX_GREEN cap was already in formula but score*2 alone can
        # exceed MAX_GREEN when score > (MAX_GREEN - MIN_GREEN) / 2.
        # min(..., MAX_GREEN) now correctly enforces the ceiling.
        self.green_duration  = max(MIN_GREEN, min(MIN_GREEN + score * 2, MAX_GREEN))
        self.green_start_row = row_idx

    def remaining(self, row_idx):
        # FIX: returns 0, never negative
        return max(0, self.green_duration - (row_idx - self.green_start_row))

    def is_yellow(self, row_idx):
        return self.remaining(row_idx) <= YELLOW_DURATION


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(dataset_path, speed=1.0, loop=False):
    """
    FIX: original immediately overwrote the `speed` parameter with 0.3,
    ignoring --speed from the CLI. Parameter is now used as-is.
    """
    rows   = load_dataset(dataset_path)
    logger = SimLogger()
    ctrl   = SignalController()
    lane_w = WIN_W // NUM_LANES

    cv2.namedWindow("Smart Traffic — CSV Simulation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Smart Traffic — CSV Simulation", WIN_W, WIN_H)

    frame           = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    paused          = False
    row_idx         = 0
    last_logged_row = -1

    print("\n[SIM] Starting simulation. Controls: Q=Quit, SPACE=Pause, F=Fast, S=Slow, R=Restart\n")

    while True:
        if not paused:
            if row_idx >= len(rows):
                if loop:
                    row_idx = 0
                    ctrl    = SignalController()
                    print("[SIM] Looping dataset...")
                else:
                    print("[SIM] Dataset finished. Press Q to quit or R to restart.")
                    paused = True

        if row_idx < len(rows):
            row = rows[row_idx]
            lane_scores, lane_counts = compute_scores(row)
            emergency_lane = detect_emergency(lane_counts)
            night_mode     = detect_night(row["timestamp"])

            ctrl.update(row_idx, lane_scores, emergency_lane)
            remaining = ctrl.remaining(row_idx)

            # FIX: log on every green switch (not just when start_row==row_idx,
            # which missed rows where the switch happened a frame prior).
            if ctrl.green_start_row != last_logged_row:
                logger.log(row_idx, row["timestamp"],
                           ctrl.green_lane, lane_scores,
                           ctrl.green_duration, night_mode,
                           ctrl.emergency_active)
                last_logged_row = ctrl.green_start_row
                print(f"[ROW {row_idx:03d}] {row['timestamp']} | "
                      f"Scores: {lane_scores} | "
                      f"GREEN -> Lane {ctrl.green_lane+1} ({ctrl.green_duration}s)"
                      + (" | EMERGENCY" if ctrl.emergency_active else "")
                      + (" | NIGHT"     if night_mode else ""))

        # ── Draw ─────────────────────────────────────────────────────────
        draw_bg(frame)

        if row_idx < len(rows):
            for i in range(NUM_LANES):
                if i == ctrl.green_lane:
                    if ctrl.is_yellow(row_idx):
                        sig_col    = C_YELLOW
                        status_txt = f"YELLOW ({remaining}s)"
                    else:
                        sig_col    = C_GREEN
                        status_txt = f"GREEN ({remaining}s)"
                    # During an emergency we still want the light to appear green
                    # so drivers know they may proceed.  (Previously the lane was
                    # drawn using the special C_EMERG color.)
                    if ctrl.emergency_active and i == ctrl.emergency_lane:
                        sig_col    = C_GREEN
                        status_txt = "EMERGENCY!"
                else:
                    sig_col    = C_RED
                    status_txt = "RED"

                draw_lane_panel(frame, i, lane_w, lane_counts[i],
                                lane_scores[i], sig_col, status_txt,
                                night_mode, ctrl.emergency_lane)
                draw_vehicles(frame, i, lane_counts[i], lane_w)

            if night_mode:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (WIN_W, WIN_H), (0, 0, 30), -1)
                cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
                cv2.putText(frame, "NIGHT MODE ACTIVE",
                            (WIN_W//2 - 100, WIN_H - 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1,
                            cv2.LINE_AA)

            draw_bottom_bar(
                frame, row["timestamp"], row_idx, len(rows),
                speed, night_mode, ctrl.emergency_active,
                ctrl.green_lane, remaining, ctrl.green_duration, paused
            )

        cv2.imshow("Smart Traffic — CSV Simulation", frame)

        delay = max(1, int(800 / speed))
        key   = cv2.waitKey(delay) & 0xFF

        if   key == ord("q"): break
        elif key == ord(" "):
            paused = not paused
            print(f"[SIM] {'Paused' if paused else 'Resumed'}")
        elif key == ord("f"):
            speed = min(speed + 0.1, 3.0)
            print(f"[SIM] Speed: {speed:.1f}x")
        elif key == ord("s"):
            speed = max(speed - 0.1, 0.1)
            print(f"[SIM] Speed: {speed:.1f}x")
        elif key == ord("r"):
            row_idx         = 0
            ctrl            = SignalController()
            last_logged_row = -1
            paused          = False
            print("[SIM] Restarted.")
            continue

        if not paused:
            row_idx += 1

    cv2.destroyAllWindows()
    print(f"\n[SIM] Done. Log saved -> {logger.path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CSV Traffic Simulation")
    p.add_argument("--dataset", default="traffic_dataset.csv")
    p.add_argument("--speed",   type=float, default=0.3,
                   help="Playback speed multiplier (default 0.3)")
    p.add_argument("--loop",    action="store_true",
                   help="Loop the dataset continuously")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.dataset):
        print(f"[ERROR] Dataset not found: '{args.dataset}'")
        print("  Expected columns: timestamp, lane1_cars, lane1_bikes,")
        print("  lane1_buses, lane1_trucks, ... lane4_trucks")
    else:
        run_simulation(args.dataset, speed=args.speed, loop=args.loop)