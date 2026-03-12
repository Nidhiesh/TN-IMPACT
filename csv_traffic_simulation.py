
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       AI SMART TRAFFIC SYSTEM — CSV SIMULATION MODE  (v3)                  ║
║       Emergency override triggered ONLY by ambulance / fire / police       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Features:                                                                  ║
║  ✅ LIVE QUEUE MODEL — vehicles decrease on GREEN, increase on RED         ║
║  ✅ Emergency override for ambulance / fire truck / police ONLY            ║
║  ✅ Emergency reason banner shown on screen with vehicle type + lane       ║
║  ✅ Dynamic green time (urgency score-based, 5–30s)                        ║
║  ✅ Starvation prevention (wait penalty)                                   ║
║  ✅ Yellow light transition (last 2s of green)                              ║
║  ✅ Night mode (reduced arrivals after 20:00)                              ║
║  ✅ CSV output log of every signal cycle                                   ║
║  ✅ Speed control / Pause / Restart                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Dataset columns required:
    timestamp,
    lane1_cars, lane1_bikes, lane1_buses, lane1_trucks,
    lane2_cars, ... lane4_trucks,
    emergency_lane,    ← "lane1"/"lane2"/"lane3"/"lane4" or "none"
    emergency_vehicle  ← "ambulance"/"fire"/"police" or "none"

Usage:
    python csv_traffic_simulation.py
    python csv_traffic_simulation.py --dataset traffic_dataset.csv --speed 0.5
    python csv_traffic_simulation.py --dataset traffic_dataset.csv --loop

Controls:
    Q — Quit   SPACE — Pause   F — Faster   S — Slower   R — Restart
"""

import cv2
import csv
import os
import argparse
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

NUM_LANES            = 4
MIN_GREEN            = 5
MAX_GREEN            = 10
YELLOW_DURATION      = 2
EMERGENCY_GREEN      = 6        # seconds of green for emergency vehicle
LOG_DIR              = "traffic_logs"

# Vehicle weight for score calculation
WEIGHT = {"car": 1, "bike": 1, "bus": 3, "truck": 3}

# How many vehicles LEAVE the queue per row when lane is GREEN
DISCHARGE_RATE = {"car": 2, "bike": 2, "bus": 1, "truck": 1}

# Fraction of CSV count that ARRIVES each row when lane is RED
ARRIVAL_FRACTION     = 0.4
NIGHT_ARRIVAL_FACTOR = 0.4      # reduced arrivals during night

# Night hours
NIGHT_HOURS = set(range(20, 24)) | set(range(0, 7))

# Starvation prevention
WAIT_PENALTY_PER_ROW = 0.5

# Recognised emergency vehicle types and their display labels + colors (BGR)
EMERGENCY_VEHICLES = {
    "ambulance":  {"label": "AMBULANCE",   "color": (50,  50,  255)},
    "fire":       {"label": "FIRE TRUCK",  "color": (0,   80,  255)},
    "police":     {"label": "POLICE CAR",  "color": (200, 100,  0 )},
}

# Window
WIN_W, WIN_H = 1200, 680

# Colors (BGR)
C_GREEN   = (30,  210,  50)
C_YELLOW  = (0,   215, 255)
C_RED     = (40,   40, 220)
C_WHITE   = (255, 255, 255)
C_DARK    = (18,   22,  35)
C_GRAY    = (130, 130, 140)
C_CYAN    = (220, 200,   0)
C_PANEL   = (30,   25,  55)
C_ROAD    = (50,   50,  60)
C_STRIPE  = (200, 200,  50)
C_ARRIVE  = (100, 255, 180)
C_DEPART  = (80,   80, 255)
C_EMERG   = (50,   50, 255)     # ambulance red


# ─────────────────────────────────────────────────────────────────────────────
# CSV LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {"timestamp":         row["timestamp"],
                      "emergency_lane":    row.get("emergency_lane",    "none").strip().lower(),
                      "emergency_vehicle": row.get("emergency_vehicle", "none").strip().lower()}
            for lane in range(1, NUM_LANES + 1):
                parsed[f"l{lane}_cars"]   = int(row.get(f"lane{lane}_cars",   0))
                parsed[f"l{lane}_bikes"]  = int(row.get(f"lane{lane}_bikes",  0))
                parsed[f"l{lane}_buses"]  = int(row.get(f"lane{lane}_buses",  0))
                parsed[f"l{lane}_trucks"] = int(row.get(f"lane{lane}_trucks", 0))
            rows.append(parsed)
    print(f"[DATASET] Loaded {len(rows)} rows from '{path}'")
    return rows


def get_csv_counts(row, lane_idx):
    i = lane_idx + 1
    return {"car":   row[f"l{i}_cars"],
            "bike":  row[f"l{i}_bikes"],
            "bus":   row[f"l{i}_buses"],
            "truck": row[f"l{i}_trucks"]}


def score_of(counts):
    return sum(counts[v] * WEIGHT[v] for v in WEIGHT)


def detect_emergency(row):
    """
    Read emergency info directly from the CSV row.
    Returns (lane_index, vehicle_type, reason_string) if emergency,
    else (-1, "", "").

    Emergency is triggered ONLY when an ambulance / fire truck / police car
    is detected approaching the intersection in a specific lane.
    Heavy vehicles like buses and trucks do NOT trigger emergency override —
    they are treated as normal high-weight traffic.
    """
    elane   = row["emergency_lane"]
    evehicle = row["emergency_vehicle"]

    if elane == "none" or evehicle == "none":
        return -1, "", ""

    # Map "lane1"→0, "lane2"→1, etc.
    lane_map = {"lane1": 0, "lane2": 1, "lane3": 2, "lane4": 3}
    lane_idx = lane_map.get(elane, -1)
    if lane_idx == -1:
        return -1, "", ""

    if evehicle not in EMERGENCY_VEHICLES:
        return -1, "", ""

    label  = EMERGENCY_VEHICLES[evehicle]["label"]
    reason = (f"{label} detected in Lane {lane_idx + 1}! "
              f"Immediate green required to clear the path. "
              f"All other lanes hold — emergency vehicle has right of way.")
    return lane_idx, evehicle, reason


def detect_night(timestamp_str):
    try:
        ts = timestamp_str.strip()
        if " " in ts:
            ts = ts.split(" ")[1]
        hour = int(ts.split(":")[0])
        return hour in NIGHT_HOURS
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# LIVE VEHICLE QUEUE
# ─────────────────────────────────────────────────────────────────────────────

class VehicleQueue:
    def __init__(self, initial_row):
        self.queues = []
        for i in range(NUM_LANES):
            c = get_csv_counts(initial_row, i)
            self.queues.append({"car": c["car"], "bike": c["bike"],
                                "bus": c["bus"], "truck": c["truck"]})
        self.total_cleared = [0] * NUM_LANES
        self.total_arrived = [0] * NUM_LANES
        self.rows_waiting  = [0] * NUM_LANES

    def update(self, csv_row, green_lane, night_mode):
        departed = [{"car":0,"bike":0,"bus":0,"truck":0} for _ in range(NUM_LANES)]
        arrived  = [{"car":0,"bike":0,"bus":0,"truck":0} for _ in range(NUM_LANES)]
        arrival_factor = NIGHT_ARRIVAL_FACTOR if night_mode else 1.0

        for i in range(NUM_LANES):
            csv_counts = get_csv_counts(csv_row, i)
            if i == green_lane:
                for vtype in DISCHARGE_RATE:
                    leave = min(self.queues[i][vtype], DISCHARGE_RATE[vtype])
                    self.queues[i][vtype] -= leave
                    departed[i][vtype]     = leave
                    self.total_cleared[i] += leave
                self.rows_waiting[i] = 0
            else:
                for vtype in ("car", "bike", "bus", "truck"):
                    arrive = max(0, round(
                        csv_counts[vtype] * ARRIVAL_FRACTION * arrival_factor))
                    self.queues[i][vtype] += arrive
                    arrived[i][vtype]      = arrive
                    self.total_arrived[i] += arrive
                self.rows_waiting[i] += 1

        return list(self.queues), departed, arrived

    def scores(self):
        return [score_of(q) for q in self.queues]

    def urgency_scores(self):
        base = self.scores()
        return [base[i] + self.rows_waiting[i] * WAIT_PENALTY_PER_ROW
                for i in range(NUM_LANES)]


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
                "L1_Score","L2_Score","L3_Score","L4_Score",
                "L1_Queue","L2_Queue","L3_Queue","L4_Queue",
                "GreenDuration","NightMode","Emergency","EmergencyVehicle"
            ])
        print(f"[LOG] Writing log -> {self.path}")

    def log(self, row_idx, ts, green_lane, scores, queues,
            green_dur, night, emergency, evehicle):
        q = [score_of(queues[i]) for i in range(NUM_LANES)]
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                row_idx, ts, f"Lane {green_lane+1}",
                scores[0], scores[1], scores[2], scores[3],
                q[0], q[1], q[2], q[3],
                green_dur,
                "YES" if night else "NO",
                "YES" if emergency else "NO",
                evehicle.upper() if evehicle else "-"
            ])


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

class SignalController:
    def __init__(self, starting_lane=0):
        # starting_lane is computed from first-row scores so the busiest
        # lane gets green immediately instead of always defaulting to Lane 1
        self.green_lane        = starting_lane
        self.green_duration    = MIN_GREEN
        self.green_start_row   = 0
        self.emergency_active  = False
        self.emergency_lane    = -1
        self.emergency_vehicle = ""

    def update(self, row_idx, urgency_scores, emergency_lane, emergency_vehicle):
        # Active emergency: wait for it to expire
        if self.emergency_active:
            if row_idx - self.green_start_row >= EMERGENCY_GREEN:
                self.emergency_active   = False
                self.emergency_lane     = -1
                self.emergency_vehicle  = ""
                self._switch(row_idx, urgency_scores)
            return

        # New emergency vehicle detected
        if emergency_lane >= 0:
            self.emergency_active   = True
            self.emergency_lane     = emergency_lane
            self.emergency_vehicle  = emergency_vehicle
            self.green_lane         = emergency_lane
            self.green_duration     = EMERGENCY_GREEN
            self.green_start_row    = row_idx
            return

        # Normal switching
        if row_idx - self.green_start_row >= self.green_duration:
            self._switch(row_idx, urgency_scores)

    def _switch(self, row_idx, urgency_scores):
        sorted_lanes = sorted(range(NUM_LANES),
                              key=lambda i: urgency_scores[i], reverse=True)
        for candidate in sorted_lanes:
            if candidate != self.green_lane:
                self.green_lane = candidate
                break
        score = urgency_scores[self.green_lane]
        self.green_duration  = max(MIN_GREEN, min(MIN_GREEN + score * 2, MAX_GREEN))
        self.green_start_row = row_idx

    def remaining(self, row_idx):
        return max(0, self.green_duration - (row_idx - self.green_start_row))

    def is_yellow(self, row_idx):
        return 0 < self.remaining(row_idx) <= YELLOW_DURATION


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────────────────────────────────────

def draw_bg(frame):
    frame[:] = C_DARK
    cv2.rectangle(frame, (0, 250), (WIN_W, WIN_H - 95), C_ROAD, -1)
    lane_w = WIN_W // NUM_LANES
    for i in range(1, NUM_LANES):
        x = i * lane_w
        for y in range(270, WIN_H - 105, 30):
            cv2.line(frame, (x, y), (x, y + 16), C_STRIPE, 2)


def _draw_icon(frame, vtype, col, x, y):
    if vtype == "car":
        cv2.rectangle(frame, (x,    y),   (x+32, y+16), col,   -1)
        cv2.rectangle(frame, (x+5,  y-8), (x+27, y+1),  col,   -1)
        cv2.circle(frame,    (x+7,  y+16), 5, C_DARK, -1)
        cv2.circle(frame,    (x+25, y+16), 5, C_DARK, -1)
    elif vtype == "bike":
        cv2.ellipse(frame, (x+17, y+9), (11,11), 0, 0, 360, col, 2)
        cv2.ellipse(frame, (x+5,  y+9), (7, 7),  0, 0, 360, col, 2)
        cv2.line(frame,    (x+5,  y+9), (x+17, y+9), col, 2)
    elif vtype == "bus":
        cv2.rectangle(frame, (x, y), (x+46, y+20), col, -1)
        for wx in [x+6, x+19, x+32]:
            cv2.rectangle(frame, (wx, y+3), (wx+9, y+11), C_DARK, -1)
        cv2.circle(frame, (x+9,  y+20), 5, C_DARK, -1)
        cv2.circle(frame, (x+37, y+20), 5, C_DARK, -1)
    elif vtype == "truck":
        cv2.rectangle(frame, (x,    y+4), (x+38, y+20), col, -1)
        cv2.rectangle(frame, (x+26, y),   (x+48, y+20), col, -1)
        cv2.circle(frame, (x+9,  y+20), 5, C_DARK, -1)
        cv2.circle(frame, (x+32, y+20), 5, C_DARK, -1)
    elif vtype == "ambulance":
        # White box with red cross
        cv2.rectangle(frame, (x, y), (x+38, y+20), (230,230,230), -1)
        cv2.rectangle(frame, (x+16, y+4), (x+22, y+16), (0,0,220), -1)
        cv2.rectangle(frame, (x+12, y+8), (x+26, y+12), (0,0,220), -1)
        cv2.circle(frame, (x+8,  y+20), 5, C_DARK, -1)
        cv2.circle(frame, (x+30, y+20), 5, C_DARK, -1)


def draw_vehicles(frame, lane_idx, counts, lane_w, departed, arrived,
                  has_emergency_vehicle=False, evehicle=""):
    x_base  = lane_idx * lane_w + 8
    y_base  = 310
    spacing = 42
    max_y   = WIN_H - 100

    vehicles = []
    # Draw ambulance icon first if present in this lane
    if has_emergency_vehicle and evehicle in EMERGENCY_VEHICLES:
        vehicles.append(("ambulance", EMERGENCY_VEHICLES[evehicle]["color"]))

    for _ in range(min(counts["car"],   5)): vehicles.append(("car",   C_GREEN))
    for _ in range(min(counts["bike"],  4)): vehicles.append(("bike",  C_CYAN))
    for _ in range(min(counts["bus"],   3)): vehicles.append(("bus",   C_YELLOW))
    for _ in range(min(counts["truck"], 3)): vehicles.append(("truck", C_RED))

    for idx, (vtype, col) in enumerate(vehicles[:10]):
        x = x_base + (idx % 2) * (lane_w // 2 - 14)
        y = y_base + (idx // 2) * spacing
        if y + 28 > max_y:
            break
        _draw_icon(frame, vtype, col, x, y)

    # Departure arrow
    total_departed = sum(departed.values())
    if total_departed > 0:
        ax = lane_idx * lane_w + lane_w // 2
        cv2.arrowedLine(frame, (ax, max_y+5), (ax, max_y-12),
                        C_ARRIVE, 3, tipLength=0.5)
        cv2.putText(frame, f"-{total_departed} left",
                    (ax-20, max_y-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_ARRIVE, 1, cv2.LINE_AA)

    # Arrival arrow
    total_arrived = sum(arrived.values())
    if total_arrived > 0:
        ax = lane_idx * lane_w + lane_w // 2
        cv2.arrowedLine(frame, (ax, max_y-12), (ax, max_y+5),
                        C_DEPART, 2, tipLength=0.5)
        cv2.putText(frame, f"+{total_arrived} joined",
                    (ax-24, max_y-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_DEPART, 1, cv2.LINE_AA)


def draw_traffic_light(frame, x, y, sig_color, emergency=False, evehicle=""):
    cv2.rectangle(frame, (x+17, y+90), (x+23, y+165), C_GRAY, -1)
    cv2.rectangle(frame, (x, y), (x+40, y+92), (30,30,40), -1)
    cv2.rectangle(frame, (x, y), (x+40, y+92), C_GRAY, 2)
    bulbs = [
        ((x+20, y+18), C_RED    if sig_color not in (C_GREEN, C_YELLOW) else (40,15,15)),
        ((x+20, y+46), C_YELLOW if sig_color == C_YELLOW                else (40,35, 5)),
        ((x+20, y+74), C_GREEN  if sig_color == C_GREEN                 else ( 5,40,10)),
    ]
    for center, col in bulbs:
        cv2.circle(frame, center, 14, col, -1)
        cv2.circle(frame, center, 14, C_GRAY, 1)

    # Emergency: white + vehicle-color rings around green bulb
    if emergency and evehicle in EMERGENCY_VEHICLES:
        ecol = EMERGENCY_VEHICLES[evehicle]["color"]
        cv2.circle(frame, (x+20, y+74), 19, C_WHITE, 3)
        cv2.circle(frame, (x+20, y+74), 23, ecol, 2)


def draw_lane_panel(frame, lane_idx, lane_w, counts, score,
                    sig_color, status_txt, night_mode,
                    emergency_lane, evehicle,
                    total_cleared, rows_waiting):
    x = lane_idx * lane_w
    cv2.rectangle(frame, (x+2, 2),       (x+lane_w-2, 245), C_PANEL,   -1)
    cv2.rectangle(frame, (x+2, 2),       (x+lane_w-2, 245), sig_color,  2)

    tl_x = x + lane_w // 2 - 20
    draw_traffic_light(frame, tl_x, 8, sig_color,
                       emergency=emergency_lane == lane_idx,
                       evehicle=evehicle)

    cv2.putText(frame, f"LANE {lane_idx+1}",
                (x+8, 196), cv2.FONT_HERSHEY_DUPLEX, 0.55, C_CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, status_txt,
                (x+8, 214), cv2.FONT_HERSHEY_SIMPLEX, 0.46, sig_color, 1, cv2.LINE_AA)

    # Score badge
    badge_cx, badge_cy = x + lane_w - 24, 22
    cv2.circle(frame, (badge_cx, badge_cy), 20, sig_color, -1)
    score_str = str(score)
    (tw, th), _ = cv2.getTextSize(score_str, cv2.FONT_HERSHEY_DUPLEX, 0.55, 2)
    cv2.putText(frame, score_str,
                (badge_cx - tw//2, badge_cy + th//2),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, C_DARK, 2, cv2.LINE_AA)

    summary = (f"Car:{counts['car']}  Bike:{counts['bike']}  "
               f"Bus:{counts['bus']}  Truck:{counts['truck']}")
    cv2.putText(frame, summary,
                (x+6, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Cleared:{total_cleared}",
                (x+6, 241), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_ARRIVE, 1, cv2.LINE_AA)
    if rows_waiting > 0:
        cv2.putText(frame, f"Wait:{rows_waiting}r",
                    (x + lane_w//2, 241),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_DEPART, 1, cv2.LINE_AA)


def draw_emergency_banner(frame, reason, evehicle):
    """
    Full-width banner at top showing emergency vehicle type and reason.
    Color matches the emergency vehicle type.
    """
    ecol = EMERGENCY_VEHICLES.get(evehicle, {}).get("color", C_EMERG)
    elabel = EMERGENCY_VEHICLES.get(evehicle, {}).get("label", "EMERGENCY VEHICLE")

    # Banner background
    cv2.rectangle(frame, (0, 0), (WIN_W, 42), (10, 10, 40), -1)
    cv2.rectangle(frame, (0, 0), (WIN_W, 42), ecol, 2)

    # Flashing title
    cv2.putText(frame, f"!! {elabel} DETECTED — EMERGENCY OVERRIDE !!",
                (8, 16), cv2.FONT_HERSHEY_DUPLEX, 0.52, ecol, 1, cv2.LINE_AA)

    # Reason line
    reason_short = reason if len(reason) <= 115 else reason[:112] + "..."
    cv2.putText(frame, f"REASON: {reason_short}",
                (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_WHITE, 1, cv2.LINE_AA)


def draw_queue_bars(frame, lane_w, live_scores, max_score=60):
    bar_y = WIN_H - 90
    for i in range(NUM_LANES):
        x     = i * lane_w + 4
        ratio = min(live_scores[i] / max(max_score, 1), 1.0)
        bw    = int((lane_w - 8) * ratio)
        cv2.rectangle(frame, (x, bar_y), (x + lane_w - 8, bar_y + 14), (40,40,50), -1)
        bar_col = C_RED if ratio > 0.66 else (C_YELLOW if ratio > 0.33 else C_GREEN)
        if bw > 0:
            cv2.rectangle(frame, (x, bar_y), (x + bw, bar_y + 14), bar_col, -1)
        cv2.putText(frame, f"Q:{live_scores[i]}",
                    (x+4, bar_y+11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_WHITE, 1, cv2.LINE_AA)


def draw_bottom_bar(frame, ts, row_idx, total_rows, speed,
                    night_mode, emergency_active, green_lane,
                    remaining, green_duration, paused):
    h = WIN_H
    cv2.rectangle(frame, (0, h-72), (WIN_W, h), (12,14,24), -1)
    prog = int((row_idx / max(total_rows-1, 1)) * WIN_W)
    cv2.rectangle(frame, (0, h-72), (prog, h-65), (80,80,160), -1)
    gbar = max(0.0, min(1.0, (green_duration - remaining) / max(green_duration, 1)))
    gw   = int(WIN_W * gbar)
    gc   = C_GREEN if remaining > YELLOW_DURATION else C_YELLOW
    cv2.rectangle(frame, (0, h-8), (gw, h), gc, -1)
    mode  = "NIGHT" if night_mode else "DAY"
    emrg  = "  EMERGENCY OVERRIDE!" if emergency_active else ""
    pause = "  PAUSED" if paused else ""
    cv2.putText(frame,
                f"{ts}   Row {row_idx+1}/{total_rows}   Speed:{speed:.1f}x   {mode}{emrg}{pause}",
                (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame,
                f"Green -> Lane {green_lane+1}  |  {remaining}s remaining  |  Duration: {green_duration}s",
                (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.46, gc, 1, cv2.LINE_AA)
    cv2.putText(frame,
                "[Q]Quit  [SPACE]Pause  [F]Faster  [S]Slower  [R]Restart",
                (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GRAY, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(dataset_path, speed=1.0, loop=False):
    rows   = load_dataset(dataset_path)
    logger = SimLogger()
    lane_w = WIN_W // NUM_LANES

    cv2.namedWindow("Smart Traffic — Live Queue Simulation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Smart Traffic — Live Queue Simulation", WIN_W, WIN_H)
    frame = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)

    def fresh_state():
        # Compute scores from the very first CSV row
        first_vq = VehicleQueue(rows[0])
        first_scores = first_vq.scores()
        # Pick the lane with the highest score as the starting green lane
        busiest_lane = max(range(NUM_LANES), key=lambda i: first_scores[i])
        print(f"[SIM] Starting GREEN lane: Lane {busiest_lane+1} "              f"(highest score={first_scores[busiest_lane]} from {first_scores})")
        return SignalController(starting_lane=busiest_lane), VehicleQueue(rows[0])

    ctrl, vqueue    = fresh_state()
    paused          = False
    row_idx         = 0
    last_logged_row = -1
    emergency_reason  = ""
    active_evehicle   = ""

    empty_flow = lambda: [{"car":0,"bike":0,"bus":0,"truck":0} for _ in range(NUM_LANES)]
    departed = empty_flow()
    arrived  = empty_flow()

    print("\n[SIM] Smart Traffic Simulation — Emergency = Ambulance / Fire / Police ONLY")
    print("[SIM] Buses and trucks are treated as normal heavy traffic (not emergency)")
    print("[SIM] Controls: Q=Quit  SPACE=Pause  F=Faster  S=Slower  R=Restart\n")

    while True:

        # ── Dataset end ───────────────────────────────────────────────────
        if not paused and row_idx >= len(rows):
            if loop:
                row_idx = 0
                ctrl, vqueue = fresh_state()
                departed, arrived = empty_flow(), empty_flow()
                emergency_reason, active_evehicle = "", ""
                last_logged_row = -1
                print("[SIM] Looping...")
            else:
                print("[SIM] Dataset finished. Press Q or R.")
                paused = True

        # ── Simulation step ───────────────────────────────────────────────
        if row_idx < len(rows):
            row        = rows[row_idx]
            night_mode = detect_night(row["timestamp"])

            # 1. Update live queues
            live_counts, departed, arrived = vqueue.update(
                row, ctrl.green_lane, night_mode)

            # 2. Scores from live queues
            live_scores    = vqueue.scores()
            urgency_scores = vqueue.urgency_scores()

            # 3. Emergency detection — ONLY ambulance / fire / police
            emergency_lane, emergency_vehicle, reason = detect_emergency(row)

            # Keep reason and vehicle type alive during active override
            if emergency_lane >= 0:
                emergency_reason = reason
                active_evehicle  = emergency_vehicle
            elif not ctrl.emergency_active:
                emergency_reason = ""
                active_evehicle  = ""

            # 4. Signal controller
            ctrl.update(row_idx, urgency_scores, emergency_lane, emergency_vehicle)
            remaining = ctrl.remaining(row_idx)

            # 5. Log on each switch
            if ctrl.green_start_row != last_logged_row:
                logger.log(row_idx, row["timestamp"],
                           ctrl.green_lane, live_scores, live_counts,
                           ctrl.green_duration, night_mode,
                           ctrl.emergency_active, active_evehicle)
                last_logged_row = ctrl.green_start_row
                print(f"[ROW {row_idx:03d}] {row['timestamp']} | "
                      f"Queue:{live_scores} | "
                      f"GREEN -> Lane {ctrl.green_lane+1} ({ctrl.green_duration}s)"
                      + (f" | EMERGENCY [{active_evehicle.upper()}]"
                         if ctrl.emergency_active else "")
                      + (" | NIGHT" if night_mode else ""))
                if ctrl.emergency_active and emergency_reason:
                    print(f"  [!] {emergency_reason}")

        # ── Draw ─────────────────────────────────────────────────────────
        draw_bg(frame)

        if row_idx < len(rows):
            for i in range(NUM_LANES):
                # Is this lane's emergency vehicle still present this row?
                lane_has_emerg = (ctrl.emergency_active and
                                  i == ctrl.emergency_lane)

                if i == ctrl.green_lane:
                    if ctrl.emergency_active and i == ctrl.emergency_lane:
                        ecol       = EMERGENCY_VEHICLES.get(
                                         active_evehicle, {}).get("color", C_EMERG)
                        elabel     = EMERGENCY_VEHICLES.get(
                                         active_evehicle, {}).get("label", "EMERGENCY")
                        sig_col    = C_GREEN
                        status_txt = f"GREEN-{elabel} ({remaining}s)"
                    elif ctrl.is_yellow(row_idx):
                        sig_col    = C_YELLOW
                        status_txt = f"YELLOW ({remaining}s)"
                    else:
                        sig_col    = C_GREEN
                        status_txt = f"GREEN ({remaining}s)"
                else:
                    sig_col    = C_RED
                    status_txt = "RED"

                draw_lane_panel(
                    frame, i, lane_w,
                    live_counts[i], live_scores[i],
                    sig_col, status_txt, night_mode,
                    ctrl.emergency_lane, active_evehicle,
                    vqueue.total_cleared[i], vqueue.rows_waiting[i]
                )
                draw_vehicles(
                    frame, i, live_counts[i], lane_w,
                    departed[i], arrived[i],
                    has_emergency_vehicle=lane_has_emerg,
                    evehicle=active_evehicle
                )

            draw_queue_bars(frame, lane_w, live_scores)

            # Emergency banner
            if ctrl.emergency_active and emergency_reason:
                draw_emergency_banner(frame, emergency_reason, active_evehicle)

            # Night overlay
            if night_mode:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0,0), (WIN_W, WIN_H), (0,0,30), -1)
                cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
                cv2.putText(frame, "NIGHT MODE — Reduced Arrivals",
                            (WIN_W//2 - 130, WIN_H - 98),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0,200,255), 1, cv2.LINE_AA)

            draw_bottom_bar(
                frame, row["timestamp"], row_idx, len(rows),
                speed, night_mode, ctrl.emergency_active,
                ctrl.green_lane, remaining, ctrl.green_duration, paused
            )

        cv2.imshow("Smart Traffic — Live Queue Simulation", frame)

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
            row_idx, ctrl, vqueue = 0, *fresh_state()
            departed, arrived = empty_flow(), empty_flow()
            emergency_reason, active_evehicle = "", ""
            last_logged_row, paused = -1, False
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
    p = argparse.ArgumentParser(description="Smart Traffic Live Queue Simulation")
    p.add_argument("--dataset", default="traffic_dataset.csv")
    p.add_argument("--speed",   type=float, default=0.8)
    p.add_argument("--loop",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.dataset):
        print(f"[ERROR] Dataset not found: '{args.dataset}'")
        print("  Required columns: timestamp, lane1_cars ... lane4_trucks,")
        print("                    emergency_lane, emergency_vehicle")
    else:
        run_simulation(args.dataset, speed=args.speed, loop=args.loop)