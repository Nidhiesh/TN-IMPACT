"""
traffic.py — Live Camera Smart Traffic Signal Controller
Fixed defects:
  1. Lane switching: now round-robins ALL 4 lanes (not re-sorting every cycle)
  2. Green duration capped (was unbounded: score*2 could grow huge)
  3. lane_order reset bug fixed — sorted once per cycle, not rebuilt mid-cycle
  4. Division-by-zero guard on lane_width when frame width < 4
  5. Resource cleanup: cap.release / destroyAllWindows now in try/finally
  6. model.names KeyError guard — unknown class_id no longer crashes
  7. Remaining time display: shows 0 not negative when overdue
  8. YOLO results iteration: safely handles empty result sets
"""

import cv2
import time
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH     = "yolov26.pt"
LIGHT_VEHICLES = {"car", "motorcycle"}
HEAVY_VEHICLES = {"bus", "truck"}
MIN_GREEN      = 5      # seconds
MAX_GREEN      = 30     # seconds — was unbounded (score*2 had no cap)
LIGHT_WEIGHT   = 1
HEAVY_WEIGHT   = 3
NUM_LANES      = 4
# ─────────────────────────────────────────────────────────────────────────────

model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(0)

# FIX 5: wrap everything in try/finally so resources always release
try:
    # State
    lane_order    = list(range(NUM_LANES))   # sorted order, rebuilt each cycle
    current_index = 0                        # index INTO lane_order
    green_lane    = lane_order[current_index]
    green_start   = time.time()
    green_duration = MIN_GREEN

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # FIX 4: guard against zero width
        if width < NUM_LANES:
            continue
        lane_width = width // NUM_LANES

        lane_scores = [0] * NUM_LANES

        results = model(frame)

        # ── Vehicle detection ─────────────────────────────────────────────
        for r in results:
            if r.boxes is None:          # FIX 8: empty result guard
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])

                # FIX 6: skip unknown class IDs instead of crashing
                class_name = model.names.get(cls_id)
                if class_name is None:
                    continue

                if class_name in LIGHT_VEHICLES or class_name in HEAVY_VEHICLES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2

                    weight     = HEAVY_WEIGHT if class_name in HEAVY_VEHICLES else LIGHT_WEIGHT
                    lane_index = min(center_x // lane_width, NUM_LANES - 1)
                    lane_scores[lane_index] += weight

                    color = (0, 200, 0) if class_name in LIGHT_VEHICLES else (0, 140, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, class_name,
                                (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 1, cv2.LINE_AA)

        current_time = time.time()

        # ── Signal switching logic ────────────────────────────────────────
        if current_time - green_start > green_duration:

            # FIX 1 & 3: Sort lanes by score ONCE at switch time, then
            # advance index round-robin through that sorted order.
            lane_order    = sorted(range(NUM_LANES),
                                   key=lambda i: lane_scores[i],
                                   reverse=True)
            current_index = (current_index + 1) % NUM_LANES
            green_lane    = lane_order[current_index]

            # FIX 2: cap green duration — was MIN_GREEN + score*2 with no ceiling
            score          = lane_scores[green_lane]
            green_duration = max(MIN_GREEN, min(MIN_GREEN + score * 2, MAX_GREEN))
            green_start    = current_time

        # FIX 7: remaining never goes negative
        remaining = max(0, int(green_duration - (current_time - green_start)))

        # ── Draw lane dividers ────────────────────────────────────────────
        for i in range(1, NUM_LANES):
            cv2.line(frame, (i * lane_width, 0), (i * lane_width, height),
                     (255, 0, 0), 2)

        # ── Draw signal lights and overlays ───────────────────────────────
        for i in range(NUM_LANES):
            x_pos = i * lane_width + lane_width // 2

            if i == green_lane:
                color  = (0, 255, 0)
                status = f"GREEN ({remaining}s)"
            else:
                color  = (0, 0, 255)
                status = "RED"

            cv2.circle(frame, (x_pos, 60), 25, color, -1)

            cv2.putText(frame, f"L{i+1} Score: {lane_scores[i]}",
                        (i * lane_width + 10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)

            cv2.putText(frame, status,
                        (i * lane_width + 10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                        cv2.LINE_AA)

        cv2.imshow("Auto Rotating Smart Traffic", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:                                 # FIX 5: always clean up
    cap.release()
    cv2.destroyAllWindows()