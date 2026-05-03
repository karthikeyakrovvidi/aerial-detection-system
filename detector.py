import cv2
import argparse
import time
from ultralytics import YOLO



MODEL_PATH = "yolov8m.pt"          
CONFIDENCE = 0.25                  
IOU_THRESHOLD = 0.4


TARGET_CLASSES = {
    0: "Human",
    2: "Car",
    7: "Truck",
}

CLASS_COLORS = {
    0: (50, 205, 50),     # Human  → green
    2: (0, 165, 255),     # Car    → orange
    7: (255, 50, 50),     # Truck  → blue
}



def parse_args():
    parser = argparse.ArgumentParser(description="Aerial Object Detector — YOLOv8m")
    parser.add_argument("--source", type=str, default="test_video.mp4",
                        help="Path to video file (default: test_video.mp4)")
    parser.add_argument("--conf", type=float, default=CONFIDENCE,
                        help=f"Confidence threshold (default: {CONFIDENCE})")
    parser.add_argument("--show-all", action="store_true",
                        help="Show ALL detections for debugging")
    return parser.parse_args()




def draw_detections(frame, results, show_all=False):
    counts = {0: 0, 2: 0, 7: 0}

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if not show_all and class_id not in TARGET_CLASSES:
                continue

            if class_id in TARGET_CLASSES:
                label = TARGET_CLASSES[class_id]
                color = CLASS_COLORS[class_id]
                counts[class_id] += 1
            else:
                label = result.names[class_id]
                color = (180, 180, 180)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_text = f"{label} {confidence:.0%}"
            (text_w, text_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            cv2.rectangle(frame,
                          (x1, y1 - text_h - 10),
                          (x1 + text_w + 8, y1),
                          color, -1)

            cv2.putText(frame, label_text,
                        (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return frame, counts




def draw_hud(frame, fps, counts, frame_number):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (270, 135), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "AERIAL DETECTION SYSTEM", (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(frame, f"FPS     : {fps:.1f}", (18, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Frame   : {frame_number}", (18, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Humans  : {counts[0]}", (18, 97),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 205, 50), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Cars    : {counts[2]}", (18, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 165, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Trucks  : {counts[7]}", (18, 133),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 50, 50), 1, cv2.LINE_AA)

    cv2.circle(frame, (w - 20, 22), 8, (0, 0, 255), -1)
    cv2.putText(frame, "LIVE", (w - 60, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 1, cv2.LINE_AA)

    return frame



def main():
    args = parse_args()

    print(f"[INFO] Loading model: {MODEL_PATH}")
    print(f"[INFO] Downloading yolov8m.pt if not cached (~52MB, one-time only)...")
    model = YOLO(MODEL_PATH)
    print(f"[INFO] Model ready. Detecting: {list(TARGET_CLASSES.values())}")
    print(f"[INFO] Confidence threshold: {args.conf}")

    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    print(f"[INFO] Opening: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[INFO] Video: {w}x{h} @ {orig_fps:.1f} FPS")
    print("[INFO] Controls: Q = quit | S = screenshot")

    frame_number = 0
    fps = 0.0
    fps_timer = time.time()
    fps_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video ended.")
            break

        frame_number += 1
        fps_counter += 1

        classes_filter = None if args.show_all else list(TARGET_CLASSES.keys())

        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=IOU_THRESHOLD,
            classes=classes_filter,
            verbose=False
        )

        frame, counts = draw_detections(frame, results, show_all=args.show_all)

        if fps_counter >= 10:
            fps = fps_counter / (time.time() - fps_timer)
            fps_timer = time.time()
            fps_counter = 0

        frame = draw_hud(frame, fps, counts, frame_number)

        cv2.imshow("Aerial Detection System — Ground View", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Quit.")
            break
        elif key == ord('s'):
            name = f"screenshot_frame_{frame_number}.jpg"
            cv2.imwrite(name, frame)
            print(f"[INFO] Screenshot saved: {name}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
