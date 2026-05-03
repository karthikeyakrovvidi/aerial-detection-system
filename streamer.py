
import cv2
import socket
import struct
import argparse
import time
import numpy as np
from ultralytics import YOLO



MODEL_PATH   = "yolov8m.pt"
CONFIDENCE   = 0.3
IOU          = 0.4
GROUND_IP    = "127.0.0.1"   
GROUND_PORT  = 5005          
JPEG_QUALITY = 60            
MAX_UDP_SIZE = 65000        

TARGET_CLASSES = {
    0: "Human",
    2: "Car",
    7: "Truck",
}

CLASS_COLORS = {
    0: (50, 205, 50),    
    2: (0, 165, 255),    
    7: (255, 50, 50),    
}



def parse_args():
    parser = argparse.ArgumentParser(description="Drone Streamer — aerial_detection_system")
    parser.add_argument("--source", type=str, default="test_video.mp4",
                        help="Video source (file path or 0 for webcam)")
    parser.add_argument("--ip", type=str, default=GROUND_IP,
                        help=f"Ground station IP (default: {GROUND_IP})")
    parser.add_argument("--port", type=int, default=GROUND_PORT,
                        help=f"UDP port (default: {GROUND_PORT})")
    parser.add_argument("--conf", type=float, default=CONFIDENCE,
                        help=f"Confidence threshold (default: {CONFIDENCE})")
    parser.add_argument("--no-detect", action="store_true",
                        help="Stream raw video without detection (for testing connection only)")
    return parser.parse_args()




def draw_detections(frame, results):
    counts = {0: 0, 2: 0, 7: 0}

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id not in TARGET_CLASSES:
                continue

            label = TARGET_CLASSES[class_id]
            color = CLASS_COLORS[class_id]
            counts[class_id] += 1

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




def draw_drone_hud(frame, fps, counts, frame_number):
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

   
    cv2.rectangle(frame, (w - 110, 8), (w - 8, 35), (30, 30, 180), -1)
    cv2.putText(frame, "DRONE TX", (w - 105, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame




def send_frame_udp(sock, frame, address):
    """
    Encodes frame as JPEG and sends over UDP.
    Splits into chunks if frame is too large for one UDP packet.
    Each chunk has a small header: [frame_id (4 bytes)] [chunk_index (2 bytes)] [total_chunks (2 bytes)]
    """
    
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    _, buffer = cv2.imencode('.jpg', frame, encode_params)
    data = buffer.tobytes()

    # Split into chunks
    chunks = [data[i:i + MAX_UDP_SIZE] for i in range(0, len(data), MAX_UDP_SIZE)]
    total_chunks = len(chunks)

    # Use timestamp as frame ID (4 bytes)
    frame_id = int(time.time() * 1000) % (2**32)

    for idx, chunk in enumerate(chunks):
      
        header = struct.pack("IHH", frame_id, idx, total_chunks)
        packet = header + chunk
        try:
            sock.sendto(packet, address)
        except Exception as e:
            print(f"[WARN] UDP send error: {e}")




def main():
    args = parse_args()

    
    if not args.no_detect:
        print(f"[DRONE] Loading model: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print(f"[DRONE] Model ready.")
    else:
        model = None
        print(f"[DRONE] Detection disabled — streaming raw video only.")

    
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[DRONE] Video opened: {w}x{h}")

    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)  
    ground_address = (args.ip, args.port)
    print(f"[DRONE] Streaming to {args.ip}:{args.port}")
    print(f"[DRONE] Make sure receiver.py is running in another terminal!")
    print(f"[DRONE] Press Q to stop streaming.")

    frame_number = 0
    fps = 0.0
    fps_timer = time.time()
    fps_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[DRONE] Video ended. Looping...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  
            continue

        frame_number += 1
        fps_counter += 1

        counts = {0: 0, 2: 0, 7: 0}

        
        if model is not None:
            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=IOU,
                classes=list(TARGET_CLASSES.keys()),
                verbose=False
            )
            frame, counts = draw_detections(frame, results)

       
        if fps_counter >= 10:
            fps = fps_counter / (time.time() - fps_timer)
            fps_timer = time.time()
            fps_counter = 0

       
        frame = draw_drone_hud(frame, fps, counts, frame_number)

        
        send_frame_udp(sock, frame, ground_address)

       
        cv2.imshow("Drone Side — Transmitting", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[DRONE] Stopped.")
            break

    cap.release()
    sock.close()
    cv2.destroyAllWindows()
    print("[DRONE] Streamer shut down.")


if __name__ == "__main__":
    main()
