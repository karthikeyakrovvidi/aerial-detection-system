# 🚁 Aerial Detection System

Real-time object detection and ground streaming pipeline for drone surveillance.

## 📌 What This Project Does
The drone captures live video, runs YOLOv8m AI detection on every frame,
and streams the annotated video wirelessly to a ground station in real time.
The ground operator sees humans, cars, and trucks correctly labelled with
confidence scores and live counts.

## 🎯 Detection Classes
| Object | Label | Box Color |
|--------|-------|-----------|
| Person | Human | 🟢 Green  |
| Car    | Car   | 🟠 Orange |
| Truck  | Truck | 🔴 Red    |

## 🗂️ Project Files
| File | Description |
|------|-------------|
| `detector.py` | Standalone detection — test YOLOv8 on any video |
| `streamer.py` | Drone side — detects and streams over UDP |
| `receiver.py` | Ground side — receives and displays live stream |

## ⚙️ Installation
```bash
pip install ultralytics opencv-python
```

## ▶️ How to Run

**Terminal 1 — Start ground station first:**
```bash
python receiver.py
```

**Terminal 2 — Start drone streamer:**
```bash
python streamer.py --source test_video.mp4
```

## 🛠️ Tech Stack
- YOLOv8m (Ultralytics)
- OpenCV
- Python UDP Sockets
- NumPy

## 📸 Live Test
![Live Test Screenshot](screenshot.png)
