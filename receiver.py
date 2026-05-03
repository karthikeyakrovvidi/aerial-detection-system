import cv2
import socket
import struct
import time
import numpy as np



LISTEN_IP    = "0.0.0.0"   
LISTEN_PORT  = 5005         
BUFFER_SIZE  = 65536        
TIMEOUT_SEC  = 10           




def draw_ground_hud(frame, fps, frame_number, packets_received):
    h, w = frame.shape[:2]

    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 45), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, f"GROUND STATION  |  FPS: {fps:.1f}  |  Frame: {frame_number}  |  Packets: {packets_received}",
                (12, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

   
    cv2.rectangle(frame, (w - 110, 8), (w - 8, 35), (20, 120, 20), -1)
    cv2.putText(frame, "GROUND RX", (w - 105, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame




def show_no_signal(frame_shape=(480, 640, 3)):
    """Shows a black screen with NO SIGNAL text while waiting for stream."""
    frame = np.zeros(frame_shape, dtype=np.uint8)
    h, w = frame.shape[:2]

    cv2.putText(frame, "NO SIGNAL", (w // 2 - 120, h // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (60, 60, 60), 2, cv2.LINE_AA)

    cv2.putText(frame, "Waiting for drone stream...", (w // 2 - 160, h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1, cv2.LINE_AA)

    cv2.putText(frame, "Run: python streamer.py --source test_video.mp4",
                (w // 2 - 260, h // 2 + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)

    return frame




class FrameReassembler:
    """
    Reassembles frames split into UDP chunks by streamer.py.
    Matches chunks by frame_id, assembles when all chunks arrive.
    """
    def __init__(self):
        self.buffers = {}   
        self.totals  = {}   

    def add_packet(self, data):
        """
        Add a received UDP packet.
        Returns complete frame bytes if frame is complete, else None.
        """
        if len(data) < 8:
            return None  

        
        header = data[:8]
        chunk  = data[8:]
        frame_id, chunk_index, total_chunks = struct.unpack("IHH", header)

        
        if frame_id not in self.buffers:
            self.buffers[frame_id] = {}
            self.totals[frame_id]  = total_chunks

        self.buffers[frame_id][chunk_index] = chunk

        
        if len(self.buffers[frame_id]) == self.totals[frame_id]:
           
            frame_data = b"".join(
                self.buffers[frame_id][i]
                for i in range(self.totals[frame_id])
            )
           
            del self.buffers[frame_id]
            del self.totals[frame_id]

           
            if len(self.buffers) > 5:
                oldest = min(self.buffers.keys())
                del self.buffers[oldest]
                del self.totals[oldest]

            return frame_data

        return None




def main():
   
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)  
    sock.bind((LISTEN_IP, LISTEN_PORT))
    sock.settimeout(1.0)  

    print(f"[GROUND] Ground station listening on port {LISTEN_PORT}")
    print(f"[GROUND] Waiting for drone stream...")
    print(f"[GROUND] Press Q to quit.")

    reassembler    = FrameReassembler()
    frame_number   = 0
    packets_received = 0
    fps            = 0.0
    fps_timer      = time.time()
    fps_counter    = 0
    last_frame_time = time.time()

   
    cv2.imshow("Aerial Detection System — Ground View", show_no_signal())
    cv2.waitKey(1)

    while True:
      
        try:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            packets_received += 1
            last_frame_time = time.time()

            
            frame_bytes = reassembler.add_packet(data)

            if frame_bytes is not None:
               
                np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                frame_number += 1
                fps_counter  += 1

                
                if fps_counter >= 10:
                    fps       = fps_counter / (time.time() - fps_timer)
                    fps_timer = time.time()
                    fps_counter = 0

                
                frame = draw_ground_hud(frame, fps, frame_number, packets_received)

                
                cv2.imshow("Aerial Detection System — Ground View", frame)

        except socket.timeout:
            
            if time.time() - last_frame_time > TIMEOUT_SEC:
                cv2.imshow("Aerial Detection System — Ground View", show_no_signal())

        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[GROUND] Quit.")
            break
        elif key == ord('s'):
            if frame_number > 0:
                name = f"ground_screenshot_{frame_number}.jpg"
                cv2.imwrite(name, frame)
                print(f"[GROUND] Screenshot saved: {name}")

    sock.close()
    cv2.destroyAllWindows()
    print("[GROUND] Ground station shut down.")


if __name__ == "__main__":
    main()
