import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2

_frame_lock = threading.Lock()
_cur_frame = None

_txt_lock = threading.Lock()
_cur_text = ""

_info_lock = threading.Lock()
_car_info = ""

def update_frame(frame):
    global _cur_frame
    ok, enc = cv2.imencode('.jpg', frame)
    if not ok:
        return
    with _frame_lock:
        _cur_frame = enc.tobytes()

def update_text(text):
    global _cur_text
    with _txt_lock:
        _cur_text = text

def update_info(data):
    global _car_info
    with _info_lock:
        _car_info = data

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/video_feed'):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--frame')
            self.end_headers()
            while True:
                with _frame_lock:
                    if _cur_frame is None:
                        continue
                    data = _cur_frame
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.05)
                except Exception:
                    break
        elif self.path.startswith('/deepseek_text'):
            with _txt_lock:
                text = _cur_text or ""
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            data = text.encode("utf-8")
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path.startswith('/vehicle_info'):
            with _info_lock:
                txt = _car_info or ""
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            data = txt.encode("utf-8")
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)
            self.end_headers()

def run_server(port=8080):
    addr = ('', port)
    srv = ThreadingHTTPServer(addr, StreamHandler)
    print(f"stream started at http://<board_ip>:{port}/video_feed")
    srv.serve_forever()

def start_stream_server(port=8080):
    t = threading.Thread(target=run_server, args=(port,), daemon=True)
    t.start()