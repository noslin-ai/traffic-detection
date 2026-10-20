import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
import urllib.parse  # 用于解析 URL 参数

_frame_lock = threading.Lock()
_cur_frame = None

_txt_lock = threading.Lock()
_cur_text = ""

_info_lock = threading.Lock()
_car_info = ""

# ================= 新增：控制状态锁和变量 =================
_control_lock = threading.Lock()
_control_state = {'paused': False, 'ff': False}

def get_control_state():
    with _control_lock:
        return dict(_control_state)

def reset_ff():
    with _control_lock:
        _control_state['ff'] = False
# ========================================================

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
        # ================= 新增：前端网页主页 =================
        if self.path == '/':
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>RK3588 交通检测</title>
                <style>
                    body { text-align: center; background-color: #222; color: white; font-family: sans-serif; margin: 0; padding: 0; overflow: hidden;}
                    .header { padding: 10px; }

                    /* 容器，用于隐藏放大后溢出边界的内容 */
                    .video-container {
                        margin-top: 10px;
                        display: inline-block;
                        position: relative;
                        overflow: hidden;
                        border: 2px solid #555;
                        border-radius: 8px;
                        background: #000;
                    }

                    /* 图片设置为块级，避免底部留白 */
                    img {
                        display: block;
                        max-width: 80vw;
                        max-height: 70vh;
                        cursor: zoom-in; /* 放大镜图标 */
                        transition: transform 0.2s ease-out; /* 平滑缩放动画 */
                        user-select: none;
                        -webkit-user-drag: none; /* 防止浏览器默认拖拽图片行为 */
                    }

                    /* 放大后的鼠标样式 */
                    img.zoomed {
                        cursor: grab;
                    }
                    img.zoomed:active {
                        cursor: grabbing;
                    }

                    .controls { margin-top: 20px; padding-bottom: 20px; }
                    button { padding: 12px 24px; font-size: 16px; margin: 0 10px; cursor: pointer;
                             background: #007bff; color: white; border: none; border-radius: 5px; transition: 0.2s; }
                    button:hover { background: #0056b3; }
                    button:active { transform: scale(0.95); }
                </style>
            </head>
            <body>
                <div class="header">
                    <p>RK3588 实时检测画面</p>
                </div>

                <!-- 包裹图片的容器 -->
                <div class="video-container" id="container">
                    <img id="videoFeed" src="/video_feed" />
                </div>

                <div class="controls">
                    <button onclick="sendCmd('pause')">暂停 / 继续</button>
                    <button onclick="sendCmd('ff')">快进 50 帧</button>
                </div>

                <script>
                    // 1. 发送控制指令（保留原有功能）
                    function sendCmd(action) {
                        fetch('/control?action=' + action)
                            .then(r => r.text())
                            .then(t => console.log('Response:', t))
                            .catch(e => console.error('Error:', e));
                    }

                    // 2. 双击放大镜与拖拽逻辑
                    const img = document.getElementById('videoFeed');
                    const container = document.getElementById('container');
                    let isZoomed = false;
                    let scale = 1;
                    const zoomScale = 3.0; // 放大倍数，可以自己改（比如 2.5 或 4）

                    // 拖拽相关变量
                    let isDragging = false;
                    let startX, startY;
                    let translateX = 0, translateY = 0;
                    let currentTranslateX = 0, currentTranslateY = 0;

                    // 双击事件
                    container.addEventListener('dblclick', function(e) {
                        if (!isZoomed) {
                            // 进入放大状态
                            // 获取鼠标点击位置相对于图片的百分比
                            const rect = img.getBoundingClientRect();
                            const xPercent = ((e.clientX - rect.left) / rect.width) * 100;
                            const yPercent = ((e.clientY - rect.top) / rect.height) * 100;

                            // 设置缩放中心为点击点
                            img.style.transformOrigin = `${xPercent}% ${yPercent}%`;
                            img.style.transform = `scale(${zoomScale}) translate(0px, 0px)`;

                            // 重置平移参数
                            translateX = 0; translateY = 0;
                            currentTranslateX = 0; currentTranslateY = 0;

                            img.classList.add('zoomed');
                            isZoomed = true;
                        } else {
                            // 恢复原状
                            img.style.transformOrigin = 'center center';
                            img.style.transform = 'scale(1) translate(0px, 0px)';
                            img.classList.remove('zoomed');
                            isZoomed = false;
                        }
                    });

                    // 拖拽平移事件 (仅在放大状态下有效)
                    img.addEventListener('mousedown', function(e) {
                        if (!isZoomed) return;
                        isDragging = true;
                        startX = e.clientX - currentTranslateX;
                        startY = e.clientY - currentTranslateY;
                        // 拖拽时取消过渡动画，让移动更跟手
                        img.style.transition = 'none';
                    });

                    img.addEventListener('mousemove', function(e) {
                        if (!isDragging) return;
                        e.preventDefault();

                        translateX = e.clientX - startX;
                        translateY = e.clientY - startY;

                        // 限制拖拽边界（防止把图片完全拖飞）
                        const maxDragX = (img.clientWidth * (zoomScale - 1)) / 2;
                        const maxDragY = (img.clientHeight * (zoomScale - 1)) / 2;

                        if (translateX > maxDragX) translateX = maxDragX;
                        if (translateX < -maxDragX) translateX = -maxDragX;
                        if (translateY > maxDragY) translateY = maxDragY;
                        if (translateY < -maxDragY) translateY = -maxDragY;

                        // 应用平移
                        // 注意：因为我们设置了 transform-origin，translate 是在缩放的基础坐标系里平移的
                        // 这里用 scale 乘以 translate 来实现相对屏幕的拖拽效果
                        img.style.transform = `scale(${zoomScale}) translate(${translateX / zoomScale}px, ${translateY / zoomScale}px)`;
                    });

                    img.addEventListener('mouseup', function() {
                        if (isDragging) {
                            isDragging = false;
                            currentTranslateX = translateX;
                            currentTranslateY = translateY;
                            img.style.transition = 'transform 0.2s ease-out';
                        }
                    });

                    img.addEventListener('mouseleave', function() {
                        if (isDragging) {
                            isDragging = false;
                            currentTranslateX = translateX;
                            currentTranslateY = translateY;
                            img.style.transition = 'transform 0.2s ease-out';
                        }
                    });
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
            return
        # ====================================================

        # ================= 新增：接收控制指令 =================
        elif self.path.startswith('/control'):
            parsed_path = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed_path.query)
            action = params.get('action', [None])[0]

            with _control_lock:
                if action == 'pause':
                    _control_state['paused'] = not _control_state['paused']
                elif action == 'ff':
                    _control_state['ff'] = True

            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
            return
        # ====================================================

        elif self.path.startswith('/video_feed'):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            while True:
                with _frame_lock:
                    data = _cur_frame
                if data is None:
                    time.sleep(0.05)
                    continue
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
            # ... (保持你原有的代码不变) ...
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
            # ... (保持你原有的代码不变) ...
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
    print(f"stream started at http://<board_ip>:{port}")
    srv.serve_forever()

def start_stream_server(port=8080):
    server = ThreadingHTTPServer(('', port), StreamHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f'stream started at http://<board_ip>:{port}')
    return server
