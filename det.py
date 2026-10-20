import argparse
import importlib
import queue
import threading
import time
from pathlib import Path
from utils_stream import start_stream_server, update_frame, get_control_state, reset_ff

PROJECT_DIR = Path(__file__).resolve().parent
yolo_file = PROJECT_DIR / 'yolo11.rknn'
car_attr_file = PROJECT_DIR / 'vehicle_attr.rknn'
person_attr_file = PROJECT_DIR / 'pphuman_attr.rknn'

q_raw = queue.Queue(maxsize=4)
q_yolo = queue.Queue(maxsize=4)
q_show = queue.Queue(maxsize=2)

car_in, car_out = queue.Queue(), queue.Queue()
person_in, person_out = queue.Queue(), queue.Queue()

car_models = []
person_model = None
attr_cache = {}
attr_interval = 5
stop_event = threading.Event()
worker_errors = queue.Queue()


def preflight(video):
    errors = []
    for path in (Path(video), yolo_file, car_attr_file, person_attr_file):
        if not path.is_file():
            errors.append(f'Missing file: {path}')
    for name in ('numpy', 'cv2', 'rknnlite.api'):
        try:
            importlib.import_module(name)
        except Exception as exc:
            errors.append(f'Cannot import {name}: {exc}')
    return errors


def send(q, value):
    while not stop_event.is_set():
        try:
            q.put(value, timeout=0.1)
            return
        except queue.Full:
            pass


def receive(q):
    while not stop_event.is_set():
        try:
            return q.get(timeout=0.1)
        except queue.Empty:
            pass
    return None


def guarded(target, *args):
    try:
        target(*args)
    except Exception as exc:
        worker_errors.put(exc)
        stop_event.set()

def read_frames(cap):
    while not stop_event.is_set():
        # 获取全局状态
        ctrl = get_control_state()

        # 1. 处理暂停
        if ctrl['paused']:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # 2. 处理快进（一次跳过 50 帧）
        if ctrl['ff']:
            curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame + 50)
            reset_ff() # 通知全局状态：快进已执行，重置标记
            continue

        # 3. 正常发送
        send(q_raw, frame)

    while not stop_event.is_set():
        # 处理暂停
        if stream_control['paused']:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # 处理快进（一次跳过 50 帧，你可以自己调）
        if stream_control['ff']:
            curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame + 50)
            stream_control['ff'] = False # 跳过一次后重置
            continue

        send(q_raw, frame)
def yolo_thread(model):
    while True:
        frame = receive(q_raw)
        if frame is None:
            send(q_yolo, None)
            break
        t0 = time.time()
        out = model.inference(inputs=[cv2.cvtColor(cv2.resize(frame, img_size), cv2.COLOR_BGR2RGB)[None]])
        send(q_yolo, (frame, out, (time.time() - t0) * 1000))

def car_thread(idx):
    mdl = car_models[idx]
    while True:
        item = receive(car_in)
        if item is None:
            break
        x1, y1, x2, y2, lbl, sc, crop, tid = item
        if crop.size:
            attr_cache[tid] = (run_vehicle_attr(mdl, crop), frame_id['cur'])
            car_out.put(1)

def person_thread():
    while True:
        item = receive(person_in)
        if item is None:
            break
        x1, y1, x2, y2, lbl, sc, crop, tid = item
        if crop.size:
            attr_cache[tid] = (run_person_attr(person_model, crop), frame_id['cur'])
            person_out.put(1)

def show_thread(headless=False):
    if not headless:
        cv2.namedWindow('Traffic', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Traffic', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        frame = receive(q_show)
        if frame is None:
            break
        update_frame(frame)
        if headless:
            continue
        cv2.imshow('Traffic', cv2.resize(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE), (800, 1280)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

def main():
    parser = argparse.ArgumentParser(description='Detect traffic objects and attributes in a video.')
    parser.add_argument('--video', nargs='?', default=str(PROJECT_DIR / 'video1.mp4'),
                        help='Input video path (default: video1.mp4 beside det.py)')
    parser.add_argument('--check', action='store_true', help='Check input files and imports without loading the NPU')
    parser.add_argument('--headless', action='store_true', help='Only publish HTTP video; do not open a GUI window')
    args = parser.parse_args()

    errors = preflight(args.video)
    if errors:
        parser.exit(1, '\n'.join(errors) + '\nRKNN inference requires a compatible Rockchip device and runtime.\n')
    if args.check:
        print('Files and imports OK. NPU initialization and model compatibility still require a real inference run.')
        return

    global cv2, RKNNLite, run_vehicle_attr, run_person_attr, img_size, update_frame
    import cv2
    from rknnlite.api import RKNNLite
    from coco_classes import COCO_CLASSES
    from yolo_postprocess import post_process
    from utils_attr import run_vehicle_attr, run_person_attr
    from utils_base import (
        traffic_set, person_tag, img_size, color_map,
        put_txt, draw_box, make_tid, load_rknn, load_rknn_multi, render_main,
    )
    from utils_stream import start_stream_server, update_frame

    global car_models, person_model, frame_id
    frame_id = {'cur': 0}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot open input video: {args.video}')

    yolo = None
    server = None
    workers = []
    try:
        yolo = load_rknn(str(yolo_file), RKNNLite.NPU_CORE_0)
        car_models = load_rknn_multi(str(car_attr_file), [RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2])
        person_model = load_rknn(str(person_attr_file), RKNNLite.NPU_CORE_2)
        server = start_stream_server(port=8080)
        jobs = [(read_frames, (cap,)), (yolo_thread, (yolo,)),
                (person_thread, ()), (show_thread, (args.headless,))]
        jobs.extend((car_thread, (i,)) for i in range(len(car_models)))
        for target, arguments in jobs:
            thread = threading.Thread(target=guarded, args=(target, *arguments), daemon=True)
            thread.start()
            workers.append(thread)
        render_main(
            q_yolo, car_in, person_in, car_out, person_out,
            COCO_CLASSES, post_process, run_vehicle_attr, run_person_attr,
            traffic_set, person_tag, color_map, make_tid,
            put_txt, draw_box, img_size, attr_cache, attr_interval,
            frame_id, q_show, stop_event=stop_event
        )
        # Let the display publish all queued frames before releasing resources.
        workers[3].join()
    finally:
        stop_event.set()
        for thread in workers:
            thread.join()
        cap.release()
        if yolo is not None:
            yolo.release()
        for m in car_models:
            m.release()
        if person_model:
            person_model.release()
        if server is not None:
            server.shutdown()
            server.server_close()
        if not args.headless:
            cv2.destroyAllWindows()
    if not worker_errors.empty():
        raise RuntimeError('Video processing worker failed') from worker_errors.get()


if __name__ == '__main__':
    main()
