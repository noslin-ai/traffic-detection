import cv2, time, threading, queue, numpy as np
from rknnlite.api import RKNNLite
from coco_classes import COCO_CLASSES
from yolo_postprocess import post_process
from utils_attr import run_vehicle_attr, run_person_attr
from utils_base import (
    TRAFFIC_SET, ATTR_SKIP_CLASSES, PERSON_LABEL, IMG_SIZE, label_color,
    put_txt, draw_box, make_tid, load_rknn_model, load_multiple_rknn_models,
    render_loop
)
from utils_stream import start_stream_server, update_frame

yolo_file = './yolo11.rknn'
car_attr_file = './vehicle_attr.rknn'
person_attr_file = './pphuman_attr.rknn'
video_file = './video1.mp4'

q_raw = queue.Queue(maxsize=4)
q_yolo = queue.Queue(maxsize=4)
q_show = queue.Queue(maxsize=2)

car_in, car_out = queue.Queue(), queue.Queue()
person_in, person_out = queue.Queue(), queue.Queue()

car_models = []
car_cores = [RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
person_model = None
attr_cache = {}
attr_interval = 5

def read_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            q_raw.put(None)
            break
        q_raw.put(frame)

def yolo_thread(model):
    while True:
        frame = q_raw.get()
        if frame is None:
            q_yolo.put(None)
            break
        t0 = time.time()
        out = model.inference(inputs=[cv2.cvtColor(cv2.resize(frame, IMG_SIZE), cv2.COLOR_BGR2RGB)[None]])
        q_yolo.put((frame, out, (time.time() - t0) * 1000))

def car_thread(idx):
    mdl = car_models[idx]
    while True:
        x1, y1, x2, y2, lbl, sc, crop, tid = car_in.get()
        if crop.size:
            attr_cache[tid] = (run_vehicle_attr(mdl, crop), frame_id['cur'])
            car_out.put(1)

def person_thread():
    while True:
        x1, y1, x2, y2, lbl, sc, crop, tid = person_in.get()
        if crop.size:
            attr_cache[tid] = (run_person_attr(person_model, crop), frame_id['cur'])
            person_out.put(1)

def show_thread():
    cv2.namedWindow('Traffic', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Traffic', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        frame = q_show.get()
        if frame is None:
            break
        update_frame(frame)
        cv2.imshow('Traffic', cv2.resize(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE), (800, 1280)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    frame_id = {'cur': 0}

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError("video open fail")

    yolo = load_rknn_model(yolo_file, RKNNLite.NPU_CORE_0)
    car_models = load_multiple_rknn_models(car_attr_file, car_cores)
    person_model = load_rknn_model(person_attr_file, RKNNLite.NPU_CORE_2)

    start_stream_server(port=8080)

    threading.Thread(target=read_frames, args=(cap,), daemon=True).start()
    threading.Thread(target=yolo_thread, args=(yolo,), daemon=True).start()
    for i in range(len(car_models)):
        threading.Thread(target=car_thread, args=(i,), daemon=True).start()
    threading.Thread(target=person_thread, daemon=True).start()
    threading.Thread(target=show_thread, daemon=True).start()

    try:
        render_loop(
            q_yolo, car_in, person_in, car_out, person_out,
            COCO_CLASSES, post_process, run_vehicle_attr, run_person_attr,
            TRAFFIC_SET, PERSON_LABEL, label_color, make_tid,
            put_txt, draw_box, IMG_SIZE, attr_cache, attr_interval,
            frame_id, q_show
        )
    finally:
        cap.release()
        yolo.release()
        for m in car_models:
            m.release()
        if person_model:
            person_model.release()