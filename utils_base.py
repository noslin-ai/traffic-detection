import cv2
import queue
import time
from rknnlite.api import RKNNLite

traffic_set = {"car", "bus", "truck"}
skip_cls = {"bicycle", "motorcycle"}
person_tag = "person"
img_size = (640, 640)

color_map = {
    'car': (0, 255, 0), 'bus': (0, 128, 255), 'truck': (255, 128, 0),
    'person': (255, 255, 255), 'bicycle': (0, 255, 255), 'motorcycle': (255, 0, 255)
}

def put_txt(img, txt, x, y, c):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)

def draw_box(img, x1, y1, x2, y2, c):
    cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)

def make_tid(x1, y1, x2, y2, l):
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return f"{l}_{cx // 50}_{cy // 50}"

def load_rknn(path, core=RKNNLite.NPU_CORE_0):
    m = RKNNLite()
    try:
        ret = m.load_rknn(path)
        if ret != 0:
            raise RuntimeError(f'Cannot load RKNN model {path}: code {ret}')
        ret = m.init_runtime(core_mask=core)
        if ret != 0:
            raise RuntimeError(f'Cannot initialize RKNN model {path} on core {core}: code {ret}')
    except BaseException:
        m.release()
        raise
    return m

def load_rknn_multi(path, core_list):
    res = []
    try:
        for c in core_list:
            res.append(load_rknn(path, c))
    except BaseException:
        for m in res:
            m.release()
        raise
    return res

def render_main(q_yolo, q_car_in, q_person_in, q_car_out, q_person_out,
                coco, post_process, run_car_attr, run_person_attr,
                traffic_set, person_tag, color_map, make_tid,
                put_txt, draw_box, img_size, attr_cache, attr_gap,
                frame_id, q_show, stop_event=None):
    fps, cnt, t0 = 0.0, 0, time.time()
    while True:
        if stop_event is not None and stop_event.is_set():
            break
        try:
            data = q_yolo.get(timeout=0.1)
        except queue.Empty:
            continue
        if data is None:
            while stop_event is None or not stop_event.is_set():
                try:
                    q_show.put(None, timeout=0.1)
                    break
                except queue.Full:
                    pass
            break
        frm, outs, t_yolo = data
        ih, iw = frm.shape[:2]
        dets = []
        boxes, ids, scrs = post_process(outs)
        if boxes is not None:
            boxes[:, [0, 2]] *= iw / img_size[0]
            boxes[:, [1, 3]] *= ih / img_size[1]
            for box, cid, sc in zip(boxes, ids, scrs):
                lbl = coco[cid]
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw - 1, x2), min(ih - 1, y2)
                tid = make_tid(x1, y1, x2, y2, lbl)
                dets.append((x1, y1, x2, y2, lbl, sc, tid))
                if lbl in traffic_set | {person_tag}:
                    last = attr_cache.get(tid)
                    if (not last) or frame_id['cur'] - last[1] >= attr_gap:
                        crop = frm[y1:y2, x1:x2].copy()
                        if crop.size:
                            (q_car_in if lbl in traffic_set else q_person_in).put(
                                (x1, y1, x2, y2, lbl, sc, crop, tid)
                            )
        while not q_car_out.empty():
            q_car_out.get()
        while not q_person_out.empty():
            q_person_out.get()
        for x1, y1, x2, y2, lbl, sc, tid in dets:
            col = color_map.get(lbl.lower(), (0, 255, 0))
            draw_box(frm, x1, y1, x2, y2, col)
            put_txt(frm, f"{lbl}:{sc:.2f}", x1, y1 - 5, col)
            attr = attr_cache.get(tid, (None, None))[0]
            if attr is None:
                continue
            if isinstance(attr, list):
                oy = y1 + 15
                for line in attr:
                    put_txt(frm, line, x1, oy, col)
                    oy += 15
            else:
                put_txt(frm, f"| {attr}", x1, y1 + 15, col)
        cnt += 1
        if time.time() - t0 >= 1:
            fps, cnt, t0 = cnt / (time.time() - t0), 0, time.time()
        if not q_show.full():
            q_show.put(frm.copy())
        frame_id['cur'] += 1
