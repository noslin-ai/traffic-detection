import numpy as np

obj_th = 0.25
nms_th = 0.45
img_sz = (640, 640)

def dfl_calc(x):
    x = x.astype(np.float32)
    n, c, h, w = x.shape
    p = 4
    m = c // p
    y = x.reshape(n, p, m, h, w)
    y = np.exp(y - np.max(y, axis=2, keepdims=True))
    y = y / np.sum(y, axis=2, keepdims=True)
    acc = np.arange(m, dtype=np.float32).reshape(1, 1, m, 1, 1)
    return (y * acc).sum(axis=2)

def make_boxes(pred):
    gh, gw = pred.shape[2:4]
    col, row = np.meshgrid(np.arange(gw), np.arange(gh))
    grid = np.concatenate((col.reshape(1, 1, gh, gw), row.reshape(1, 1, gh, gw)), axis=1)
    stride = np.array([img_sz[1] // gw, img_sz[0] // gh]).reshape(1, 2, 1, 1)
    pos = dfl_calc(pred)
    tl = grid + 0.5 - pos[:, 0:2, :, :]
    br = grid + 0.5 + pos[:, 2:4, :, :]
    xyxy = np.concatenate((tl * stride, br * stride), axis=1)
    return xyxy

def pick_boxes(boxes, b_conf, c_conf, th=obj_th):
    b_conf = b_conf.reshape(-1)
    c_max = np.max(c_conf, axis=-1)
    c_idx = np.argmax(c_conf, axis=-1)
    keep = np.where(c_max * b_conf >= th)
    return boxes[keep], c_idx[keep], (c_max * b_conf)[keep]

def nms(boxes, scores, th=nms_th):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        inter = w * h
        iou = inter / (area[i] + area[order[1:]] - inter)
        order = order[1:][iou <= th]
    return keep

def post_process(outs):
    br = 3
    step = len(outs) // br
    box_all, cls_all, scr_all = [], [], []
    for i in range(br):
        box_all.append(make_boxes(outs[step * i]))
        cls_all.append(outs[step * i + 1])
        scr_all.append(np.ones_like(outs[step * i + 1][:, :1, :, :], dtype=np.float32))

    def flat(x):
        ch = x.shape[1]
        return x.transpose(0, 2, 3, 1).reshape(-1, ch)

    boxes = np.concatenate([flat(b) for b in box_all])
    cls_p = np.concatenate([flat(c) for c in cls_all])
    scrs = np.concatenate([flat(s) for s in scr_all])

    boxes, classes, scores = pick_boxes(boxes, scrs, cls_p)
    if boxes.shape[0] == 0:
        return None, None, None
    keep = nms(boxes, scores)
    return boxes[keep], classes[keep], scores[keep]