import cv2
import numpy as np
from rknnlite.api import RKNNLite

COLOR_LABELS = ["yellow", "orange", "green", "gray", "red", "blue",
                "white", "golden", "brown", "black"]

TYPE_LABELS = ["sedan", "suv", "van", "hatchback", "mpv",
               "pickup", "bus", "truck", "estate"]

PERSON_ATTR_LABELS = [
    "male", "female", "age<18", "18–60", "age>60",
    "front", "back", "side", "no accessory", "glasses", "hat",
    "handheld", "backpack", "single shoulder bag", "handbag",
    "striped upper", "logo upper", "plaid upper", "spliced upper",
    "striped lower", "patterned lower",
    "short sleeve", "long sleeve", "long coat",
    "short trousers", "long trousers", "skirt",
]

def split_text_lines(text, per_line=2):
    parts = text.split(', ')
    return [', '.join(parts[i:i+per_line]) for i in range(0, len(parts), per_line)]

def preprocess_attr_input(crop):
    crop = cv2.resize(crop, (192, 256))
    return np.expand_dims(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), 0).astype(np.uint8)

def run_vehicle_attr(model, crop):
    inp = preprocess_attr_input(crop)
    res = model.inference(inputs=[inp])[0][0]
    color_id = int(np.argmax(res[:10]))
    type_id = int(np.argmax(res[10:]))
    return f'{COLOR_LABELS[color_id]}({res[color_id]:.2f}) {TYPE_LABELS[type_id]}({res[10+type_id]:.2f})'

def run_person_attr(model, crop, thr=0.6, topk=5):
    inp = preprocess_attr_input(crop)
    out = model.inference(inputs=[inp])[0][0]
    idxs = np.where(out > thr)[0]
    if len(idxs) == 0:
        idxs = [int(np.argmax(out))]
    idxs = idxs[:topk]
    attrs = [f'{PERSON_ATTR_LABELS[i]} ({out[i]:.2f})' for i in idxs]
    return split_text_lines(', '.join(attrs), per_line=2)
