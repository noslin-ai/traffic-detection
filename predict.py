import cv2
import numpy as np
from rknnlite.api import RKNNLite
import os

color_list = ["yellow", "orange", "green", "gray", "red", "blue", "white", "golden", "brown", "black"]
car_types = ["sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus", "truck", "estate"]

model_file = '/userdata/aidemo/tzb/classifaction/detection/vehicle_attr.rknn'
img_file = '/userdata/aidemo/tzb/classifaction/car_00014.jpg'
save_path = './output.jpg'

print("init rknn lite...")
rknn = RKNNLite()
if rknn.load_rknn(model_file) != 0:
    print("load model fail")
    exit()
if rknn.init_runtime() != 0:
    print("runtime init fail")
    exit()
print("runtime ok")

frame = cv2.imread(img_file)
if frame is None:
    print(f"read img fail: {img_file}")
    exit()

w, h = 192, 256
resized = cv2.resize(frame, (w, h))
img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
img = np.expand_dims(img, axis=0).astype(np.uint8)

out = rknn.inference(inputs=[img])
res = out[0][0]

c_score = res[:10]
t_score = res[10:]
c_idx = int(np.argmax(c_score))
t_idx = int(np.argmax(t_score))

label = f"{color_list[c_idx]} ({c_score[c_idx]:.2f}), {car_types[t_idx]} ({t_score[t_idx]:.2f})"
print("result:", label)

cv2.rectangle(resized, (0, 0), (w, h), (0, 255, 0), 2)
cv2.putText(resized, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.imwrite(save_path, resized)
print(f"save to {save_path}")

rknn.release()