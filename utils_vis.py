import cv2

def draw_box_fancy(img, x1, y1, x2, y2, color, thick=2):
    seg = 12
    cv2.line(img, (x1, y1), (x1 + seg, y1), color, thick)
    cv2.line(img, (x1, y1), (x1, y1 + seg), color, thick)
    cv2.line(img, (x2, y1), (x2 - seg, y1), color, thick)
    cv2.line(img, (x2, y1), (x2, y1 + seg), color, thick)
    cv2.line(img, (x1, y2), (x1 + seg, y2), color, thick)
    cv2.line(img, (x1, y2), (x1, y2 - seg), color, thick)
    cv2.line(img, (x2, y2), (x2 - seg, y2), color, thick)
    cv2.line(img, (x2, y2), (x2, y2 - seg), color, thick)

def draw_label(img, txt, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.5, 1
    (w, h), _ = cv2.getTextSize(txt, font, scale, thick)
    temp = img.copy()
    cv2.rectangle(temp, (x, y - h - 6), (x + w + 4, y + 2), color, -1)
    cv2.addWeighted(temp, 0.4, img, 0.6, 0, img)
    cv2.putText(img, txt, (x + 2, y - 2), font, scale, (255, 255, 255), thick, cv2.LINE_AA)