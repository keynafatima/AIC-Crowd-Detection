import os
import cv2
import numpy as np
from typing import Dict, Any
from ultralytics import YOLO  # pip install ultralytics

MODEL = YOLO("yolov8n.pt")  # versi ringan
OUT_DIR = "static"
os.makedirs(OUT_DIR, exist_ok=True)

def infer_on_image_path(image_path: str, out_dir: str = OUT_DIR) -> Dict[str, Any]:
    results = MODEL.predict(image_path, conf=0.4)
    res = results[0]

    person_boxes = [b for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()) if int(c) == 0]
    count = len(person_boxes)

    img = cv2.imread(image_path)
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, f"{base}_yolo.jpg")
    cv2.imwrite(out_path, img)

    return {
        "file": image_path,
        "count": count,
        "heatmap": out_path
    }

def infer_on_image_bytes(image_bytes: bytes, out_dir: str = OUT_DIR, name: str = "upload.jpg") -> Dict[str, Any]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")

    temp_path = os.path.join(out_dir, name)
    cv2.imwrite(temp_path, img)

    res = infer_on_image_path(temp_path, out_dir)

    result_img = cv2.imread(res["heatmap"])
    _, buffer = cv2.imencode(".jpg", result_img)
    res["image_bytes"] = buffer.tobytes()

    return res
