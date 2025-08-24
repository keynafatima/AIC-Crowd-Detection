import cv2
import torch

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Open video / webcam
cap = cv2.VideoCapture(0)  # ganti dengan path video kalau mau

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Filter hanya orang (class=0 untuk COCO "person")
    persons = results.pred[0][results.pred[0][:, -1] == 0]
    count = len(persons)

    # Tentukan crowded level
    if count < 5:
        status = "LOW"
        color = (0, 255, 0)   # hijau
    elif count < 15:
        status = "MEDIUM"
        color = (0, 255, 255) # kuning
    else:
        status = "HIGH"
        color = (0, 0, 255)   # merah

    # Tampilkan bounding box
    for *xyxy, conf, cls in persons:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Tampilkan teks indikator
    cv2.putText(frame, f"Crowd: {count} ({status})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Crowd Detection YOLO", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
