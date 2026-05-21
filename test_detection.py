from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\AL-OFQ\runs\detect\sperm_multiclass-2\weights\best.pt")

VIDEO_PATH = r"C:\Users\AL-OFQ\Downloads\spai_data\data\raw\Dataset\BM 210682\Sessione 120925\BM 210682 (1).mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

results = model(frame, conf=0.1, imgsz=1024, device=0)

for r in results:
    boxes = r.boxes
    print(f"عدد الكشوفات: {len(boxes)}")
    for box in boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        print(f"  class: {cls} | conf: {conf:.2f}")