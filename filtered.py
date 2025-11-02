import cv2
import torch

# Load your custom YOLOv5n model
model = torch.hub.load(
    r"C:\College_Projects\Yolo\yolov5", "custom", path=r"yolov5\runs\train\yolo-batch4\weights\best.pt", source="local"
)
model.conf = 0.25  # Confidence threshold

# Load video
# cap = cv2.VideoCapture("yolov5/../Videos/video4.mp4")
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

paused = False

# 👇 Add this block for full-screen display
cv2.namedWindow("People Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("People Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("✅ Video ended.")
            break

        results = model(frame)
        detections = results.xyxy[0]

        people_count = 0

        for det in detections:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if int(cls) == 0 and conf > 0.4:  # class 0 is usually "person"
                area = (x2 - x1) * (y2 - y1)
                if area > 1000:
                    people_count += 1
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Get frame dimensions for centering the text
        frame_height, frame_width = frame.shape[:2]
        cv2.putText(
            frame,
            f"People Count: {people_count}",
            (frame_width // 3, frame_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv2.imshow("People Detection", frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("p"):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
