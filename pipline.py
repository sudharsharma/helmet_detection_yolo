import sys
import cv2
import os
from ultralytics import YOLO
from audio import play_beep
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.makedirs("outputs", exist_ok=True)

# Loading model
model = YOLO("model/best.pt")

# Detecting the helmet on the frame
def detect_helmet(frame, conf_threshold=0.25, beep_on_helmet=True):
    frame_resized = cv2.resize(frame, (640, 640))
    results = model.predict(source=frame_resized, conf=conf_threshold, verbose=False)
    annotated_frame = results[0].plot()
    detected_classes = [int(cls) for cls in results[0].boxes.cls]
    print("Detected classes:", detected_classes) 
    if beep_on_helmet and 1 in detected_classes:
        play_beep()
    return annotated_frame

def process_input(source=None, is_webcam=False):
    if is_webcam:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("[ERROR] Could not access webcam.")
            return
        out = None

    elif source.endswith((".mp4", ".avi")):  # for Video file
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {source}")
            return
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("outputs/output_video.mp4", fourcc, 20, (640, 640))

    else:  
        cap = None
        img = cv2.imread(source)
        if img is None:
            print(f"[ERROR] Could not read image: {source}")
            return
        annotated = detect_helmet(img)
        cv2.imshow("Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("outputs/result.jpg", annotated)
        print("[INFO] Saved result to outputs/result.jpg")
        return

    # for video or webcame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = detect_helmet(frame)
        if out: out.write(annotated)
        cv2.imshow("Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if cap: cap.release()
    if out: out.release()
    cv2.destroyAllWindows()
    if out: print("[INFO] Video saved to outputs/output_video.mp4")

if __name__ == "__main__":
    print("Select input type:")
    print("1 - Single image")
    print("2 - Video file")
    print("3 - Webcam")
    choice = input("Enter choice (1/2/3): ")
    if choice == "1":
        path = input("Enter image path: ")
        process_input(path)
    elif choice == "2":
        path = input("Enter video path: ")
        process_input(path)
    elif choice == "3":
        process_input(is_webcam=True)
    else:
        print("Invalid choice")


