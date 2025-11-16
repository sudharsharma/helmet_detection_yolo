import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import threading
import os
import sys

# Add path for importing the audio service
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from service.audio import play_beep   # <-- make sure this file exists

# Load YOLO model
model = YOLO("model/best.pt")

st.title("Helmet Detection App")

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# User chooses input
input_type = st.radio("Select Input Type:", ("Image", "Video", "Webcam"))
CONF_THRESHOLD = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# ---------------------- IMAGE ----------------------
if input_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
        annotated_frame = results[0].plot()

        # Beep when helmet detected (assuming class ID 1 = helmet)
        detected_classes = [int(cls) for cls in results[0].boxes.cls]
        if 1 in detected_classes:
            play_beep()

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st.image(annotated_frame, caption="Detection Result", use_column_width=True)

# ---------------------- VIDEO ----------------------
elif input_type == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
    if uploaded_file is not None:
        tfile = os.path.join("outputs", "temp_video.mp4")
        with open(tfile, "wb") as f:
            f.write(uploaded_file.read())
        st.video(tfile)
        st.info("Video detection works best in local OpenCV GUI, not Streamlit.")

# ---------------------- WEBCAM ----------------------
elif input_type == "Webcam":
    st.warning("Real-time webcam detection will open in a separate OpenCV window.")
    
    if st.button("Start Webcam Detection"):
        def run_webcam():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                st.error("Cannot access webcam.")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
                annotated = results[0].plot()

                # Check for helmet class (change ID if needed)
                detected_classes = [int(cls) for cls in results[0].boxes.cls]
                if 1 in detected_classes:
                    play_beep()

                cv2.imshow("Real-time Helmet Detection", annotated)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        threading.Thread(target=run_webcam).start()
        st.info("Press 'q' in the OpenCV window to stop the webcam.")
