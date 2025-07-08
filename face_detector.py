import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

def main():
    def detect_face(image):
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return image

    st.title("Face Detection App")
    st.write("Upload an image to detect faces.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image.convert("RGB"))
        
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        if st.button("Detect Faces"):
            detected_image = detect_face(image)
            st.image(detected_image, caption='Detected Faces', use_container_width=True)
            st.success("Faces detected successfully!")
        else:
            st.write("Click the button to detect faces in the uploaded image.")
            
