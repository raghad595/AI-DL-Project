import cv2
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import streamlit as st
import numpy as np

def main():
    # load model
    @st.cache_resource
    def load_model():
        return YOLO("yolov8n.pt")


    model = load_model()

    st.title("Object Detection with YOLO")

    upload_file = st.file_uploader("Upload an image", type=["png", "jpg"])

    if upload_file is not None:
        image = Image.open(upload_file).convert("RGB")  # Ensure image is in RGB format
        image_np = np.array(image)

        results = model(image_np)[0]

        boxes = results.boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy().astype(int)

        class_names = [model.names[c] for c in class_ids]
        unique_classes = sorted(set(class_names))

        selected_classes = st.multiselect("Select classes to be showen", unique_classes, default=unique_classes)

        for box, cls_name, conf in zip(xyxy, class_names, confidences):
            if cls_name in selected_classes:
                x1, y1, x2, y2 = box
                label = f"{cls_name} {conf:.2f}"
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        st.image(image_np, caption="Detected Objects", use_container_width=True)
