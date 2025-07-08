import cv2
import streamlit as st
from PIL import Image
import numpy as np

def main():
    st.title("Filters Application")

    # Filters
    def blackwhite(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def pencil_sketch(img, ksize=5):
        blur = cv2.GaussianBlur(img, (ksize, ksize), 0, 0)
        sketch, _ = cv2.pencilSketch(blur)
        return sketch

    def HDR(img, level=50, sigma_s=10, sigma_r=0.1):
        bright = cv2.convertScaleAbs(img, beta=level)
        return cv2.detailEnhance(bright, sigma_s=sigma_s, sigma_r=sigma_r)

    def Brightness(img, level=50):
        return cv2.convertScaleAbs(img, beta=level)

    def style_image(img, ksize=5, sigma_s=10, sigma_r=0.1):
        blur = cv2.GaussianBlur(img, (ksize, ksize), 0, 0)
        return cv2.stylization(blur, sigma_s=sigma_s, sigma_r=sigma_r)

    # Upload
    upload = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if upload is not None:
        img = Image.open(upload).convert("RGB")  # force RGB
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # convert for OpenCV

        original_image, output_image = st.columns(2)

        with original_image:
            st.header("Original")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.header("Filters List")
        options = st.selectbox("Select Filter", ("None", "BlackWhite", "style_image", "pencil_sketch", "HDR", "Brightness"))

        if options == "None":
            output = img
        elif options == "BlackWhite":
            output = blackwhite(img)
        elif options == "pencil_sketch":
            kvalue = st.slider("Kernel Size", 1, 9, 5, step=2)
            output = pencil_sketch(img, kvalue)
        elif options == "HDR":
            level = st.slider("Brightness Level", -50, 100, 10, step=10)
            sigma_s = st.slider("sigma_s", 5, 150, 50, step=5)
            sigma_r = st.slider("sigma_r", 0.01, 1.0, 0.2)
            output = HDR(img, level=level, sigma_s=sigma_s, sigma_r=sigma_r)
        elif options == "Brightness":
            level = st.slider("Brightness Level", -50, 100, 10, step=10)
            output = Brightness(img, level=level)
        elif options == "style_image":
            kvalue = st.slider("Kernel Size", 1, 9, 3, step=2)
            sigma_s = st.slider("sigma_s", 5, 150, 60, step=5)
            sigma_r = st.slider("sigma_r", 0.01, 1.0, 0.45)
            output = style_image(img, kvalue, sigma_s, sigma_r)

        with output_image:
            st.header("Output Image")
            if len(output.shape) == 2:
                st.image(output, use_container_width=True, clamp=True)
            else:
                st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.download_button("Download Output Image", data=cv2.imencode('.png', output)[1].tobytes(), file_name="output.png", mime="image/png")
    else:
        st.warning("Please upload an image to apply filters.")
    st.markdown("Made with ❤️ by [Raghad]")
