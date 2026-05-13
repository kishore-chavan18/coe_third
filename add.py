import streamlit as st
from PIL import Image, ImageOps, ImageEnhance, ImageDraw
import numpy as np
import cv2
from io import BytesIO

st.set_page_config(
    page_title="Fun AI Filters",
    page_icon="😺",
    layout="wide"
)

st.title("😺 Fun AI Photo Filters")
st.write(
    "Upload a photo and apply cute social-media style filters "
    "like kitten ears, puppy tongue, black & white, polar region, and nature vibes."
)

# -----------------------------------
# Helper Functions
# -----------------------------------

def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def download_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# -----------------------------------
# Face Detection
# -----------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------------
# Filters
# -----------------------------------

def black_white_filter(image):
    return ImageOps.grayscale(image).convert("RGB")

def nature_filter(image):
    img = ImageEnhance.Color(image).enhance(1.8)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Brightness(img).enhance(1.05)
    return img

def polar_filter(image):
    img = ImageEnhance.Color(image).enhance(0.5)

    np_img = np.array(img)
    blue_overlay = np.full(np_img.shape, (180, 220, 255), dtype=np.uint8)

    blended = cv2.addWeighted(np_img, 0.75, blue_overlay, 0.25, 0)

    return Image.fromarray(blended)

# -----------------------------------
# Cat Filter
# -----------------------------------

def cat_filter(image):

    img = pil_to_cv2(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    pil_img = cv2_to_pil(img)
    draw = ImageDraw.Draw(pil_img)

    for (x, y, w, h) in faces:

        # Cat ears
        left_ear = [
            (x + w * 0.15, y + h * 0.15),
            (x + w * 0.30, y - h * 0.30),
            (x + w * 0.45, y + h * 0.15)
        ]

        right_ear = [
            (x + w * 0.55, y + h * 0.15),
            (x + w * 0.70, y - h * 0.30),
            (x + w * 0.85, y + h * 0.15)
        ]

        draw.polygon(left_ear, fill=(255, 170, 200))
        draw.polygon(right_ear, fill=(255, 170, 200))

        # Cat nose
        draw.ellipse(
            (
                x + w * 0.42,
                y + h * 0.60,
                x + w * 0.58,
                y + h * 0.72
            ),
            fill=(255, 120, 140)
        )

        # Whiskers
        for i in range(3):
            offset = i * 10

            draw.line(
                (
                    x + w * 0.15,
                    y + h * 0.65 + offset,
                    x + w * 0.42,
                    y + h * 0.68
                ),
                fill="white",
                width=2
            )

            draw.line(
                (
                    x + w * 0.58,
                    y + h * 0.68,
                    x + w * 0.85,
                    y + h * 0.65 + offset
                ),
                fill="white",
                width=2
            )

    return pil_img

# -----------------------------------
# Dog Filter
# -----------------------------------

def dog_filter(image):

    img = pil_to_cv2(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    pil_img = cv2_to_pil(img)
    draw = ImageDraw.Draw(pil_img)

    for (x, y, w, h) in faces:

        # Dog ears
        draw.ellipse(
            (
                x - w * 0.15,
                y,
                x + w * 0.20,
                y + h * 0.70
            ),
            fill=(160, 100, 60)
        )

        draw.ellipse(
            (
                x + w * 0.80,
                y,
                x + w * 1.15,
                y + h * 0.70
            ),
            fill=(160, 100, 60)
        )

        # Dog nose
        draw.ellipse(
            (
                x + w * 0.40,
                y + h * 0.58,
                x + w * 0.60,
                y + h * 0.75
            ),
            fill="black"
        )

        # Tongue
        draw.ellipse(
            (
                x + w * 0.44,
                y + h * 0.75,
                x + w * 0.56,
                y + h * 0.95
            ),
            fill=(255, 80, 120)
        )

    return pil_img

# -----------------------------------
# Sidebar
# -----------------------------------

st.sidebar.header("🎨 Choose Filter")

filter_option = st.sidebar.selectbox(
    "Select Filter",
    [
        "Cute Kitten 😺",
        "Cute Dog 🐶",
        "Black & White ⚫",
        "Nature 🌿",
        "Polar Region ❄️"
    ]
)

# -----------------------------------
# Upload
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("✨ Apply Filter"):

        with st.spinner("Applying cool filter..."):

            if filter_option == "Cute Kitten 😺":
                result = cat_filter(image)

            elif filter_option == "Cute Dog 🐶":
                result = dog_filter(image)

            elif filter_option == "Black & White ⚫":
                result = black_white_filter(image)

            elif filter_option == "Nature 🌿":
                result = nature_filter(image)

            elif filter_option == "Polar Region ❄️":
                result = polar_filter(image)

            else:
                result = image

            with col2:
                st.subheader("Filtered Result")
                st.image(result, use_container_width=True)

            st.success("Filter applied successfully!")

            st.download_button(
                label="⬇ Download Image",
                data=download_image(result),
                file_name="filtered_image.png",
                mime="image/png"
            )

# -----------------------------------
# Footer
# -----------------------------------

st.markdown("---")
st.caption("Built with Streamlit + OpenCV + PIL")
