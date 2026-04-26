
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import tempfile
import os

st.set_page_config(
    page_title="Smart Parking AI",
    page_icon="🅿️",
    layout="wide"
)

st.title("🅿️ Smart Parking AI System")

st.write(
    "Upload a parking lot image and the trained YOLO model will detect "
    "available parking spaces. The system only displays spaces predicted as empty."
)

st.sidebar.header("Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.25,
    step=0.05
)

st.sidebar.write("Model Classes:")
st.sidebar.write("0 = empty")
st.sidebar.write("1 = occupied")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
    st.sidebar.success("Model loaded successfully")
except:
    st.error("Model could not be loaded. Make sure best.pt is uploaded in Colab.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a parking lot image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        image_path = temp_file.name

    results = model.predict(
        source=image_path,
        conf=confidence_threshold,
        save=False
    )

    result = results[0]
    annotated_image = image_np.copy()

    empty_spaces = []
    empty_count = 0

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if class_name.lower() == "empty":
            empty_count += 1
            slot_id = f"Empty Slot {empty_count}"

            empty_spaces.append({
                "Slot ID": slot_id,
                "Class": class_name,
                "Confidence": round(confidence, 2),
                "X1": x1,
                "Y1": y1,
                "X2": x2,
                "Y2": y2,
                "Status": "Available"
            })

            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(annotated_image, slot_id, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    with col2:
        st.subheader("Detected Empty Spaces")
        st.image(annotated_image, use_container_width=True)

    st.divider()
    st.metric("Empty Spaces Detected", empty_count)
    st.subheader("Detection Results")

    if empty_spaces:
        df = pd.DataFrame(empty_spaces)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV Results",
            data=csv,
            file_name="empty_parking_spaces.csv",
            mime="text/csv"
        )
    else:
        st.warning("No empty parking spaces detected.")

    os.remove(image_path)

else:
    st.info("Upload a parking lot image to begin.")
