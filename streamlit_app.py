
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import yaml
from pathlib import Path

# Set page config
st.set_page_config(page_title="YOLOv8 Object Detection App", layout="wide")

st.title("Traffic Road Object Detection with YOLOv8")
st.write("Upload an image to detect objects like cars.")

# Load the model
@st.cache_resource
def load_model():
    model_path = Path("best.pt") # Model is expected in the same directory
    if not model_path.exists():
        st.error(f"Model file not found at {model_path}. Please ensure best.pt is in the same directory as the app.")
        return None
    model = YOLO(str(model_path))
    return model

model = load_model()

# Load class names from data.yaml
@st.cache_data
def load_class_names():
    data_yaml_path = Path("data.yaml")
    if not data_yaml_path.exists():
        st.warning(f"data.yaml not found at {data_yaml_path}. Using default class names.")
        return {0: 'object'}
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    return {i: name for i, name in enumerate(data_config.get('names', ['object']))}

class_names = load_class_names()

if model is None:
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file
    image = Image.open(uploaded_file)
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Convert PIL Image to OpenCV format
    img_np = np.array(image)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    if st.button("Detect Objects"):
        with st.spinner("Detecting objects..."):
            # Perform inference
            results = model.predict(source=img_cv2, conf=0.25, iou=0.7, verbose=False)

            # Process and display results
            for r in results:
                # Annotated image with bounding boxes and labels
                annotated_img_bgr = r.plot()
                annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)

                st.subheader("Detection Results:")
                st.image(annotated_img_rgb, caption="Detected Objects", use_container_width=True)

                # Optionally display confidence scores and bounding box coordinates
                if len(r.boxes) > 0:
                    st.write("Found objects:")
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        bbox = box.xyxy[0].cpu().numpy().astype(int) # xyxy format
                        label = class_names.get(class_id, f"Class {class_id}")
                        st.write(f"- **{label}**: Confidence {conf:.2f} (Coords: x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]})")
                else:
                    st.write("No objects detected.")
else:
    st.info("Please upload an image to start detection.")
