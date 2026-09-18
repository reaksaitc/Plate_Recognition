"""
streamlit_app.py

Interactive test UI for the full VC/PC/PN pipeline (run_full_pipeline.py).
Layout follows the sketch: upload raw image/video -> raw preview +
annotated vehicle crop side by side -> plate crop + structured
VC/PC/PN results below.

Reuses PlateRecognitionPipeline directly from run_full_pipeline.py
(loaded dynamically by filename pattern, since that file has a
numeric prefix like "27.run_full_pipeline.py" and Python can't
import a module name starting with a digit -- same fix already used
in balance_pc_dataset.py) -- same models, same preprocessing, single
source of truth. Model loading is cached (@st.cache_resource) so it
only happens once per session, not on every interaction.

Run with (from inside the scripts/ folder):
    streamlit run 28.streamlit_app.py
    (use whatever numeric prefix this file actually has)
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import glob
import importlib.util

SCRIPTS_DIR = Path(__file__).resolve().parent
# Load run_full_pipeline.py by filename pattern, not a plain import --
# the actual file has a numeric prefix (e.g. "27.run_full_pipeline.py"),
# and Python can't import a module name starting with a digit.
_matches = glob.glob(str(SCRIPTS_DIR / "*pipeline_without_dir.py"))
if not _matches:
    raise FileNotFoundError(
        "Could not find *pipeline_without_dir.py in the scripts/ folder -- "
        "this app needs it importable for PlateRecognitionPipeline."
    )
_spec = importlib.util.spec_from_file_location("pipeline_without_dir", _matches[0])
_run_full_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run_full_pipeline)
PlateRecognitionPipeline = _run_full_pipeline.PlateRecognitionPipeline


st.set_page_config(page_title="Cambodian Plate Recognition", layout="wide")


@st.cache_resource
def load_pipeline():
    return PlateRecognitionPipeline()


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decode_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def extract_video_frame(uploaded_file, frame_index):
    """Writes the uploaded video to a temp file (cv2.VideoCapture needs
    a real file path, not an in-memory buffer), reads the requested
    frame, returns it as a BGR array + total frame count."""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    return (frame if ok else None), total_frames


def render_result(raw_image, result):
    if "error" in result:
        st.error(f"Pipeline stopped: {result['error']}")
        # show whatever debug images DID get produced before the error
        if "vehicle_crop" in result:
            st.image(bgr_to_rgb(result["vehicle_crop"]), caption="Vehicle crop (last successful stage)")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Preview Raw")
        st.image(bgr_to_rgb(raw_image), use_container_width=True)
    with col2:
        st.subheader("Result")
        st.image(bgr_to_rgb(result["annotated_image"]), use_container_width=True)

    st.divider()

    col3, col4 = st.columns([1, 1])
    with col3:
        st.subheader("Plate Crop")
        st.image(bgr_to_rgb(result["plate_crop"]), width=400)
    with col4:
        st.subheader("Recognition Result")
        st.markdown(f"- **Vehicle Category:** {result.get('VC', '-')}")
        pc_conf = result.get("PC_confidence")
        conf_str = f" (confidence: {pc_conf:.0%})" if pc_conf is not None else ""
        pc_display = result.get('PC', '-')
        if pc_display.startswith("PC_"):
            pc_display = pc_display[3:]
        st.markdown(f"- **PC:** {pc_display}{conf_str}")
        st.markdown(f"- **PN:** {result.get('PN', '-')}")


def main():
    st.title("🚗 Cambodian License Plate Recognition")

    pipeline = load_pipeline()

    uploaded_file = st.file_uploader(
        "Upload raw image or video",
        type=["jpg", "jpeg", "png", "mp4", "mov", "avi"],
    )

    if uploaded_file is None:
        st.info("Upload an image or video to run the full VC \u2192 PC \u2192 PN pipeline.")
        return

    is_video = uploaded_file.type.startswith("video")

    if is_video:
        # Video: let the user scrub to a frame, then run the (image)
        # pipeline on that single frame -- no continuous/real-time
        # video processing.
        frame_index = st.slider("Pick a frame", 0, 500, 0)
        raw_image, total_frames = extract_video_frame(uploaded_file, frame_index)
        st.caption(f"Video has ~{total_frames} frames.")
        if raw_image is None:
            st.error("Could not read that frame -- try a different frame index.")
            return
    else:
        raw_image = decode_uploaded_image(uploaded_file)

    with st.spinner("Running pipeline..."):
        result = pipeline.run(raw_image, return_debug=True)

    render_result(raw_image, result)


if __name__ == "__main__":
    main()