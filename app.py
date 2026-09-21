"""
app.py

Streamlit UI for the Cambodian License Plate Recognition pipeline.

Pipeline:
    Raw Image / Video
        ↓
    Vehicle Classification (VC)
        ↓
    Plate Detection
        ↓
    Plate Number OCR (PN)
        ↓
    Province Classification (PC)

Current PC model:
    PC-v2.1 ResNet-18

The pipeline is imported dynamically from pipeline_without_dir.py.
Models are cached using st.cache_resource so they are loaded only once.

Run locally:
    streamlit run app.py
"""

from pathlib import Path
import glob
import importlib.util
import tempfile

import cv2
import numpy as np
import streamlit as st


# =========================================================
# APP / MODEL VERSION
# =========================================================

MODEL_VERSION = "PC-v2.1-ResNet18"
APP_VERSION = "v2.1"


# =========================================================
# LOAD PIPELINE MODULE
# =========================================================

SCRIPTS_DIR = Path(__file__).resolve().parent

_matches = glob.glob(str(SCRIPTS_DIR / "*pipeline_without_dir.py"))

if not _matches:
    raise FileNotFoundError(
        "Could not find pipeline_without_dir.py. "
        "The Streamlit app requires this file."
    )

_spec = importlib.util.spec_from_file_location(
    "pipeline_without_dir",
    _matches[0],
)

_pipeline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline_module)

PlateRecognitionPipeline = _pipeline_module.PlateRecognitionPipeline


# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="Cambodian Plate Recognition",
    page_icon="🚗",
    layout="wide",
)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_pipeline(model_version: str):
    """
    Load the complete recognition pipeline.

    model_version is intentionally passed as a cache key.
    When MODEL_VERSION changes, Streamlit creates a new
    cached pipeline instead of reusing an older model.
    """

    print("=" * 60)
    print(f"Loading pipeline: {model_version}")
    print("=" * 60)

    pipeline = PlateRecognitionPipeline()

    print("=" * 60)
    print(f"Pipeline ready: {model_version}")
    print("=" * 60)

    return pipeline


# =========================================================
# IMAGE UTILITIES
# =========================================================

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decode_uploaded_image(uploaded_file):
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8,
    )

    return cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR,
    )


# =========================================================
# VIDEO UTILITIES
# =========================================================

def extract_video_frame(uploaded_file, frame_index):
    """
    Save uploaded video temporarily and extract one frame.
    """

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4",
    ) as tmp:

        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_index,
    )

    ok, frame = cap.read()

    cap.release()

    return (
        frame if ok else None,
        total_frames,
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

def show_model_information(pipeline):
    """
    Display information about the model that is actually
    running inside Streamlit.
    """

    st.sidebar.divider()

    st.sidebar.subheader("🧠 Model Information")

    st.sidebar.success(
        f"PC Model: {MODEL_VERSION}"
    )

    checkpoint = getattr(
        pipeline,
        "pc_checkpoint",
        None,
    )

    if checkpoint:

        model_name = checkpoint.get(
            "model_name",
            "resnet18",
        )

        epoch = checkpoint.get(
            "epoch",
            "Unknown",
        )

        class_names = checkpoint.get(
            "class_names",
            [],
        )

        input_height = checkpoint.get(
            "input_height",
            160,
        )

        input_width = checkpoint.get(
            "input_width",
            416,
        )

        st.sidebar.write(
            f"**Architecture:** {model_name}"
        )

        st.sidebar.write(
            f"**Checkpoint epoch:** {epoch}"
        )

        st.sidebar.write(
            f"**Classes:** {len(class_names)}"
        )

        st.sidebar.write(
            f"**Input:** {input_width} × {input_height}"
        )

    else:
        st.sidebar.warning(
            "Checkpoint metadata not available."
        )


# =========================================================
# RESULT UI
# =========================================================

def render_result(raw_image, result):

    if "error" in result:

        st.error(
            f"Pipeline stopped: {result['error']}"
        )

        if "vehicle_crop" in result:

            st.image(
                bgr_to_rgb(
                    result["vehicle_crop"]
                ),
                caption=(
                    "Vehicle crop "
                    "(last successful stage)"
                ),
            )

        return

    # -----------------------------------------------------
    # Raw + Annotated Result
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Preview Raw")

        st.image(
            bgr_to_rgb(raw_image),
            use_container_width=True,
        )

    with col2:

        st.subheader("Result")

        st.image(
            bgr_to_rgb(
                result["annotated_image"]
            ),
            use_container_width=True,
        )

    st.divider()

    # -----------------------------------------------------
    # Plate + Recognition Result
    # -----------------------------------------------------

    col3, col4 = st.columns(
        [1, 1]
    )

    with col3:

        st.subheader("Plate Crop")

        st.image(
            bgr_to_rgb(
                result["plate_crop"]
            ),
            width=400,
        )

        # Useful for debugging PC-v2.1
        if "masked_plate" in result:

            with st.expander(
                "🔍 Show PC classifier input"
            ):

                st.caption(
                    "This is the plate image sent "
                    "to PC-v2.1 after the plate "
                    "number region is masked."
                )

                st.image(
                    bgr_to_rgb(
                        result["masked_plate"]
                    ),
                    width=400,
                )

    with col4:

        st.subheader(
            "Recognition Result"
        )

        # VC
        st.markdown(
            f"- **Vehicle Category:** "
            f"{result.get('VC', '-')}"
        )

        # PC
        pc_conf = result.get(
            "PC_confidence"
        )

        if pc_conf is not None:
            conf_str = (
                f" ({pc_conf:.1%})"
            )
        else:
            conf_str = ""

        pc_display = result.get(
            "PC",
            "-",
        )

        if pc_display.startswith(
            "PC_"
        ):
            pc_display = pc_display[3:]

        st.markdown(
            f"- **Province Classification:** "
            f"{pc_display}{conf_str}"
        )

        # PN
        st.markdown(
            f"- **Plate Number:** "
            f"{result.get('PN', '-')}"
        )


# =========================================================
# MAIN
# =========================================================

def main():

    st.title(
        "🚗 Cambodian License Plate Recognition"
    )

    st.caption(
        f"Running model: "
        f"**{MODEL_VERSION}**"
    )

    # -----------------------------------------------------
    # Reload model button
    # -----------------------------------------------------

    if st.sidebar.button(
        "🔄 Reload Models",
        use_container_width=True,
    ):

        load_pipeline.clear()

        st.sidebar.success(
            "Model cache cleared."
        )

        st.rerun()

    # -----------------------------------------------------
    # Load pipeline
    # -----------------------------------------------------

    with st.spinner(
        f"Loading {MODEL_VERSION}..."
    ):

        pipeline = load_pipeline(
            MODEL_VERSION
        )

    show_model_information(
        pipeline
    )

    # -----------------------------------------------------
    # Upload
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload raw image or video",
        type=[
            "jpg",
            "jpeg",
            "png",
            "mp4",
            "mov",
            "avi",
        ],
    )

    if uploaded_file is None:

        st.info(
            "Upload an image or video to run "

        )

        return

    # -----------------------------------------------------
    # Image / Video
    # -----------------------------------------------------

    is_video = (
        uploaded_file.type.startswith(
            "video"
        )
    )

    if is_video:

        frame_index = st.slider(
            "Pick a frame",
            0,
            500,
            0,
        )

        raw_image, total_frames = (
            extract_video_frame(
                uploaded_file,
                frame_index,
            )
        )

        st.caption(
            f"Video contains approximately "
            f"{total_frames} frames."
        )

        if raw_image is None:

            st.error(
                "Could not read this frame. "
                "Try another frame."
            )

            return

    else:

        raw_image = decode_uploaded_image(
            uploaded_file
        )

        if raw_image is None:

            st.error(
                "Could not decode the image."
            )

            return

    # -----------------------------------------------------
    # Run inference
    # -----------------------------------------------------

    with st.spinner(
        "Running "
    ):

        result = pipeline.run(
            raw_image,
            return_debug=True,
        )

    render_result(
        raw_image,
        result,
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()