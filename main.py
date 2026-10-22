
import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageOps

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    rembg_remove = None
    REMBG_AVAILABLE = False


APP_TITLE = "Motion Photo Studio"
MAX_DIMENSION = 1800


# ---------------------------------------------------------
# IMAGE AND VIDEO UTILITIES
# ---------------------------------------------------------

def open_rgb_image(image):
    """Return a PIL RGB image from a PIL image or file path."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")

    raise ValueError("Please provide a valid image.")


def resize_if_needed(image, max_dimension=MAX_DIMENSION):
    """Resize large images while preserving their aspect ratio."""
    image = image.copy()
    image.thumbnail(
        (max_dimension, max_dimension),
        Image.Resampling.LANCZOS,
    )
    return image


def extract_video_frames(video_path, sample_interval, max_frames):
    """Extract frames at regular intervals from a video."""
    if not video_path:
        raise ValueError("Please upload a video first.")

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError("The video could not be opened.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError("Could not read the video's frame information.")

    duration = frame_count / fps
    interval = max(float(sample_interval), 0.05)
    total_samples = min(
        int(duration / interval) + 1,
        int(max_frames),
    )

    frames = []
    labels = []

    for index in range(total_samples):
        timestamp = index * interval

        if timestamp >= duration:
            break

        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        success, frame = capture.read()

        if not success:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame)
        pil_frame = resize_if_needed(pil_frame)

        frames.append(pil_frame)
        labels.append(f"Frame {len(frames)} — {timestamp:.2f}s")

    capture.release()

    if not frames:
        raise ValueError(
            "No frames were extracted. Try a different video or interval."
        )

    return frames, labels


def extract_frames(video_path, sample_interval, max_frames):
    """Gradio event handler for video extraction."""
    try:
        frames, labels = extract_video_frames(
            video_path,
            sample_interval,
            max_frames,
        )

        status = (
            f"Successfully extracted {len(frames)} frames. "
            "Select the frames you want to use."
        )

        return (
            frames,
            labels,
            gr.update(choices=labels, value=labels),
            status,
        )

    except Exception as exc:
        return (
            [],
            [],
            gr.update(choices=[], value=[]),
            f"Frame extraction failed: {exc}",
        )


def select_all_frames(labels):
    return gr.update(choices=labels, value=labels)


def select_no_frames(labels):
    return gr.update(choices=labels, value=[])


# ---------------------------------------------------------
# COMPOSITION MODES
# ---------------------------------------------------------

def average_frames(frames):
    """Blend selected frames into one image using pixel averaging."""
    if not frames:
        raise ValueError("No frames were selected.")

    images = [open_rgb_image(frame) for frame in frames]
    reference_size = images[0].size

    aligned = [
        image.resize(reference_size, Image.Resampling.LANCZOS)
        for image in images
    ]

    arrays = [np.asarray(image, dtype=np.float32) for image in aligned]
    result = np.mean(arrays, axis=0)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)


def remove_background_from_frames(frames):
    """Remove the background from each selected frame."""
    if not REMBG_AVAILABLE:
        raise RuntimeError(
            "Background removal is unavailable. Install rembg[cpu] "
            "and restart the application."
        )

    if not frames:
        raise ValueError("No frames were selected.")

    cutouts = []

    for frame in frames:
        image = open_rgb_image(frame)
        image = resize_if_needed(image)

        result = rembg_remove(image)
        result = result.convert("RGBA")

        cutouts.append(result)

    return cutouts


def align_and_overlay(frames, opacity=0.55, remove_background=False):
    """
    Align selected frames to the first frame using ORB features,
    then overlay them into one composite.
    """
    if not frames:
        raise ValueError("No frames were selected.")

    rgb_images = [open_rgb_image(frame) for frame in frames]
    reference = rgb_images[0]
    reference_size = reference.size

    reference_array = np.array(reference)
    reference_gray = cv2.cvtColor(
        reference_array,
        cv2.COLOR_RGB2GRAY,
    )

    orb = cv2.ORB_create(nfeatures=2500)
    reference_keypoints, reference_descriptors = orb.detectAndCompute(
        reference_gray,
        None,
    )

    if reference_descriptors is None or len(reference_keypoints) < 4:
        raise ValueError(
            "Not enough visual features were found in the first frame "
            "to align the images. Try a different frame or use Average."
        )

    height, width = reference_array.shape[:2]
    base = reference_array.astype(np.float32)
    accumulated = base.copy()
    contribution_count = 1

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    for image in rgb_images[1:]:
        image = image.resize(reference_size, Image.Resampling.LANCZOS)
        image_array = np.array(image)
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 4:
            continue

        matches = matcher.knnMatch(
            descriptors,
            reference_descriptors,
            k=2,
        )

        good_matches = [
            pair[0]
            for pair in matches
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
        ]

        if len(good_matches) < 4:
            continue

        source_points = np.float32(
            [keypoints[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        destination_points = np.float32(
            [reference_keypoints[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        transform, inliers = cv2.estimateAffinePartial2D(
            source_points,
            destination_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
        )

        if transform is None:
            continue

        aligned = cv2.warpAffine(
            image_array,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        accumulated += aligned.astype(np.float32)
        contribution_count += 1

    average = accumulated / contribution_count
    average = np.clip(average, 0, 255).astype(np.uint8)

    # Keep the reference image dominant and add the aligned result.
    opacity = float(np.clip(opacity, 0.0, 1.0))
    composite = cv2.addWeighted(
        reference_array,
        1.0 - opacity,
        average,
        opacity,
        0,
    )

    result = Image.fromarray(composite)

    if remove_background:
        if not REMBG_AVAILABLE:
            raise RuntimeError(
                "Background removal requires rembg[cpu]."
            )

        result = rembg_remove(result).convert("RGBA")

    return result


def compose_frames(
    all_frames,
    labels,
    selected_labels,
    mode,
    opacity,
    remove_background,
):
    """Create a composition from the frames selected by the user."""
    try:
        if not all_frames or not labels:
            raise ValueError("Extract frames from a video first.")

        if not selected_labels:
            raise ValueError("Select at least one frame.")

        selected_indices = [
            index
            for index, label in enumerate(labels)
            if label in selected_labels
        ]

        selected_frames = [
            all_frames[index]
            for index in selected_indices
        ]

        if not selected_frames:
            raise ValueError("No valid selected frames were found.")

        if mode == "Average":
            result = average_frames(selected_frames)

        elif mode == "Align and Overlay":
            result = align_and_overlay(
                selected_frames,
                opacity=opacity,
                remove_background=remove_background,
            )

        elif mode == "Background Removal":
            cutouts = remove_background_from_frames(selected_frames)
            result = cutouts[0]

            # Overlay additional cutouts on the first cutout.
            for cutout in cutouts[1:]:
                if cutout.size != result.size:
                    cutout = cutout.resize(
                        result.size,
                        Image.Resampling.LANCZOS,
                    )

                result.alpha_composite(cutout)

        else:
            raise ValueError("Choose a valid composition mode.")

        status = (
            f"Composition created successfully using "
            f"{len(selected_frames)} frame(s) with '{mode}'."
        )

        if remove_background and mode != "Background Removal":
            if not REMBG_AVAILABLE:
                raise RuntimeError(
                    "Background removal requires rembg[cpu]."
                )
            result = rembg_remove(result).convert("RGBA")
            status += " Background removal applied."

        # Return the preview AND the actual image for export.
        return result, result, status

    except Exception as exc:
        return (
            None,
            None,
            f"Composition failed: {exc}",
        )


# ---------------------------------------------------------
# EXPORT AND DOWNLOAD
# ---------------------------------------------------------

def export_result(composite, format_choice):
    """Save the composite to a temporary file for Gradio download."""
    try:
        if composite is None:
            raise ValueError(
                "Create a composition before exporting."
            )

        if not isinstance(composite, Image.Image):
            composite = open_rgb_image(composite)

        format_choice = format_choice.upper()

        if format_choice not in ("PNG", "JPG"):
            format_choice = "PNG"

        suffix = ".png" if format_choice == "PNG" else ".jpg"

        file_handle = tempfile.NamedTemporaryFile(
            prefix="motion_photo_",
            suffix=suffix,
            delete=False,
        )
        output_path = file_handle.name
        file_handle.close()

        if format_choice == "PNG":
            composite.save(output_path, format="PNG", optimize=True)
        else:
            # JPEG does not support transparency.
            if composite.mode in ("RGBA", "LA"):
                rgba = composite.convert("RGBA")
                background = Image.new(
                    "RGBA",
                    rgba.size,
                    (255, 255, 255, 255),
                )
                background.alpha_composite(rgba)
                composite = background.convert("RGB")
            else:
                composite = composite.convert("RGB")

            composite.save(
                output_path,
                format="JPEG",
                quality=95,
                optimize=True,
            )

        return (
            output_path,
            f"Your {format_choice} image is ready to download.",
        )

    except Exception as exc:
        return (
            None,
            f"Export failed: {exc}",
        )


# ---------------------------------------------------------
# MODERN INTERFACE
# ---------------------------------------------------------


# ---------------------------------------------------------
# MOTION PHOTO STUDIO V3 — MODERN WORKSPACE
# ---------------------------------------------------------

CUSTOM_CSS = """
:root {
    --bg: #07111f;
    --panel: #0d1b2e;
    --panel2: #11233b;
    --border: #203956;
    --text: #f1f6ff;
    --muted: #9bb0cc;
    --blue: #4387ff;
    --purple: #9255ff;
}

.gradio-container {
    max-width: 1800px !important;
    margin: auto !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

body {
    background: var(--bg) !important;
}

footer {
    display: none !important;
}

.mps-header {
    padding: 24px 30px;
    border: 1px solid var(--border);
    border-radius: 20px;
    margin-bottom: 20px;
    background:
        radial-gradient(ellipse at 90% 0%, #4932a8 0%, transparent 38%),
        linear-gradient(115deg, #112b4c, #10182d 65%, #191633);
}

.mps-brand {
    font-size: 30px;
    font-weight: 850;
    letter-spacing: -1px;
    color: white;
}

.mps-brand span {
    color: #8e9cff;
}

.mps-tagline {
    color: #bfd0ed;
    font-size: 14px;
    margin-top: 8px;
    line-height: 1.6;
}

.mps-card {
    border: 1px solid var(--border);
    border-radius: 17px;
    padding: 18px;
    margin-bottom: 14px;
    background: linear-gradient(145deg, #0f2035, #0b1728);
}

.mps-step {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 15px;
}

.mps-number {
    display: flex;
    flex-shrink: 0;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #3987ff, #7655ff);
    color: white;
    font-weight: 850;
    font-size: 17px;
}

.mps-heading {
    color: white;
    font-size: 18px;
    font-weight: 800;
}

.mps-description {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.6;
    margin-top: 3px;
}

.mps-sidebar {
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px;
    background: #0b192b;
}

.mps-nav-title {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #809ac0;
    font-weight: 800;
    margin: 6px 0 14px;
}

.mps-tip {
    border: 1px solid #24446d;
    background: #102440;
    padding: 13px;
    border-radius: 12px;
    font-size: 12px;
    color: #c6d9fa;
    line-height: 1.7;
}

.mps-status {
    border: 1px solid #24436a;
    background: #10233b;
    border-radius: 11px;
    padding: 12px 14px;
    color: #b9d4ff;
    font-size: 13px;
    line-height: 1.6;
}

.mps-preview-title {
    color: white;
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 12px;
}

.mps-footer {
    color: #8298b8;
    text-align: center;
    font-size: 12px;
    padding: 16px;
}

.primary-button {
    background: linear-gradient(105deg, #3486ff, #9252f5) !important;
    border: none !important;
    color: white !important;
    font-weight: 800 !important;
    min-height: 44px;
}

.secondary-button {
    background: #142943 !important;
    border: 1px solid #2b4669 !important;
    color: #e7f0ff !important;
}

.mode-choice label {
    border: 1px solid #294361 !important;
    border-radius: 10px !important;
    padding: 10px !important;
}

button, input, textarea {
    border-radius: 9px !important;
}

@media (max-width: 850px) {
    .mps-header {
        padding: 20px;
    }

    .mps-brand {
        font-size: 25px;
    }
}
"""


with gr.Blocks(
    title="Motion Photo Studio v4.01",
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="purple",
        neutral_hue="slate",
        font=["Arial", "sans-serif"],
    ),
    css=CUSTOM_CSS,
) as demo:

    # Shared application state
    frames_state = gr.State([])
    labels_state = gr.State([])
    composite_state = gr.State(None)

    with gr.Row(equal_height=False):

        # LEFT NAVIGATION
        with gr.Column(scale=1, min_width=175):

            gr.HTML("""
            <div class="mps-sidebar">
                <div class="mps-nav-title">WORKSPACE</div>
                <div style="font-size:15px;font-weight:800;line-height:2.8">
                    📁 &nbsp; Studio Home<br>
                    🎞️ &nbsp; Extract Frames<br>
                    ✨ &nbsp; Compose<br>
                    ⬇️ &nbsp; Export
                </div>
                <hr style="border-color:#203956;margin:18px 0">
                <div class="mps-nav-title">QUICK GUIDE</div>
                <div class="mps-tip">
                    <strong>Turn moments into art</strong><br><br>
                    1. Upload a video<br>
                    2. Select your frames<br>
                    3. Create a composition<br>
                    4. Download your image
                </div>
                <div style="font-size:11px;color:#8298b8;margin-top:18px">
                    MOTION PHOTO STUDIO<br>Version 3.0
                </div>
            </div>
            """)

        # MAIN WORKSPACE
        with gr.Column(scale=5, min_width=320):

            gr.HTML("""
            <div class="mps-header">
                <div class="mps-brand">📷 Motion <span>Photo Studio</span></div>
                <div class="mps-tagline">
                    Video frames. Creative compositions.<br>
                    Transform movement into memorable still images.
                </div>
            </div>
            """)

            # STEP 1: VIDEO IMPORT
            with gr.Group(elem_classes=["mps-card"]):

                gr.HTML("""
                <div class="mps-step">
                    <div class="mps-number">1</div>
                    <div>
                        <div class="mps-heading">Import your video</div>
                        <div class="mps-description">
                            Upload a video and configure frame sampling.
                        </div>
                    </div>
                </div>
                """)

                video_input = gr.Video(
                    label="Source video",
                    sources=["upload"],
                )

                with gr.Row():
                    sample_interval = gr.Slider(
                        minimum=0.1,
                        maximum=3.0,
                        value=0.5,
                        step=0.1,
                        label="Sampling interval (seconds)",
                    )

                    max_frames = gr.Slider(
                        minimum=2,
                        maximum=30,
                        value=12,
                        step=1,
                        label="Maximum frames",
                    )

                extract_button = gr.Button(
                    "🎞️  Extract Frames",
                    variant="primary",
                    elem_classes=["primary-button"],
                )

                extraction_status = gr.Markdown(
                    "Upload a video to get started.",
                    elem_classes=["mps-status"],
                )

            # STEP 2: FRAME SELECTION
            with gr.Group(elem_classes=["mps-card"]):

                gr.HTML("""
                <div class="mps-step">
                    <div class="mps-number">2</div>
                    <div>
                        <div class="mps-heading">Choose your moments</div>
                        <div class="mps-description">
                            Select the frames you want to combine.
                        </div>
                    </div>
                </div>
                """)

                frames_gallery = gr.Gallery(
                    label="Extracted frames",
                    columns=4,
                    height="auto",
                    object_fit="contain",
                    preview=True,
                )

                frame_selection = gr.CheckboxGroup(
                    choices=[],
                    value=[],
                    label="Select frames for your composition",
                    interactive=True,
                )

                with gr.Row():
                    select_all_button = gr.Button(
                        "✓ Select All",
                        elem_classes=["secondary-button"],
                    )

                    select_none_button = gr.Button(
                        "✕ Clear Selection",
                        elem_classes=["secondary-button"],
                    )

            # STEP 3: COMPOSITION
            with gr.Group(elem_classes=["mps-card"]):

                gr.HTML("""
                <div class="mps-step">
                    <div class="mps-number">3</div>
                    <div>
                        <div class="mps-heading">Create your composition</div>
                        <div class="mps-description">
                            Choose a processing method and fine-tune the result.
                        </div>
                    </div>
                </div>
                """)

                composition_mode = gr.Radio(
                    choices=[
                        "Align and Overlay",
                        "Average",
                        "Background Removal",
                    ],
                    value="Align and Overlay",
                    label="Composition method",
                    elem_classes=["mode-choice"],
                )

                opacity = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.55,
                    step=0.05,
                    label="Overlay opacity",
                )

                remove_background_checkbox = gr.Checkbox(
                    value=False,
                    label="Remove the background from the final result",
                    interactive=REMBG_AVAILABLE,
                )

                if REMBG_AVAILABLE:
                    background_text = (
                        "Background removal is available."
                    )
                else:
                    background_text = (
                        "To enable background removal, install rembg[cpu]. "
                        "Other composition modes can still be used."
                    )

                gr.HTML(
                    f'<div class="mps-tip">{background_text}</div>'
                )

                compose_button = gr.Button(
                    "✨  Create Composition",
                    variant="primary",
                    elem_classes=["primary-button"],
                )

                composition_status = gr.Markdown(
                    "Your finished image will appear in the preview panel.",
                    elem_classes=["mps-status"],
                )

        # RIGHT PREVIEW AND EXPORT PANEL
        with gr.Column(scale=4, min_width=300):

            with gr.Group(elem_classes=["mps-card"]):

                gr.HTML("""
                <div class="mps-preview-title">Preview & Gallery</div>
                <div class="mps-description">
                    Your finished composition appears here.
                </div>
                """)

                preview = gr.Image(
                    label="Final composition",
                    type="pil",
                    height=460,
                    interactive=False,
                )

                gr.Markdown(
                    "The preview updates when you create a composition."
                )

            with gr.Group(elem_classes=["mps-card"]):

                gr.HTML("""
                <div class="mps-preview-title">Export your result</div>
                <div class="mps-description">
                    Choose a format and prepare your image for download.
                </div>
                """)

                format_choice = gr.Radio(
                    choices=["PNG", "JPG"],
                    value="PNG",
                    label="Image format",
                )

                export_button = gr.Button(
                    "⬇️  Prepare Download",
                    variant="primary",
                    elem_classes=["primary-button"],
                )

                download_file = gr.File(
                    label="Your downloadable image",
                    interactive=False,
                )

                export_status = gr.Markdown(
                    "Create a composition before exporting.",
                    elem_classes=["mps-status"],
                )

            with gr.Group(elem_classes=["mps-card"]):
                gr.HTML("""
                <div class="mps-preview-title">Studio notes</div>
                <div class="mps-tip">
                    <strong>PNG</strong> — Best for preserving transparency.<br><br>
                    <strong>JPG</strong> — Best for general photography and sharing.<br><br>
                    Keep your original video as a backup.
                </div>
                """)

    gr.HTML("""
    <div class="mps-footer">
        Motion Photo Studio v3 · Creative photography workspace
    </div>
    """)

    # -----------------------------------------------------
    # EVENTS
    # -----------------------------------------------------

    def clear_previous_result():
        return None, None, "Create a new composition before exporting."

    extract_button.click(
        fn=extract_frames,
        inputs=[
            video_input,
            sample_interval,
            max_frames,
        ],
        outputs=[
            frames_state,
            labels_state,
            frame_selection,
            extraction_status,
        ],
    ).then(
        fn=lambda frames: frames,
        inputs=[frames_state],
        outputs=[frames_gallery],
    ).then(
        fn=clear_previous_result,
        inputs=[],
        outputs=[
            preview,
            composite_state,
            composition_status,
        ],
    ).then(
        fn=lambda: (
            None,
            "Create a new composition before exporting.",
        ),
        inputs=[],
        outputs=[download_file, export_status],
    )

    select_all_button.click(
        fn=select_all_frames,
        inputs=[labels_state],
        outputs=[frame_selection],
    )

    select_none_button.click(
        fn=select_no_frames,
        inputs=[labels_state],
        outputs=[frame_selection],
    )

    compose_button.click(
        fn=compose_frames,
        inputs=[
            frames_state,
            labels_state,
            frame_selection,
            composition_mode,
            opacity,
            remove_background_checkbox,
        ],
        outputs=[
            preview,
            composite_state,
            composition_status,
        ],
    ).then(
        fn=lambda: (
            None,
            "Composition updated. Prepare a new download.",
        ),
        inputs=[],
        outputs=[download_file, export_status],
    )

    export_button.click(
        fn=export_result,
        inputs=[
            composite_state,
            format_choice,
        ],
        outputs=[
            download_file,
            export_status,
        ],
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )
