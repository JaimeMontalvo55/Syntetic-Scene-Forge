from pathlib import Path
from io import BytesIO
import sys
import tempfile
import threading
import time
from zipfile import ZIP_DEFLATED, ZipFile
from math import ceil

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))
from synthetic_dataset_generator.config import GenerationConfig
from synthetic_dataset_generator.models import load_models, model_scale_report
from synthetic_dataset_generator.generation import (
    create_class_plan,
    generate,
    render_orientation_preview,
    render_preview,
)
from synthetic_dataset_generator.sources import extract_video_frames


st.set_page_config(page_title="Synthetic Scene Forge", layout="wide")
st.title("Synthetic Scene Forge")
st.caption("Standalone synthetic object-detection dataset generation from STL models")

if "model_files" not in st.session_state:
    st.session_state.model_files = {}
if "scenario_images" not in st.session_state:
    st.session_state.scenario_images = []
if "last_preview_change" not in st.session_state:
    st.session_state.last_preview_change = "initial"
if "generation_job" not in st.session_state:
    st.session_state.generation_job = None
if "generation_result" not in st.session_state:
    st.session_state.generation_result = None


def help_text(text: str) -> str:
    return text


def mark_preview_change(name: str) -> None:
    st.session_state.last_preview_change = name


def _output_files(output_dir: Path) -> set[Path]:
    if not output_dir.exists():
        return set()
    return {
        path for path in output_dir.rglob("*")
        if path.is_file()
    }


def _remove_generated_files(output_dir: Path, files_before: set[Path]) -> None:
    for path in _output_files(output_dir) - files_before:
        path.unlink(missing_ok=True)


def _folder_link(label: str, path: Path) -> str:
    return f"[{label}]({path.resolve().as_uri()})"


def _zip_directory(directory: Path) -> bytes:
    archive = BytesIO()
    with ZipFile(archive, "w", ZIP_DEFLATED) as zip_file:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                zip_file.write(path, path.relative_to(directory))
    return archive.getvalue()


def _show_downloads(output_path: Path) -> None:
    if not output_path.exists():
        return
    dataset_zip = _zip_directory(output_path)
    config_path = output_path / "generation_config.json"
    st.download_button(
        "Download complete dataset (ZIP)",
        data=dataset_zip,
        file_name="synthetic_dataset.zip",
        mime="application/zip",
        key="download_dataset_zip",
        help="Downloads images, labels, masks, metadata, COCO annotations, reports and configuration.",
    )
    if config_path.is_file():
        st.download_button(
            "Download generation configuration",
            data=config_path.read_bytes(),
            file_name="generation_config.json",
            mime="application/json",
            key="download_generation_config",
        )


def _dataset_is_ready(output_path: Path) -> bool:
    return (output_path / "reports" / "generation_report.json").is_file()


def _run_generation(job, config, models, classes, backgrounds, plan, background_mode):
    def update_progress(done, total, message):
        job["done"] = done
        job["total"] = total
        job["message"] = message

    try:
        job["result"] = generate(
            config,
            models,
            classes,
            backgrounds,
            plan,
            background_mode,
            stop=job["cancel_event"].is_set,
            progress=update_progress,
        )
    except (RuntimeError, ValueError, OSError) as error:
        job["error"] = str(error)
    finally:
        if job["cancel_event"].is_set():
            _remove_generated_files(config.output_dir, job["files_before"])
        job["finished"] = True


with st.sidebar:
    st.header("1. Project")
    output_dir = Path(st.text_input(
        "Output folder",
        "outputs/dataset",
        help=help_text("Folder where images, labels, depth data and data.yaml will be written."),
    ))
    st.header("2. STL models")
    uploads = st.file_uploader(
        "Upload STL files",
        type=["stl"],
        accept_multiple_files=True,
        help=help_text("Upload one or more 3D models. Each model can appear as an independent class."),
    )
    current_model_files = {
        Path(upload.name).stem: upload
        for upload in uploads or []
    }
    previous_model_names = set(st.session_state.model_files)
    if current_model_files != st.session_state.model_files:
        st.session_state.model_files = current_model_files
    model_names = list(st.session_state.model_files)
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = model_names
    else:
        retained_models = [
            name for name in st.session_state.selected_models
            if name in model_names
        ]
        new_models = [
            name for name in model_names
            if name not in previous_model_names
        ]
        if not previous_model_names and model_names:
            retained_models = model_names
        else:
            retained_models.extend(
                name for name in new_models
                if name not in retained_models
            )
        st.session_state.selected_models = retained_models
    selected = st.multiselect(
        "Included models",
        model_names,
        key="selected_models",
        help=help_text("Select the models that may appear in scenes."),
    )
    class_names = {}
    for name in selected:
        class_names[name] = st.text_input(
            f"Class for {name}",
            name,
            key=f"class_{name}",
            help=help_text("Name that will be written to data.yaml for this model."),
        )

preview_expander = st.sidebar.expander("5. Dataset preview", expanded=True)

st.header("3. Scene sources")
mode = st.radio(
    "Generation method",
    ["Random backgrounds", "Videos and images", "Combined"],
    horizontal=True,
    help=help_text(
        "Random backgrounds uses randomly generated colors. Images are only "
        "used with Videos and images or Combined."
    ),
)

scenario_uploads = []
video_uploads = []
if mode in {"Videos and images", "Combined"}:
    source_col, video_col = st.columns(2)
    with source_col:
        scenario_uploads = st.file_uploader(
            "Scene images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help=help_text("Images used as backgrounds behind the models."),
        )
    with video_col:
        video_uploads = st.file_uploader(
            "Videos",
            type=["mp4", "avi", "mov"],
            accept_multiple_files=True,
            help=help_text("Frames will be extracted and used as scene backgrounds."),
        )
    if scenario_uploads:
        st.session_state.scenario_images = scenario_uploads
else:
    st.info("Flat randomly colored backgrounds will be used. No images will be loaded.")
    st.session_state.scenario_images = []

config_panel = st.container()
config_panel.header("4. Configuration")
config_cols = config_panel.columns(4)
with config_cols[0]:
    num_scenes = st.number_input("Number of images", min_value=1, value=1500, step=1, help=help_text("Target number of valid images."))
    width = st.number_input("Width", min_value=1, value=640, help=help_text("Final image width in pixels."))
    height = st.number_input("Height", min_value=1, value=480, help=help_text("Final image height in pixels."))
with config_cols[1]:
    train_ratio = st.number_input("Train ratio", min_value=0.0, max_value=1.0, value=0.80, step=0.1, format="%.2f", help=help_text("Proportion of images assigned to train."))
    val_ratio = st.number_input("Validation ratio", min_value=0.0, max_value=1.0, value=0.20, step=0.1, format="%.2f", help=help_text("Proportion of images assigned to validation."))
    test_ratio = st.number_input("Test ratio", min_value=0.0, max_value=1.0, value=0.0, step=0.1, format="%.2f", help=help_text("Proportion of images assigned to test."))
    negative_ratio = st.number_input("Images without objects", min_value=0.0, max_value=1.0, value=0.0, step=0.1, format="%.2f", help=help_text("Proportion of background-only images, useful for reducing false positives."))
    split_total = train_ratio + val_ratio + test_ratio
    if abs(split_total - 1.0) > 0.001:
        config_panel.warning(f"Ratios must add up to 1. They currently total {split_total:.3f}.")
with config_cols[2]:
    min_separation = st.number_input("Minimum separation (px)", min_value=0, value=30, help=help_text("Minimum distance between boxes before rejecting scenes with nearby objects."))
    seed = st.number_input("Seed", value=42, step=1, help=help_text("Seed used to reproduce the same random sequence."))
    save_masks = st.checkbox("Save 2D masks", value=True, help=help_text("Save one PNG mask per image, using 0 for background and class ID + 1 for each object."))
with config_cols[3]:
    distance_mean = st.number_input("Mean distance (m)", min_value=0.1, value=3.5, step=0.1, format="%.2f", key="distance_mean", on_change=mark_preview_change, args=("distance_mean",), help=help_text("Mean distance between the camera and the object center."))
    distance_min = st.number_input("Minimum distance (m)", min_value=0.1, value=3.0, step=0.1, format="%.2f", key="distance_min", on_change=mark_preview_change, args=("distance_min",), help=help_text("Minimum distance between the camera and the object."))
    distance_max = st.number_input("Maximum distance (m)", min_value=0.1, value=4.0, step=0.1, format="%.2f", key="distance_max", on_change=mark_preview_change, args=("distance_max",), help=help_text("Maximum distance between the camera and the object."))
    brightness_min = st.number_input(
        "Minimum brightness",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.1,
        format="%.2f",
        key="brightness_min",
        on_change=mark_preview_change,
        args=("brightness_min",),
        help=help_text("Minimum object material brightness. 0 is very dark and 1 preserves full brightness."),
    )
    brightness_max = st.number_input(
        "Maximum brightness",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.1,
        format="%.2f",
        key="brightness_max",
        on_change=mark_preview_change,
        args=("brightness_max",),
        help=help_text("Maximum object material brightness. A random value is chosen between both limits."),
    )
    st.caption("STL files are automatically normalized to meters. Emission is disabled.")

config_panel.subheader("Object distribution per image")
available_object_count = len(selected)
default_object_ratios = {count: 0.0 for count in range(1, available_object_count + 1)}
if available_object_count:
    default_object_ratios[1] = 1.0 if available_object_count == 1 else 0.75
if available_object_count >= 2:
    default_object_ratios[2] = 0.25 if available_object_count == 2 else 0.15
if available_object_count >= 3:
    for count in range(3, available_object_count + 1):
        default_object_ratios[count] = 0.10 / (available_object_count - 2)
if st.session_state.get("ratio_model_count") != available_object_count:
    for count in range(1, available_object_count + 1):
        st.session_state[f"object_ratio_{count}"] = default_object_ratios[count]
    st.session_state.ratio_model_count = available_object_count
ratio_cols = config_panel.columns(available_object_count) if available_object_count else []
object_ratios = {}
for index, count in enumerate(range(1, available_object_count + 1)):
    with ratio_cols[index]:
        object_ratios[count] = st.number_input(
            f"{count} object(s)",
            min_value=0.0,
            max_value=1.0,
            value=default_object_ratios[count],
            step=0.1,
            format="%.2f",
            key=f"object_ratio_{count}",
            on_change=mark_preview_change,
            args=(f"object_ratio_{count}",),
            help=help_text(f"Target proportion of scenes containing exactly {count} object(s)."),
        )
ratio_total = sum(object_ratios.values())
if abs(ratio_total - 1.0) > 0.001:
    config_panel.warning(f"Object distribution must add up to 1. It currently totals {ratio_total:.3f}.")
if distance_min > distance_max:
    config_panel.warning("Minimum distance cannot be greater than maximum distance.")
elif not distance_min <= distance_mean <= distance_max:
    config_panel.warning("Mean distance must be between minimum and maximum distance.")
if not brightness_min <= brightness_max:
    config_panel.warning("Minimum brightness cannot be greater than maximum brightness.")
if negative_ratio + 0.0 > 1.0:
    config_panel.warning("The proportion of images without objects must be between 0 and 1.")

rotation_axes_by_class = {}
rotation_step_by_class_deg = {}
initial_rotation_by_class_deg = {}
config_panel.subheader("Rotation per object")
config_panel.markdown(
    "**STL local-axis reference:** "
    ":red[**X** = left/right] · "
    ":green[**Y** = down/up] · "
    ":blue[**Z** = depth forward/backward]. "
    "These axes belong to each model's original orientation and may not match the screen orientation."
)
config_panel.caption("Initial orientation corrects the model before random rotations are applied. Angles are expressed in degrees.")
if selected:
    rotation_cols = config_panel.columns(min(3, len(selected)))
    for index, name in enumerate(selected):
        with rotation_cols[index % len(rotation_cols)]:
            st.markdown(f"**{name}**")
            rotation_axes_by_class[class_names[name]] = st.multiselect(
                "Random rotation axes",
                ["x", "y", "z"],
                default=["x", "y", "z"],
                key=f"rotation_axes_{name}",
                on_change=mark_preview_change,
                args=(f"rotation_axes_{name}",),
                help=help_text("X rotates around the local left/right axis, Y around the local down/up axis, and Z around the local depth axis."),
            )
            rotation_step_by_class_deg[class_names[name]] = st.number_input(
                "Angular step (degrees)",
                min_value=0.1,
                max_value=360.0,
                value=45.0,
                step=0.1,
                format="%.1f",
                key=f"rotation_step_{name}",
                on_change=mark_preview_change,
                args=(f"rotation_step_{name}",),
                help=help_text("Spacing between possible angles; for example, 45 degrees generates 0, 45, 90..."),
            )
            initial_rotation_by_class_deg[class_names[name]] = {
                axis: st.number_input(
                    f"Initial {axis.upper()} orientation (degrees)",
                    min_value=-360.0,
                    max_value=360.0,
                    value=0.0,
                    step=1.0,
                    format="%.1f",
                    key=f"initial_rotation_{axis}_{name}",
                    on_change=mark_preview_change,
                    args=(f"initial_rotation_{axis}_{name}",),
                    help=help_text(
                        f"Fixed correction around the local {axis.upper()} axis. "
                        "Applied before random rotation."
                    ),
                )
                for axis in ("x", "y", "z")
            }

config_panel.subheader("Angular coverage")
config_panel.info(
    "Each STL uses the axes and interval configured above. "
    "As a reference, 45° provides 8 angles per selected axis and 15° provides 24. "
    "The image recommendation depends on the selected axes and steps."
)
if selected:
    angular_combinations = []
    for name in selected:
        axes = rotation_axes_by_class[class_names[name]]
        step = rotation_step_by_class_deg[class_names[name]]
        angles_per_axis = max(1, ceil(360 / step))
        angular_combinations.append(angles_per_axis ** len(axes))
    minimum_angular_images = sum(angular_combinations)
    recommended_angular_images = minimum_angular_images * 2
    if num_scenes < minimum_angular_images:
        config_panel.warning(
            f"There are {num_scenes} images for {len(selected)} STL files. "
            f"At least {minimum_angular_images} images are needed to cover "
            f"the configured angular combinations once."
        )
    elif num_scenes < recommended_angular_images:
        config_panel.info(
            f"Minimum coverage is met. {recommended_angular_images} images are "
            f"recommended to compensate for random sampling."
        )

config_panel.subheader("Lateral position")
x_range = config_panel.slider("X range", -2.0, 2.0, (-0.9, 0.9), step=0.1, key="x_range", on_change=mark_preview_change, args=("x_range",), help=help_text("Horizontal displacement range for objects."))
y_range = config_panel.slider("Y range", -2.0, 2.0, (-0.65, 0.65), step=0.1, key="y_range", on_change=mark_preview_change, args=("y_range",), help=help_text("Vertical displacement range for objects."))
if x_range[0] > x_range[1]:
    config_panel.warning("The minimum X limit cannot be greater than the maximum.")
if y_range[0] > y_range[1]:
    config_panel.warning("The minimum Y limit cannot be greater than the maximum.")


def load_uploaded_inputs():
    work = Path(tempfile.mkdtemp(prefix="synthetic_dataset_"))
    paths = {}
    for name in selected:
        target = work / f"{name}.stl"
        target.write_bytes(st.session_state.model_files[name].getvalue())
        paths[name] = target
    models = load_models(paths)
    models_for_generation = {class_names[name]: models[name] for name in selected}
    classes = {class_names[name]: index for index, name in enumerate(selected)}
    backgrounds = [Image.open(upload).convert("RGB") for upload in st.session_state.scenario_images]
    if mode in {"Videos and images", "Combined"}:
        for video in video_uploads:
            video_path = work / video.name
            video_path.write_bytes(video.getvalue())
            frame_paths = extract_video_frames(video_path, work / "frames")
            backgrounds.extend(Image.open(path).convert("RGB") for path in frame_paths)
    return work, models_for_generation, classes, backgrounds


with st.expander("Visualizador de ejes locales", expanded=True):
    st.caption("Red = X, green = Y and blue = Z. The axes are shown relative to the STL's initial orientation.")
    show_axis_viewer = st.checkbox(
        "Show STL orientation",
        value=True,
        key="show_axis_viewer",
        help="Render each model centered with its local axes and configured initial orientation.",
    )
    if show_axis_viewer and selected:
        try:
            _, orientation_models, _, _ = load_uploaded_inputs()
            viewer_cols = st.columns(min(3, len(selected)))
            for index, name in enumerate(selected):
                with viewer_cols[index % len(viewer_cols)]:
                    st.caption(class_names[name])
                    st.image(
                        render_orientation_preview(
                            GenerationConfig(output_dir=output_dir),
                            orientation_models[class_names[name]],
                            initial_rotation_by_class_deg[class_names[name]],
                        ),
                        use_container_width=True,
                    )
        except (RuntimeError, ValueError, OSError) as error:
            st.warning(f"The axis viewer could not be displayed: {error}")


preview_error = None
generation_active = (
    st.session_state.generation_job is not None
    and not st.session_state.generation_job["finished"]
)
with preview_expander:
    if generation_active:
        st.info("The preview is paused while the dataset is generated to avoid OpenGL conflicts.")
    elif selected:
        try:
            _, preview_models, preview_classes, preview_backgrounds = load_uploaded_inputs()
            preview_distance_min = min(distance_min, distance_max)
            preview_distance_max = max(distance_min, distance_max)
            preview_distance_mean = min(
                max(distance_mean, preview_distance_min),
                preview_distance_max,
            )
            preview_brightness_min = min(brightness_min, brightness_max)
            preview_brightness_max = max(brightness_min, brightness_max)
            preview_config = GenerationConfig(
                output_dir=output_dir,
                width=width,
                height=height,
                object_distance_mean_m=preview_distance_mean,
                object_distance_min_m=preview_distance_min,
                object_distance_max_m=preview_distance_max,
                object_brightness_min=preview_brightness_min,
                object_brightness_max=preview_brightness_max,
                negative_ratio=0.0,
                save_masks=save_masks,
                rotation_axes_by_class=rotation_axes_by_class,
                rotation_step_by_class_deg=rotation_step_by_class_deg,
                initial_rotation_by_class_deg=initial_rotation_by_class_deg,
                x_min=x_range[0],
                x_max=x_range[1],
                y_min=y_range[0],
                y_max=y_range[1],
            )
            preview_values = {
                "distance_mean": distance_mean,
                "distance_min": distance_min,
                "distance_max": distance_max,
                "brightness_min": brightness_min,
                "brightness_max": brightness_max,
            }
            last_change = st.session_state.last_preview_change
            preview_values = {
                "distance": preview_values.get(last_change, preview_distance_mean)
                if last_change.startswith("distance_")
                else preview_distance_mean,
                "brightness": preview_values.get(last_change, preview_brightness_max)
                if last_change.startswith("brightness_")
                else preview_brightness_max,
                "x": x_range[1],
                "y": y_range[1],
            }
            st.caption(
                f"Distancia {preview_values['distance']:.2f} m · "
                f"brightness {preview_values['brightness']:.2f}"
            )
            preview_mode = "random_color" if mode == "Random backgrounds" else "mixed" if mode == "Combined" else "images"
            for object_count in range(1, len(preview_classes) + 1):
                object_names = list(preview_classes)[:object_count]
                preview = render_preview(
                    preview_config,
                    preview_models,
                    preview_classes,
                    object_names,
                    preview_backgrounds,
                    preview_mode,
                    preview_values,
                )
                st.image(
                    preview,
                    caption=f"{object_count} object(s)",
                    use_container_width=True,
                )
        except (RuntimeError, ValueError, OSError) as error:
            preview_error = str(error)
    if preview_error:
        st.warning(f"The preview could not be updated: {preview_error}")
    elif not selected:
        st.info("Upload at least one STL to see an example scene.")

st.header("6. Generation")
if st.button("Generate dataset", type="primary"):
    if st.session_state.generation_job and not st.session_state.generation_job["finished"]:
        st.warning("A generation is already in progress.")
    elif not selected:
        st.error("Upload and select at least one STL.")
    elif len(set(class_names.values())) != len(class_names):
        st.error("Each model must have a unique class name.")
    elif abs(split_total - 1.0) > 0.001:
        st.error("Train, validation and test ratios must add up to 1.")
    elif abs(ratio_total - 1.0) > 0.001:
        st.error("Object distribution must add up to 1.")
    elif distance_min > distance_max:
        st.error("Minimum distance cannot be greater than maximum distance.")
    elif not distance_min <= distance_mean <= distance_max:
        st.error("Mean distance must be between minimum and maximum distance.")
    elif brightness_min > brightness_max:
        st.error("Minimum brightness cannot be greater than maximum brightness.")
    elif not 0.0 <= negative_ratio <= 1.0:
        st.error("The proportion of images without objects must be between 0 and 1.")
    elif x_range[0] > x_range[1] or y_range[0] > y_range[1]:
        st.error("Minimum X and Y limits cannot be greater than their maximums.")
    elif mode != "Random backgrounds" and not st.session_state.scenario_images and not video_uploads:
        st.error("Upload at least one image or video.")
    else:
        try:
            _, models, classes, backgrounds = load_uploaded_inputs()
            plan = create_class_plan(list(classes), object_ratios, total=num_scenes)
            config = GenerationConfig(
                output_dir=output_dir,
                width=width,
                height=height,
                num_scenes=num_scenes,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                negative_ratio=negative_ratio,
                save_masks=save_masks,
                min_separation_px=min_separation,
                object_distance_mean_m=distance_mean,
                object_distance_min_m=distance_min,
                object_distance_max_m=distance_max,
                object_brightness_min=brightness_min,
                object_brightness_max=brightness_max,
                rotation_axes_by_class=rotation_axes_by_class,
                rotation_step_by_class_deg=rotation_step_by_class_deg,
                initial_rotation_by_class_deg=initial_rotation_by_class_deg,
                x_min=x_range[0],
                x_max=x_range[1],
                y_min=y_range[0],
                y_max=y_range[1],
                seed=seed,
            )
            background_mode = "random_color" if mode == "Random backgrounds" else "mixed" if mode == "Combined" else "images"
            job = {
                "cancel_event": threading.Event(),
                "done": 0,
                "total": num_scenes,
                "message": "Starting generation.",
                "result": None,
                "error": None,
                "finished": False,
                "files_before": _output_files(config.output_dir),
                "output_dir": config.output_dir,
            }
            job["thread"] = threading.Thread(
                target=_run_generation,
                args=(job, config, models, classes, backgrounds, plan, background_mode),
                daemon=True,
            )
            st.session_state.generation_job = job
            st.session_state.generation_result = None
            job["thread"].start()
            st.rerun()
        except (RuntimeError, ValueError, OSError) as error:
            st.error(str(error))

job = st.session_state.generation_job
if job:
    if not job["finished"]:
        progress_value = job["done"] / job["total"] if job["total"] else 0.0
        st.progress(progress_value, text=job["message"])
        if st.button("Stop and delete generated data", type="secondary"):
            job["cancel_event"].set()
            st.info("Canceling generation and deleting its files...")
            time.sleep(0.2)
            st.rerun()
        time.sleep(0.5)
        st.rerun()
    elif job["error"]:
        st.error(job["error"])
        st.session_state.generation_job = None
    elif job["result"] is not None:
        if job["cancel_event"].is_set():
            st.warning("Generation canceled. Data created during this run has been deleted.")
        else:
            st.success(f"Generation complete: {job['result']['generated']} images.")
            st.json(job["result"])
            output_path = Path(job["output_dir"])
            st.session_state.generation_result = {
                "result": job["result"],
                "output_dir": str(output_path),
            }
            st.subheader("Generated folders and files")
            coco_links = [
                f"- {_folder_link(path.name, path)}"
                for path in sorted(output_path.glob("annotations_*.json"))
            ]
            st.markdown(
                "\n".join([
                    f"- {_folder_link('Dataset root folder', output_path)}",
                    f"- {_folder_link('Images', output_path / 'images')}",
                    f"- {_folder_link('Labels', output_path / 'labels')}",
                    f"- {_folder_link('2D masks', output_path / 'masks')}",
                    f"- {_folder_link('Depth and metadata', output_path / 'depth')}",
                    f"- {_folder_link('Reports', output_path / 'reports')}",
                    f"- {_folder_link('data.yaml', output_path / 'data.yaml')}",
                    f"- {_folder_link('Generation configuration', output_path / 'generation_config.json')}",
                ] + coco_links)
            )
        st.session_state.generation_result = job["result"]
        st.session_state.generation_job = None

st.header("7. Results and export")
st.info("The output folder contains images/train, images/val, images/test, labels, masks, depth, COCO annotations, reports, data.yaml and generation_config.json.")
st.caption("Default path: C:\\Users\\jaime\\synthetic-dataset-generator\\outputs\\dataset")
saved_generation = st.session_state.get("generation_result")
saved_output_path = (
    Path(saved_generation["output_dir"])
    if saved_generation
    else output_dir
)
if saved_generation and _dataset_is_ready(saved_output_path):
    st.success("The generated dataset is ready to download.")
    _show_downloads(saved_output_path)
elif _dataset_is_ready(output_dir):
    st.success("A completed dataset was found in the output folder.")
    _show_downloads(output_dir)
else:
    st.caption("Generate a dataset to enable the download buttons.")
