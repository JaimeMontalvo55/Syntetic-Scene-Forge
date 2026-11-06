import json
import random
import time
import colorsys
import os
from pathlib import Path
from collections.abc import Callable

if os.name != "nt" and "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
from OpenGL.error import GLError
import pyrender
import trimesh
from PIL import Image

from .config import GenerationConfig
from .export import (
    prepare_output,
    write_coco_annotations,
    write_data_yaml,
    write_generation_config,
    write_generation_report,
)
from .rendering import create_offscreen_renderer


CAMERA_Z_M = 3.5


def _delete_renderer(renderer) -> None:
    try:
        renderer.delete()
    except GLError:
        pass


def _segmentation_colors(classes: dict[str, int]) -> dict[str, list[int]]:
    total = max(len(classes), 1)
    colors = {}
    for name, class_id in classes.items():
        red, green, blue = colorsys.hsv_to_rgb((class_id % total) / total, 0.85, 1.0)
        colors[name] = [round(red * 255), round(green * 255), round(blue * 255)]
    return colors


def _material(
    config: GenerationConfig,
    brightness: float | None = None,
) -> pyrender.MetallicRoughnessMaterial:
    if brightness is None:
        brightness = random.uniform(
            config.object_brightness_min,
            config.object_brightness_max,
        )
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[random.uniform(0.25, 0.9) * brightness for _ in range(3)] + [1.0],
        metallicFactor=0.0,
        roughnessFactor=random.uniform(0.65, 0.98),
    )


def _scene(config: GenerationConfig) -> pyrender.Scene:
    scene = pyrender.Scene(bg_color=[0.5, 0.5, 0.5, 1.0], ambient_light=[0.45] * 3)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=random.uniform(2.0, 4.0))
    light_pose = np.eye(4)
    light_pose[:3, 3] = [1.0, -1.0, 3.0]
    scene.add(light, pose=light_pose)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=config.width / config.height)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0.0, 0.0, 3.5]
    scene.add(camera, pose=camera_pose)
    return scene


def _transform(
    mesh: trimesh.Trimesh,
    config: GenerationConfig,
    distance: float | None = None,
    x: float | None = None,
    y: float | None = None,
    rotation_axes: list[str] | None = None,
    rotation_step_deg: float | None = None,
    initial_rotation_deg: dict[str, float] | None = None,
) -> tuple[np.ndarray, float]:
    rotation = np.eye(4)
    axis_vectors = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
    for axis_name in ("x", "y", "z"):
        angle = (initial_rotation_deg or {}).get(axis_name, 0.0)
        rotation = rotation @ trimesh.transformations.rotation_matrix(
            np.radians(angle),
            axis_vectors[axis_name],
        )
    selected_axes = rotation_axes or []
    step = rotation_step_deg or 360.0
    angles = np.arange(0.0, 360.0, step)
    for axis_name in selected_axes:
        angle = random.choice(angles.tolist()) if len(angles) else 0.0
        rotation = rotation @ trimesh.transformations.rotation_matrix(
            np.radians(angle),
            axis_vectors[axis_name],
        )
    translation = np.eye(4)
    if distance is None:
        distance = random.triangular(
            config.object_distance_min_m,
            config.object_distance_max_m,
            config.object_distance_mean_m,
        )
    target_center = np.array([
        random.uniform(config.x_min, config.x_max) if x is None else x,
        random.uniform(config.y_min, config.y_max) if y is None else y,
        CAMERA_Z_M - distance,
    ])
    rotated_center = rotation @ np.append(mesh.bounding_box.centroid, 1.0)
    translation[:3, 3] = target_center - rotated_center[:3]
    transform = translation @ rotation
    center = transform @ np.append(mesh.bounding_box.centroid, 1.0)
    return transform, float(CAMERA_Z_M - center[2])


def _boxes(segmentation: np.ndarray, classes: dict[str, int], colors: dict[str, list[int]], width: int, height: int):
    labels, masks, boxes = [], {}, {}
    pixel_counts = {}
    for name, class_id in classes.items():
        mask = np.all(segmentation[:, :, :3] == np.array(colors[name], dtype=np.uint8), axis=2)
        masks[name] = mask
        coordinates = np.argwhere(mask)
        pixel_counts[name] = len(coordinates)
        if len(coordinates) < 20:
            continue
        y_min, x_min = coordinates.min(axis=0)
        y_max, x_max = coordinates.max(axis=0)
        labels.append(f"{class_id} {((x_min + x_max) / 2 / width):.6f} {((y_min + y_max) / 2 / height):.6f} {((x_max - x_min + 1) / width):.6f} {((y_max - y_min + 1) / height):.6f}")
        boxes[name] = (int(x_min), int(y_min), int(x_max), int(y_max))
    return labels, masks, boxes, pixel_counts


def _coco_scene_annotations(
    image_id: int,
    file_name: str,
    width: int,
    height: int,
    classes: dict[str, int],
    masks: dict[str, np.ndarray],
    boxes: dict[str, tuple[int, int, int, int]],
) -> tuple[dict, list[dict]]:
    image = {"id": image_id, "file_name": file_name, "width": width, "height": height}
    annotations = []
    for annotation_index, (name, (x_min, y_min, x_max, y_max)) in enumerate(boxes.items()):
        mask = masks[name]
        annotations.append({
            "id": image_id * 100000 + annotation_index,
            "image_id": image_id,
            "category_id": classes[name],
            "bbox": [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
            "area": int(mask.sum()),
            "iscrowd": 0,
        })
    return image, annotations


def _separated(boxes: dict[str, tuple[int, int, int, int]], minimum: int) -> bool:
    names = list(boxes)
    for index, first in enumerate(names):
        ax1, ay1, ax2, ay2 = boxes[first]
        for second in names[index + 1:]:
            bx1, by1, bx2, by2 = boxes[second]
            distance = ((max(ax1 - bx2, bx1 - ax2, 0)) ** 2 + (max(ay1 - by2, by1 - ay2, 0)) ** 2) ** 0.5
            if distance < minimum:
                return False
    return True


def create_class_plan(
    class_names: list[str],
    object_ratios: dict[int, float] | None = None,
    total: int = 1500,
) -> list[list[str]]:
    if not class_names:
        raise ValueError("Debe existir al menos una clase.")
    ratios = object_ratios or {1: 1.0}
    valid_counts = sorted(count for count, ratio in ratios.items() if ratio > 0 and count <= len(class_names))
    if not valid_counts:
        raise ValueError("La distribución de objetos no es compatible con los modelos seleccionados.")
    ratio_sum = sum(ratios[count] for count in valid_counts)
    raw_quotas = {count: total * ratios[count] / ratio_sum for count in valid_counts}
    quotas = {count: int(raw_quotas[count]) for count in valid_counts}
    remainder = total - sum(quotas.values())
    for count in sorted(valid_counts, key=lambda item: raw_quotas[item] - quotas[item], reverse=True)[:remainder]:
        quotas[count] += 1
    plan: list[list[str]] = []
    for count in valid_counts:
        combinations = [list(class_names)] if count == len(class_names) else []
        if not combinations:
            combinations = [
                list(pair)
                for index, first in enumerate(class_names)
                for pair in [class_names[index:index + count]]
                if len(pair) == count
            ]
            if count == 2 and len(class_names) > 2:
                combinations = [
                    [class_names[index], class_names[other]]
                    for index in range(len(class_names))
                    for other in range(index + 1, len(class_names))
                ]
        for index in range(quotas[count]):
            plan.append(combinations[index % len(combinations)])
    random.shuffle(plan)
    return plan


def _random_color_background(width: int, height: int) -> Image.Image:
    color = tuple(int(value) for value in np.random.randint(0, 256, size=3))
    return Image.new("RGB", (width, height), color)


def _compose_background(
    rgb: np.ndarray,
    masks: dict[str, np.ndarray],
    backgrounds: list[Image.Image],
    mode: str,
    width: int,
    height: int,
) -> np.ndarray:
    if mode == "random_color" or (mode == "mixed" and random.random() < 0.5):
        background = np.asarray(_random_color_background(width, height)).copy()
    else:
        if not backgrounds:
            raise RuntimeError("No hay imágenes de escenario disponibles.")
        background = np.asarray(
            random.choice(backgrounds)
            .resize((width, height), Image.Resampling.LANCZOS)
            .convert("RGB")
        ).copy()
    for name in masks:
        background[masks[name]] = rgb[masks[name]]
    return background


def render_preview(
    config,
    models,
    classes,
    object_names,
    backgrounds=None,
    background_mode="random_color",
    preview_values: dict[str, float] | None = None,
) -> Image.Image:
    colors = _segmentation_colors(classes)
    preview_values = preview_values or {}
    renderer = create_offscreen_renderer(config.width, config.height)
    try:
        max_preview_attempts = 100
        for _ in range(max_preview_attempts):
            scene = _scene(config)
            nodes = {}
            for name in object_names:
                mesh = pyrender.Mesh.from_trimesh(
                    models[name],
                    material=_material(config, preview_values.get("brightness")),
                    smooth=False,
                )
                fixed_x = preview_values.get("x") if len(object_names) == 1 else None
                fixed_y = preview_values.get("y") if len(object_names) == 1 else None
                transform, _ = _transform(
                    models[name],
                    config,
                    distance=preview_values.get("distance"),
                    x=fixed_x,
                    y=fixed_y,
                    rotation_axes=config.rotation_axes_by_class.get(name, ["x", "y", "z"]),
                    rotation_step_deg=config.rotation_step_by_class_deg.get(name, 45.0),
                    initial_rotation_deg=config.initial_rotation_by_class_deg.get(name),
                )
                nodes[scene.add(mesh, pose=transform)] = colors[name]
            rgb, _ = renderer.render(scene)
            segmentation, _ = renderer.render(scene, flags=pyrender.RenderFlags.SEG, seg_node_map=nodes)
            _, masks, boxes, _ = _boxes(segmentation, classes, colors, config.width, config.height)
            if _separated(boxes, config.min_separation_px):
                return Image.fromarray(
                    _compose_background(
                        rgb,
                        masks,
                        backgrounds or [],
                        background_mode,
                        config.width,
                        config.height,
                    )
                )
        raise RuntimeError(
            "Could not create a preview satisfying the configured minimum separation. "
            "Increase the position ranges or reduce the minimum separation."
        )
    finally:
        _delete_renderer(renderer)


def _axis_mesh(axis: str, length: float, color: list[float]) -> trimesh.Trimesh:
    direction = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}[axis]
    shaft = trimesh.creation.cylinder(
        radius=max(length * 0.018, 0.008),
        height=length * 0.82,
        sections=12,
    )
    shaft.apply_translation([0.0, 0.0, length * 0.41])
    head = trimesh.creation.cone(
        radius=max(length * 0.055, 0.02),
        height=length * 0.18,
        sections=12,
    )
    head.apply_translation([0.0, 0.0, length * 0.91])
    axis_mesh = trimesh.util.concatenate([shaft, head])
    axis_mesh.apply_transform(
        trimesh.geometry.align_vectors([0.0, 0.0, 1.0], direction)
    )
    axis_mesh.visual.face_colors = [*color, 255]
    return axis_mesh


def render_orientation_preview(
    config: GenerationConfig,
    mesh: trimesh.Trimesh,
    initial_rotation_deg: dict[str, float] | None = None,
) -> Image.Image:
    """Render one STL centered with its local X/Y/Z axes overlaid."""
    width = max(320, config.width)
    height = max(280, config.height)
    renderer = create_offscreen_renderer(width, height)
    try:
        scene = pyrender.Scene(bg_color=[0.08, 0.08, 0.08, 1.0], ambient_light=[0.5] * 3)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=width / height)
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = [0.0, 0.0, 3.2]
        scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [1.5, -1.5, 3.0]
        scene.add(light, pose=light_pose)

        centered = mesh.copy()
        centered.apply_translation(-centered.bounding_box.centroid)
        rotation, _ = _transform(
            centered,
            config,
            distance=3.2,
            x=0.0,
            y=0.0,
            rotation_axes=[],
            initial_rotation_deg=initial_rotation_deg,
        )
        centered.apply_transform(np.block([
            [rotation[:3, :3], np.zeros((3, 1))],
            [np.zeros((1, 3)), np.ones((1, 1))],
        ]))
        extent = max(float(np.max(centered.extents)), 0.1)
        centered.apply_scale(1.35 / extent)
        model_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.72, 0.76, 0.82, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.75,
        )
        scene.add(pyrender.Mesh.from_trimesh(centered, material=model_material, smooth=False))

        axis_length = 1.05
        axis_colors = {"x": [0.95, 0.15, 0.15], "y": [0.2, 0.9, 0.25], "z": [0.2, 0.45, 1.0]}
        for axis, color in axis_colors.items():
            axis_material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[*color, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.55,
            )
            scene.add(pyrender.Mesh.from_trimesh(_axis_mesh(axis, axis_length, color), material=axis_material, smooth=False))
        color, _ = renderer.render(scene)
        return Image.fromarray(color)
    finally:
        _delete_renderer(renderer)


def generate(config: GenerationConfig, models, classes, backgrounds, plan, background_mode="images", stop: Callable[[], bool] | None = None, progress: Callable[[int, int, str], None] | None = None) -> dict:
    if config.width < 1 or config.height < 1:
        raise ValueError("El ancho y el alto deben ser mayores que cero.")
    if config.num_scenes < 1:
        raise ValueError("El número de imágenes debe ser mayor que cero.")
    if config.min_separation_px < 0:
        raise ValueError("La separación mínima no puede ser negativa.")
    if not 0.0 <= config.negative_ratio <= 1.0:
        raise ValueError("La proporción de imágenes negativas debe estar entre 0 y 1.")
    if not 0.0 <= config.train_ratio <= 1.0:
        raise ValueError("La proporción train debe estar entre 0 y 1.")
    if not 0.0 <= config.val_ratio <= 1.0:
        raise ValueError("La proporción val debe estar entre 0 y 1.")
    if not 0.0 <= config.test_ratio <= 1.0:
        raise ValueError("La proporción test debe estar entre 0 y 1.")
    if not np.isclose(
        config.train_ratio + config.val_ratio + config.test_ratio,
        1.0,
        atol=0.001,
    ):
        raise ValueError("Las proporciones train, val y test deben sumar 1.")
    if config.x_min > config.x_max or config.y_min > config.y_max:
        raise ValueError("Los límites mínimos no pueden ser mayores que los máximos.")
    if config.object_distance_min_m <= 0 or config.object_distance_max_m <= 0:
        raise ValueError("Las distancias deben ser mayores que cero.")
    if not plan:
        raise ValueError("El plan de generación no puede estar vacío.")
    if not classes:
        raise ValueError("Debe existir al menos una clase.")
    unknown_models = set(classes) - set(models)
    if unknown_models:
        raise ValueError(f"No hay modelos cargados para: {', '.join(sorted(unknown_models))}.")
    unknown_plan_names = {
        name
        for scene in plan
        for name in scene
        if name not in classes
    }
    if unknown_plan_names:
        raise ValueError(
            f"El plan contiene clases no configuradas: {', '.join(sorted(unknown_plan_names))}."
        )
    if background_mode != "random_color" and not backgrounds:
        raise RuntimeError("No hay fondos disponibles.")
    prepare_output(config.output_dir)
    write_data_yaml(config.output_dir, classes)
    write_generation_config(config.output_dir, config, classes)
    random.seed(config.seed)
    np.random.seed(config.seed)
    if not config.object_distance_min_m <= config.object_distance_mean_m <= config.object_distance_max_m:
        raise ValueError("La distancia media debe estar entre la distancia mínima y máxima.")
    if not 0.0 <= config.object_brightness_min <= config.object_brightness_max <= 1.0:
        raise ValueError("El rango de iluminación debe estar entre 0 y 1.")
    valid_axes = {"x", "y", "z"}
    for name in classes:
        axes = config.rotation_axes_by_class.get(name, ["x", "y", "z"])
        if any(axis not in valid_axes for axis in axes):
            raise ValueError(f"Los ejes de rotación de {name} no son válidos.")
        step = config.rotation_step_by_class_deg.get(name, 45.0)
        if step <= 0 or step > 360:
            raise ValueError(f"El paso de rotación de {name} debe estar entre 0 y 360 grados.")
    colors = _segmentation_colors(classes)
    coco = {
        split: {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": class_id, "name": name}
                for name, class_id in sorted(classes.items(), key=lambda item: item[1])
            ],
        }
        for split in ("train", "val", "test")
    }
    report = {
        "requested_images": config.num_scenes,
        "generated_images": 0,
        "attempts": 0,
        "rejected_labels": 0,
        "rejected_separation": 0,
        "negative_images": 0,
        "objects_by_class": {name: 0 for name in classes},
        "images_by_split": {"train": 0, "val": 0, "test": 0},
        "objects_per_image": {},
    }
    renderer = create_offscreen_renderer(config.width, config.height)
    generated = attempts = rejected_separation = rejected_labels = 0
    last_pixel_counts: dict[str, int] = {}
    started_at = time.perf_counter()

    def report_progress(message: str) -> None:
        if not progress:
            return
        elapsed = time.perf_counter() - started_at
        rate = generated / elapsed if generated and elapsed > 0 else 0
        eta = (config.num_scenes - generated) / rate if rate else None
        eta_text = _format_duration(eta) if eta is not None else "calculando"
        progress(
            generated,
            config.num_scenes,
            f"{generated}/{config.num_scenes} imágenes "
            f"({generated / config.num_scenes:.1%}) | "
            f"tiempo transcurrido: {_format_duration(elapsed)} | "
            f"tiempo estimado restante: {eta_text} | "
            f"intentos: {attempts}; rechazos: "
            f"etiquetas={rejected_labels}, separación={rejected_separation} | "
            f"{message}",
        )

    report_progress("Renderizador inicializado.")
    try:
        while generated < config.num_scenes and attempts < config.max_attempts:
            if stop and stop():
                break
            attempts += 1
            scene = _scene(config)
            nodes, depths = {}, {}
            scene_names = (
                []
                if random.random() < config.negative_ratio
                else plan[generated % len(plan)]
            )
            for name in scene_names:
                mesh = pyrender.Mesh.from_trimesh(models[name], material=_material(config), smooth=False)
                transform, depth = _transform(
                    models[name],
                    config,
                    rotation_axes=config.rotation_axes_by_class.get(name, ["x", "y", "z"]),
                    rotation_step_deg=config.rotation_step_by_class_deg.get(name, 45.0),
                    initial_rotation_deg=config.initial_rotation_by_class_deg.get(name),
                )
                node = scene.add(mesh, pose=transform)
                nodes[node] = colors[name]
                depths[name] = depth
            rgb, _ = renderer.render(scene)
            segmentation, _ = renderer.render(scene, flags=pyrender.RenderFlags.SEG, seg_node_map=nodes)
            labels, masks, boxes, pixel_counts = _boxes(segmentation, classes, colors, config.width, config.height)
            if len(boxes) != len(scene_names):
                rejected_labels += 1
                last_pixel_counts = pixel_counts
                visible = ", ".join(f"{name}={count}px" for name, count in pixel_counts.items())
                report_progress(f"Escena rechazada por etiquetas ({visible}).")
                continue
            if not _separated(boxes, config.min_separation_px):
                rejected_separation += 1
                report_progress("Escena rechazada por separación.")
                continue
            background = _compose_background(rgb, masks, backgrounds, background_mode, config.width, config.height)
            random_split = random.random()
            if random_split < config.train_ratio:
                split = "train"
            elif random_split < config.train_ratio + config.val_ratio:
                split = "val"
            else:
                split = "test"
            stem = f"escena_{generated:05d}"
            Image.fromarray(background).save(config.output_dir / "images" / split / f"{stem}.jpg", quality=95)
            (config.output_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(labels), encoding="utf-8")
            if config.save_masks:
                mask_dtype = np.uint16 if len(classes) > 254 else np.uint8
                mask_image = np.zeros((config.height, config.width), dtype=mask_dtype)
                for name, mask in masks.items():
                    mask_image[mask] = classes[name] + 1
                Image.fromarray(mask_image).save(config.output_dir / "masks" / split / f"{stem}.png")
            objects = []
            for name, depth in depths.items():
                x_min, y_min, x_max, y_max = boxes[name]
                objects.append({
                    "class": name,
                    "class_id": classes[name],
                    "depth_m": round(depth, 4),
                    "visible_pixels": int(masks[name].sum()),
                    "bbox_px": [x_min, y_min, x_max, y_max],
                    "bbox_normalized": [
                        ((x_min + x_max) / 2) / config.width,
                        ((y_min + y_max) / 2) / config.height,
                        (x_max - x_min + 1) / config.width,
                        (y_max - y_min + 1) / config.height,
                    ],
                })
            (config.output_dir / "depth" / split / f"{stem}.json").write_text(
                json.dumps({"objects": objects}, indent=2),
                encoding="utf-8",
            )
            image_info, scene_annotations = _coco_scene_annotations(
                generated,
                f"images/{split}/{stem}.jpg",
                config.width,
                config.height,
                classes,
                masks,
                boxes,
            )
            coco[split]["images"].append(image_info)
            coco[split]["annotations"].extend(scene_annotations)
            report["images_by_split"][split] += 1
            report["objects_per_image"][str(len(scene_names))] = report["objects_per_image"].get(str(len(scene_names)), 0) + 1
            if not scene_names:
                report["negative_images"] += 1
            for name in scene_names:
                report["objects_by_class"][name] += 1
            generated += 1
            report_progress("Última escena guardada.")
    finally:
        _delete_renderer(renderer)
    elapsed = time.perf_counter() - started_at
    report.update({
        "generated_images": generated,
        "attempts": attempts,
        "rejected_labels": rejected_labels,
        "rejected_separation": rejected_separation,
        "elapsed_seconds": round(elapsed, 2),
    })
    write_coco_annotations(config.output_dir, coco)
    write_generation_report(config.output_dir, report)
    return {
        "generated": generated,
        "attempts": attempts,
        "rejected_labels": rejected_labels,
        "rejected_separation": rejected_separation,
        "last_pixel_counts": last_pixel_counts,
        "elapsed_seconds": round(elapsed, 2),
    }


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "calculando"
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {seconds_part:02d} s"
    return f"{seconds_part} s"
