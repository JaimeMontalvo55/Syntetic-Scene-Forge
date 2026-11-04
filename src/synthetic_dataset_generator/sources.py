from pathlib import Path
import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_scenario_images(paths: list[Path], width: int, height: int) -> list[np.ndarray]:
    result = []
    for path in paths:
        result.append(np.asarray(Image.open(path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)))
    return result


def extract_video_frames(path: Path, output_dir: Path, frame_step: int = 1) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {path}")
    paths: list[Path] = []
    index = 0
    saved = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % frame_step == 0:
                target = output_dir / f"{path.stem}_{saved:06d}.jpg"
                if not cv2.imwrite(str(target), frame):
                    raise OSError(f"No se pudo guardar el fotograma: {target}")
                paths.append(target)
                saved += 1
            index += 1
    finally:
        capture.release()
    return paths
