from dataclasses import asdict, dataclass, field
from pathlib import Path
import json


@dataclass
class GenerationConfig:
    output_dir: Path
    width: int = 640
    height: int = 480
    num_scenes: int = 1500
    train_ratio: float = 0.80
    val_ratio: float = 0.20
    test_ratio: float = 0.0
    negative_ratio: float = 0.0
    save_masks: bool = True
    min_separation_px: int = 30
    max_attempts_multiplier: int = 30
    object_distance_mean_m: float = 3.5
    object_distance_min_m: float = 3.0
    object_distance_max_m: float = 4.0
    object_brightness_min: float = 0.45
    object_brightness_max: float = 0.85
    rotation_axes_by_class: dict[str, list[str]] = field(default_factory=dict)
    rotation_step_by_class_deg: dict[str, float] = field(default_factory=dict)
    initial_rotation_by_class_deg: dict[str, dict[str, float]] = field(default_factory=dict)
    x_min: float = -0.9
    x_max: float = 0.9
    y_min: float = -0.65
    y_max: float = 0.65
    seed: int = 42

    @property
    def max_attempts(self) -> int:
        return self.num_scenes * self.max_attempts_multiplier

    def save(self, path: Path) -> None:
        values = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}
        path.write_text(json.dumps(values, indent=2), encoding="utf-8")
