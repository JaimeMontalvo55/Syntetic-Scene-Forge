from pathlib import Path
import json


def prepare_output(output_dir: Path) -> None:
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "depth" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (output_dir / "masks" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml(output_dir: Path, classes: dict[str, int]) -> None:
    names = [name for name, _ in sorted(classes.items(), key=lambda item: item[1])]
    content = f"path: {output_dir.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
    if (output_dir / "images" / "test").exists():
        content += "test: images/test\n"
    content += "names:\n"
    content += "".join(f"  {index}: {name}\n" for index, name in enumerate(names))
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def write_generation_config(output_dir: Path, config, classes: dict[str, int]) -> None:
    data = {"config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(config).items()}, "classes": classes}
    (output_dir / "generation_config.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_coco_annotations(output_dir: Path, annotations_by_split: dict[str, dict]) -> None:
    for split, data in annotations_by_split.items():
        (output_dir / f"annotations_{split}.json").write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )


def write_generation_report(output_dir: Path, report: dict) -> None:
    (output_dir / "reports" / "generation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
