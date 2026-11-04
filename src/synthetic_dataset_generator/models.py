from pathlib import Path
import trimesh


def load_models(model_paths: dict[str, Path], target_size_m: float = 1.0) -> dict[str, trimesh.Trimesh]:
    models: dict[str, trimesh.Trimesh] = {}
    for name, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"No existe el STL: {path}")
        mesh = trimesh.load(str(path), force="mesh").copy()
        mesh.apply_translation(-mesh.bounding_box.centroid)
        largest_extent = float(max(mesh.extents))
        if largest_extent <= 0:
            raise ValueError(f"El STL no tiene una geometría válida: {path}")
        mesh.apply_scale(target_size_m / largest_extent)
        mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
        models[name] = mesh
    return models


def model_scale_report(models: dict[str, trimesh.Trimesh]) -> list[str]:
    return [
        f"{name}: dimensiones normalizadas (m) "
        f"{mesh.extents[0]:.3f} x {mesh.extents[1]:.3f} x {mesh.extents[2]:.3f}"
        for name, mesh in models.items()
    ]
