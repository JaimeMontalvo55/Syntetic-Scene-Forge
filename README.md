# Synthetic Scene Forge

Standalone local application for generating synthetic object-detection datasets
from STL models. The rendering core reuses the original mesh transformation,
RGB/segmentation rendering, bounding boxes, minimum object separation, depth,
and `train`/`val` split workflow.

The project does not contain ROS, perception nodes, or detectors.

## Installation

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux, system OpenGL libraries may be required for `pyrender` offscreen
rendering.

On Windows, the application creates the context with a configuration
compatible with drivers that do not export `WGL_ARB_pixel_format`. If the
driver does not provide OpenGL 3.3, update it before generating a dataset.

The project pins `pyglet<2` because it is compatible with `pyrender 0.1.x`.

## Running

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The application supports any number of STL files, class names, random colors,
images and videos, generation settings, and output in `images`, `labels`,
`depth`, `previews`, `reports`, and `data.yaml`. Object distribution per image
adapts to the number of selected models and uses distinct segmentation colors
for each class.

It also exports 2D masks (`masks/`), COCO annotations per split
(`annotations_train.json`, `annotations_val.json`, and `annotations_test.json`),
and a statistical report in `reports/generation_report.json`. Images without
objects can be configured with `negative_ratio`.

Generation runs in the background and provides a stop button. When canceled,
files created during that run are deleted without removing files that already
existed in the output folder. Combined mode alternates between random colors
and image or video scenes.

Depth position is configured in meters: mean distance is the center of the
triangular distribution, while minimum and maximum distances define its
bounds. STL files are automatically normalized so their largest dimension is
1 meter; file scale therefore does not change distance interpretation.

The interface also supports object material brightness variation with minimum
and maximum values from 0 to 1. A random value is selected per object and
scene so objects do not all stand out equally against the background. Emission
is disabled; objects do not act as light sources.

If all scenes are rejected by labels, the interface reports visible pixels per
class. A `0px` value means the STL is not reaching the camera or is not being
rasterized; values below `20px` indicate that the object is too small for the
visibility threshold.

STL files do not contain units. Models are normalized to a camera-compatible
scale. Validation only stops models that are clearly out of scale; large but
still visible models are allowed and checked through segmentation masks.

Each STL receives an independent random orientation. The interface displays
the model's local axes using X (left/right), Y (down/up), and Z (depth). Users
can select one or more axes and define the angular step, as well as a fixed
initial orientation in degrees for each axis. This correction is applied before
random rotation and allows models exported with a different orientation to be
placed correctly. Random angles are selected from 0 to 360 degrees using the
configured step; unselected axes do not rotate. The exported configuration
preserves both angle sets for reproducibility.

The interface includes an STL viewer with overlaid local axes: X red, Y green,
and Z blue.

The original workflow does not include video frame extraction or additional
scene-generation procedures. Therefore, videos and images are provided as
scene sources through independent adapters.

## Structure

```text
app.py
src/synthetic_dataset_generator/
  config.py       Serializable parameters and configuration
  models.py       STL loading and normalization
  sources.py      Backgrounds, images, and frame extraction
  generation.py   Rendering, composition, and labeling
  export.py       Label and configuration export
example_config.json
```

Generated datasets and uploaded files are excluded by `.gitignore`.
