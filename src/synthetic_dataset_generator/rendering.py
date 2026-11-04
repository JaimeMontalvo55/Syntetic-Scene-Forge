import os

import pyrender

from pyrender.constants import MIN_OPEN_GL_MAJOR, MIN_OPEN_GL_MINOR
from pyrender.renderer import Renderer
from pyrender.platforms.pyglet_platform import PygletPlatform


class WindowsPygletPlatform(PygletPlatform):
    """Create a context without requiring WGL_ARB_pixel_format."""

    def init_context(self):
        import pyglet

        pyglet.options["shadow_window"] = False
        pyglet.options["headless"] = False
        self._window = None
        configs = [
            pyglet.gl.Config(depth_size=24, double_buffer=True),
            pyglet.gl.Config(
                depth_size=24,
                double_buffer=True,
                major_version=MIN_OPEN_GL_MAJOR,
                minor_version=MIN_OPEN_GL_MINOR,
            ),
        ]
        last_error = None
        for config in configs:
            try:
                self._window = pyglet.window.Window(
                    config=config,
                    visible=False,
                    resizable=False,
                    width=1,
                    height=1,
                )
                return
            except Exception as error:
                last_error = error
                if self._window is not None:
                    try:
                        self._window.close()
                    finally:
                        self._window = None
        raise RuntimeError(
            "No se pudo crear el contexto OpenGL de Windows. "
            "Actualiza el controlador gráfico o ejecuta la aplicación "
            "en un sistema con OpenGL 3.3."
        ) from last_error


class WindowsOffscreenRenderer(pyrender.OffscreenRenderer):
    def _create(self):
        if os.environ.get("PYOPENGL_PLATFORM"):
            super()._create()
            return
        self._platform = WindowsPygletPlatform(
            self.viewport_width,
            self.viewport_height,
        )
        try:
            self._platform.init_context()
            self._platform.make_current()
            self._renderer = Renderer(
                self.viewport_width,
                self.viewport_height,
                point_size=self.point_size,
            )
        except Exception:
            if self._platform is not None:
                self._platform.delete_context()
            self._platform = None
            raise


def create_offscreen_renderer(width: int, height: int) -> pyrender.OffscreenRenderer:
    last_error = None
    try:
        return pyrender.OffscreenRenderer(width, height)
    except Exception as error:
        last_error = error
    for _ in range(2):
        try:
            return WindowsOffscreenRenderer(width, height)
        except Exception as error:
            last_error = error
    raise RuntimeError(
        "No se pudo inicializar el renderizado offscreen. "
        "Comprueba que la aceleración gráfica esté activa y que no haya "
        "otra sesión bloqueando el contexto OpenGL. "
        f"Detalle técnico: {last_error}"
    ) from last_error
