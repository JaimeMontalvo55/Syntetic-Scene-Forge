import os
import ctypes

if os.name != "nt" and "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import pyrender

from pyrender.constants import MIN_OPEN_GL_MAJOR, MIN_OPEN_GL_MINOR
from pyrender.platforms.base import Platform
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
            "Could not create the Windows OpenGL context. "
            "Update the graphics driver or run the application "
            "on a system with OpenGL 3.3."
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


class OSMesaOffscreenRenderer(pyrender.OffscreenRenderer):
    """Use OSMesa's legacy context API with PyOpenGL 3.1.0."""

    def _create(self):
        from OpenGL import GL as gl, arrays, platform
        from OpenGL.osmesa import (
            OSMesaCreateContextExt,
            OSMesaDestroyContext,
            OSMesaGetCurrentContext,
            OSMesaMakeCurrent,
            OSMESA_RGBA,
        )

        class OSMesaPlatform(Platform):
            def __init__(platform_self, viewport_width, viewport_height):
                super().__init__(viewport_width, viewport_height)
                platform_self._context = None
                platform_self._buffer = None

            def init_context(platform_self):
                platform_self._context = OSMesaCreateContextExt(OSMESA_RGBA, 24, 0, 0, None)
                if platform_self._context is None:
                    raise RuntimeError("OSMesa could not create a rendering context.")
                platform_self._buffer = arrays.GLubyteArray.zeros(
                    (platform_self.viewport_height, platform_self.viewport_width, 4)
                )

            def make_current(platform_self):
                if not OSMesaMakeCurrent(
                    platform_self._context,
                    platform_self._buffer,
                    gl.GL_UNSIGNED_BYTE,
                    platform_self.viewport_width,
                    platform_self.viewport_height,
                ):
                    raise RuntimeError("OSMesa could not activate the rendering context.")
                context = OSMesaGetCurrentContext()
                context_id = ctypes.cast(context, ctypes.c_void_p).value
                if not context_id:
                    raise RuntimeError("OSMesa did not expose an active rendering context.")
                platform.GetCurrentContext = lambda: context_id
                platform.CurrentContextIsValid = lambda: bool(context_id)
                platform.PLATFORM.GetCurrentContext = lambda: context_id
                platform.PLATFORM.CurrentContextIsValid = lambda: bool(context_id)
                platform.PLATFORM.checkExtension = lambda extension: True

            def make_uncurrent(platform_self):
                return None

            def delete_context(platform_self):
                if platform_self._context is not None:
                    OSMesaDestroyContext(platform_self._context)
                platform_self._context = None
                platform_self._buffer = None

            def supports_framebuffers(platform_self):
                return False

        self._platform = OSMesaPlatform(self.viewport_width, self.viewport_height)
        try:
            self._platform.init_context()
            self._platform.make_current()
            self._renderer = Renderer(
                self.viewport_width,
                self.viewport_height,
                point_size=self.point_size,
            )
        except Exception:
            self._platform.delete_context()
            self._platform = None
            raise


def create_offscreen_renderer(width: int, height: int) -> pyrender.OffscreenRenderer:
    last_error = None
    renderer_type = (
        OSMesaOffscreenRenderer
        if os.environ.get("PYOPENGL_PLATFORM") == "osmesa"
        else pyrender.OffscreenRenderer
    )
    try:
        return renderer_type(width, height)
    except Exception as error:
        last_error = error
    for _ in range(2):
        try:
            return WindowsOffscreenRenderer(width, height)
        except Exception as error:
            last_error = error
    raise RuntimeError(
        "Could not initialize offscreen rendering. "
        "Check that OpenGL/EGL is available and that no other session "
        f"is blocking the rendering context. Technical detail: {last_error}"
    ) from last_error
