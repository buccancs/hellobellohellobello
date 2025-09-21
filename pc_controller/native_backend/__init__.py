"""Python package wrapper for the native backend extension.

This package expects a compiled extension named `native_backend` (.pyd/.so)
located in the same directory. It exposes NativeShimmer and NativeWebcam
classes. If the extension is missing, importing from this package will raise
ImportError; the GUI uses Python fallbacks in that case.
"""
from __future__ import annotations

try:
    from .native_backend import NativeShimmer, NativeWebcam, __version__, shimmer_capi_enabled  # type: ignore[attr-defined]
    __all__ = ["NativeShimmer", "NativeWebcam", "__version__", "shimmer_capi_enabled"]
except Exception as exc:  # pragma: no cover - optional
    raise ImportError(
        "native_backend extension not found. Build it with CMake and place the compiled "
        "artifact (native_backend.pyd/.so) into this directory."
    ) from exc
