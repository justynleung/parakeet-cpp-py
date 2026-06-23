"""High-level file transcription plus direct Parakeet C API bindings."""

from pathlib import Path
import subprocess

from . import _parakeet_cpp as _parakeet_cpp
from ._parakeet_cpp import *  # noqa: F401,F403

_MODEL_REPOSITORY = "ggml-org/parakeet-GGUF"
_MODEL_NAME = "parakeet-tdt-0.6b-v3-f16.bin"


def _default_model_path():
    """Resolve the default model through Hugging Face's CLI cache."""
    try:
        completed = subprocess.run(
            ["hf", "download", _MODEL_REPOSITORY, _MODEL_NAME],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Hugging Face CLI is unavailable; install huggingface_hub") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "unknown error").strip()
        raise RuntimeError("failed to download the default Parakeet model: {}".format(detail)) from error

    model_path = Path(completed.stdout.strip())
    if not model_path.is_file():
        raise RuntimeError("Hugging Face CLI did not return a model file path")
    return model_path


class Parakeet:
    """File-transcription facade backed by an in-process Parakeet context."""

    def __init__(self, model_path=None, *, use_gpu=True, gpu_device=0, n_threads=None):
        self.model_path = Path(model_path) if model_path is not None else _default_model_path()
        if not self.model_path.is_file():
            raise FileNotFoundError("Parakeet model is not readable: {}".format(self.model_path))
        params = _parakeet_cpp.ContextParams()
        params.use_gpu = use_gpu
        params.gpu_device = gpu_device
        self._context = _parakeet_cpp.init_from_file(str(self.model_path), params)
        self.n_threads = n_threads

    def transcribe(self, audio_path, print_segments=False):
        params = _parakeet_cpp.FullParams()
        if self.n_threads is not None:
            params.n_threads = self.n_threads
        return _parakeet_cpp.transcribe_file(
            self._context, str(audio_path), params, print_segments
        )

    def close(self):
        self._context.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


__all__ = [name for name in dir(_parakeet_cpp) if not name.startswith("_")] + ["Parakeet"]
