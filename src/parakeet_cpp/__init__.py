from pathlib import Path

from ._parakeet_cpp import Parakeet as _NativeParakeet

_MODEL_NAME = "ggml-parakeet-tdt-0.6b-v3-q8_0.bin"


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _default_paths():
    whisper_root = _repo_root() / "third_party" / "whisper.cpp"
    return (
        whisper_root / "models" / _MODEL_NAME,
        whisper_root / "build" / "bin" / "parakeet-cli",
    )


class Parakeet:
    def __init__(self):
        model_path, cli_path = _default_paths()
        if not model_path.is_file():
            raise FileNotFoundError(
                "Parakeet model is missing: {}. Download {} into third_party/whisper.cpp/models/.".format(
                    model_path, _MODEL_NAME
                )
            )
        if not cli_path.is_file() or not cli_path.stat().st_mode & 0o111:
            raise FileNotFoundError(
                "Parakeet CLI is missing or not executable: {}. Build it with "
                "cmake -S third_party/whisper.cpp -B third_party/whisper.cpp/build "
                "then cmake --build third_party/whisper.cpp/build --target parakeet-cli.".format(cli_path)
            )
        self._native = _NativeParakeet(str(model_path), str(cli_path))

    def transcribe(self, audio_path, print_segments=False):
        return self._native.transcribe(str(audio_path), print_segments)


__all__ = ["Parakeet"]
