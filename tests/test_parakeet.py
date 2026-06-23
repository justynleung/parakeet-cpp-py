import subprocess
import wave

import numpy as np
import pytest

import parakeet_cpp
from parakeet_cpp import _parakeet_cpp as native
from parakeet_cpp import cli


MODEL_PATH = "third_party/whisper.cpp/models/for-tests-ggml-parakeet-tdt.bin"


@pytest.fixture
def context():
    params = native.ContextParams()
    params.use_gpu = False
    value = native.init_from_file(MODEL_PATH, params)
    yield value
    value.close()


@pytest.fixture
def pcm():
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def wav_path(tmp_path, pcm):
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(native.SAMPLE_RATE)
        output.writeframes((pcm * 32767).astype("<i2").tobytes())
    return path


def test_direct_api_initialization_and_metadata(context):
    assert native.version()
    assert native.n_vocab(context) > 0
    assert native.model_n_mels(context) > 0
    assert native.token_count(context, "test") >= 0


def test_pcm_mel_encode_full_chunk_and_results(context, pcm):
    params = native.FullParams()
    params.n_threads = 1
    assert native.pcm_to_mel(context, pcm, 1) == 0
    assert native.encode(context, 0, 1) == 0
    assert native.full(context, params, pcm) == 0
    result = native.result(context, print_segments=True)
    assert set(result) == {"text", "segments"}
    assert isinstance(native.get_logits(context), np.ndarray)

    state = native.init_state(context)
    assert native.pcm_to_mel(context, pcm, 1, state) == 0
    assert native.full(context, params, pcm, state) == 0
    assert native.chunk(context, state, params, pcm) == 0
    assert native.n_segments(context, state) >= 0
    state.close()


def test_callbacks_are_owned_by_the_parameter_instance(context, pcm):
    progress = []
    params = native.FullParams()
    params.n_threads = 1
    params.progress_callback = progress.append
    assert native.full(context, params, pcm) == 0
    assert progress


def test_native_file_transcription_returns_compatible_shape(context, wav_path):
    params = native.FullParams()
    params.n_threads = 1
    result = native.transcribe_file(context, str(wav_path), params, print_segments=True)
    assert set(result) == {"text", "segments"}
    assert all({"index", "t0", "t1", "text", "tokens"} <= set(item) for item in result["segments"])


def test_default_model_download_and_explicit_path_bypass(monkeypatch, tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")

    class Completed:
        stdout = str(model) + "\n"

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Completed())
    assert parakeet_cpp._default_model_path() == model

    called = False
    monkeypatch.setattr(parakeet_cpp, "_default_model_path", lambda: pytest.fail("downloaded"))
    monkeypatch.setattr(native, "init_from_file", lambda *args: object())
    instance = parakeet_cpp.Parakeet(model)
    assert instance.model_path == model
    assert not called


def test_default_model_download_failure_is_clear(monkeypatch):
    error = subprocess.CalledProcessError(1, ["hf"], stderr="network unavailable")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(error))
    with pytest.raises(RuntimeError, match="failed to download"):
        parakeet_cpp._default_model_path()


def test_cli_help_and_supported_flags(monkeypatch, capsys):
    with pytest.raises(SystemExit, match="0"):
        cli.main(["--help"])
    assert "--print-segments" in capsys.readouterr().out
    assert "stream" not in cli._parser().format_help()

    calls = []

    class FakeParakeet:
        def __init__(self, model, **kwargs):
            calls.append((model, kwargs))

        def transcribe(self, path, print_segments=False):
            calls.append((path, print_segments))
            return {"text": "ok", "segments": []}

    monkeypatch.setattr(cli, "Parakeet", FakeParakeet)
    assert cli.main(["first.wav", "-f", "second.wav", "-t", "2", "-m", "model.bin", "-ng", "-dev", "1", "-ps"]) == 0
    assert calls[0] == ("model.bin", {"use_gpu": False, "gpu_device": 1, "n_threads": 2})
    assert calls[1:] == [("first.wav", True), ("second.wav", True)]
