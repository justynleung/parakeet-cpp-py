import os
import stat
import textwrap

import pytest

import parakeet_cpp
from parakeet_cpp._parakeet_cpp import Parakeet as _NativeParakeet


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"model")
    return path


@pytest.fixture
def audio_path(tmp_path):
    path = tmp_path / "audio with spaces.wav"
    path.write_bytes(b"audio")
    return path


def make_cli(tmp_path, body):
    path = tmp_path / "fake parakeet cli.py"
    path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_transcribe_returns_text_only(model_path, audio_path, tmp_path):
    cli_path = make_cli(
        tmp_path,
        """
        import sys
        assert "--model" in sys.argv
        assert "--file" in sys.argv
        assert "--no-prints" in sys.argv
        assert "--print-segments" not in sys.argv
        print("hello world")
        """,
    )

    result = _NativeParakeet(str(model_path), str(cli_path)).transcribe(str(audio_path))

    assert result == {"text": "hello world"}


def test_transcribe_returns_all_segment_details(model_path, audio_path, tmp_path):
    cli_path = make_cli(
        tmp_path,
        """
        import sys
        assert "--print-segments" in sys.argv
        print("Hello world")
        sys.stderr.write('''
        Segments (1):
        Segment 0: [0 -> 1101] "Hello world"
        Tokens [2]:
          [ 0] id= 1976 frame=  3 dur_idx= 4 dur_val= 4 p=0.9996 plog=-15.6206 t0=  24 t1=  56 word_start=true "▁Hello"
          [ 1] id=  547 frame=  7 dur_idx= 2 dur_val= 2 p=1.0000 plog=-18.7922 t0=  56 t1=  88 word_start=false "world"
        ''')
        """,
    )

    result = _NativeParakeet(str(model_path), str(cli_path)).transcribe(str(audio_path), print_segments=True)

    assert result["text"] == "Hello world"
    assert result["segments"] == [
        {
            "index": 0,
            "t0": 0,
            "t1": 1101,
            "text": "Hello world",
            "tokens": [
                {
                    "index": 0,
                    "id": 1976,
                    "frame_index": 3,
                    "duration_idx": 4,
                    "duration_value": 4,
                    "p": 0.9996,
                    "plog": -15.6206,
                    "t0": 24,
                    "t1": 56,
                    "word_start": True,
                    "text": "▁Hello",
                },
                {
                    "index": 1,
                    "id": 547,
                    "frame_index": 7,
                    "duration_idx": 2,
                    "duration_value": 2,
                    "p": 1.0,
                    "plog": -18.7922,
                    "t0": 56,
                    "t1": 88,
                    "word_start": False,
                    "text": "world",
                },
            ],
        }
    ]


def test_transcribe_stream_forwards_default_window_options(model_path, audio_path, tmp_path):
    cli_path = make_cli(
        tmp_path,
        """
        import sys
        assert "--stream" in sys.argv
        assert sys.argv[sys.argv.index("--left-context-ms") + 1] == "10000"
        assert sys.argv[sys.argv.index("--chunk-ms") + 1] == "2000"
        assert sys.argv[sys.argv.index("--right-context-ms") + 1] == "2000"
        assert "--print-segments" not in sys.argv
        print("streamed transcript")
        """,
    )

    result = _NativeParakeet(str(model_path), str(cli_path)).transcribe_stream(str(audio_path))

    assert result == {"text": "streamed transcript"}


def test_transcribe_stream_forwards_custom_window_options_and_segments(model_path, audio_path, tmp_path):
    cli_path = make_cli(
        tmp_path,
        """
        import sys
        assert sys.argv[sys.argv.index("--left-context-ms") + 1] == "8000"
        assert sys.argv[sys.argv.index("--chunk-ms") + 1] == "1600"
        assert sys.argv[sys.argv.index("--right-context-ms") + 1] == "2400"
        assert "--print-segments" in sys.argv
        print("streamed transcript")
        sys.stderr.write('''
        Segments (1):
        Segment 0: [0 -> 20] "streamed transcript"
        Tokens [0]:
        ''')
        """,
    )

    result = _NativeParakeet(str(model_path), str(cli_path)).transcribe_stream(
        str(audio_path), 8000, 1600, 2400, print_segments=True
    )

    assert result == {
        "text": "streamed transcript",
        "segments": [{"index": 0, "t0": 0, "t1": 20, "text": "streamed transcript", "tokens": []}],
    }


def test_constructor_rejects_missing_or_non_executable_cli(model_path, tmp_path):
    with pytest.raises(ValueError, match="not executable"):
        _NativeParakeet(str(model_path), str(tmp_path / "missing-cli"))

    cli_path = tmp_path / "not-executable"
    cli_path.write_text("#!/bin/sh\n")
    with pytest.raises(ValueError, match="not executable"):
        _NativeParakeet(str(model_path), str(cli_path))


def test_transcribe_rejects_missing_audio(model_path, tmp_path):
    cli_path = make_cli(tmp_path, "print('unused')")

    with pytest.raises(ValueError, match="audio file is not readable"):
        _NativeParakeet(str(model_path), str(cli_path)).transcribe(str(tmp_path / "missing.wav"))


def test_transcribe_reports_cli_failure(model_path, audio_path, tmp_path):
    cli_path = make_cli(
        tmp_path,
        """
        import sys
        sys.stderr.write("model failed\\n")
        raise SystemExit(3)
        """,
    )

    with pytest.raises(RuntimeError, match="model failed"):
        _NativeParakeet(str(model_path), str(cli_path)).transcribe(str(audio_path))


def test_transcribe_stream_reports_cli_failure_for_invalid_timing(model_path, audio_path, tmp_path):
    cli_path = make_cli(
        tmp_path,
        """
        import sys
        assert sys.argv[sys.argv.index("--chunk-ms") + 1] == "100"
        sys.stderr.write("invalid streaming parameters\\n")
        raise SystemExit(1)
        """,
    )

    with pytest.raises(RuntimeError, match="invalid streaming parameters"):
        _NativeParakeet(str(model_path), str(cli_path)).transcribe_stream(str(audio_path), chunk_ms=100)


def test_transcribe_rejects_malformed_segment_output(model_path, audio_path, tmp_path):
    cli_path = make_cli(
        tmp_path,
        """
        import sys
        print("text")
        sys.stderr.write("Segments (1):\\ninvalid\\n")
        """,
    )

    with pytest.raises(RuntimeError, match="malformed segment"):
        _NativeParakeet(str(model_path), str(cli_path)).transcribe(str(audio_path), print_segments=True)


def test_public_constructor_uses_source_tree_defaults(audio_path, tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    model_path = (
        repo_root
        / "third_party"
        / "whisper.cpp"
        / "models"
        / "ggml-parakeet-tdt-0.6b-v3-q8_0.bin"
    )
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    cli_path = repo_root / "third_party" / "whisper.cpp" / "build" / "bin" / "parakeet-cli"
    cli_path.parent.mkdir(parents=True)
    cli_path.write_text("#!/usr/bin/env python3\nprint('default path transcript')\n")
    cli_path.chmod(cli_path.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(parakeet_cpp, "_repo_root", lambda: repo_root)

    result = parakeet_cpp.Parakeet().transcribe(audio_path)

    assert result == {"text": "default path transcript"}


def test_public_transcribe_stream_uses_source_tree_defaults(audio_path, tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    model_path = (
        repo_root
        / "third_party"
        / "whisper.cpp"
        / "models"
        / "ggml-parakeet-tdt-0.6b-v3-q8_0.bin"
    )
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    cli_path = repo_root / "third_party" / "whisper.cpp" / "build" / "bin" / "parakeet-cli"
    cli_path.parent.mkdir(parents=True)
    cli_path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "assert '--stream' in sys.argv\n"
        "print('streamed default path transcript')\n"
    )
    cli_path.chmod(cli_path.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(parakeet_cpp, "_repo_root", lambda: repo_root)

    result = parakeet_cpp.Parakeet().transcribe_stream(audio_path)

    assert result == {"text": "streamed default path transcript"}


def test_public_constructor_reports_missing_model(tmp_path, monkeypatch):
    monkeypatch.setattr(parakeet_cpp, "_repo_root", lambda: tmp_path)

    with pytest.raises(FileNotFoundError, match="Parakeet model is missing"):
        parakeet_cpp.Parakeet()


def test_public_constructor_reports_missing_cli(tmp_path, monkeypatch):
    model_path = (
        tmp_path
        / "third_party"
        / "whisper.cpp"
        / "models"
        / "ggml-parakeet-tdt-0.6b-v3-q8_0.bin"
    )
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    monkeypatch.setattr(parakeet_cpp, "_repo_root", lambda: tmp_path)

    with pytest.raises(FileNotFoundError, match="Parakeet CLI is missing"):
        parakeet_cpp.Parakeet()
