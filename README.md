# parakeet-cpp-py

`parakeet-cpp-py` is an in-process pybind11 binding for whisper.cpp's unpatched
Parakeet C API. It does not build or invoke whisper.cpp's `parakeet-cli`.

## Install

```console
python -m pip install .
```

The extension includes whisper.cpp's native audio reader for WAV, MP3, FLAC,
and OGG input. NumPy supplies PCM arrays. The default F16 model is downloaded
on first use through `hf download ggml-org/parakeet-GGUF parakeet-tdt-0.6b-v3-f16.bin`.

## Python

```python
from parakeet_cpp import Parakeet

with Parakeet() as model:
    result = model.transcribe("samples/jfk.wav", print_segments=True)
    print(result["text"])
    for segment in result["segments"]:
        print(segment["t0"], segment["t1"], segment["text"])
```

Pass `model_path` to `Parakeet(model_path)` to avoid automatic model
resolution. The direct C API is available from `parakeet_cpp._parakeet_cpp`
and re-exported from `parakeet_cpp`: use `ContextParams`, `init_from_file`,
`FullParams`, `full`, `chunk`, and the result accessors with contiguous,
one-dimensional `float32` NumPy PCM arrays.

## CLI

```console
parakeet-cli -m model.bin -t 4 -ng -f audio.wav
parakeet-cli audio.wav another.mp3 -ps
```

Supported flags are positional input files, `-f/--file`, `-t/--threads`,
`-m/--model`, `-ng/--no-gpu`, `-dev/--device`, and `-ps/--print-segments`.

## Credits

- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) provides Parakeet and audio decoding.
- [ggml-org/parakeet-GGUF](https://huggingface.co/ggml-org/parakeet-GGUF) provides the model.
