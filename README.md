# parakeet-cpp-py

`parakeet-cpp-py` exposes the pinned `whisper.cpp` Parakeet CLI through a C++11 pybind11 extension. 
## Install

```console
git clone https://github.com/justynleung/parakeet-cpp-py.git
cd parakeet-cpp-py
python -m pip install -e .
```

The scikit-build-core CMake workflow handles the remaining setup inside `third_party/whisper.cpp`: it configures and builds `parakeet-cli`, then downloads `ggml-parakeet-tdt-0.6b-v3-q8_0.bin` into `models/` when the model is absent.

## Usage

```python
from parakeet_cpp import Parakeet

pk = Parakeet()
result = pk.transcribe("samples/jfk.wav")
print(result["text"])

for segment in pk.transcribe("samples/jfk.wav", print_segments=True)["segments"]:
    print(segment["t0"], segment["t1"], segment["text"])

streamed = pk.transcribe_stream(
    "samples/jfk.wav", left_context_ms=8000, chunk_ms=1600, right_context_ms=2400
)
print(streamed["text"])
```

`Parakeet()` always loads:

- `third_party/whisper.cpp/models/ggml-parakeet-tdt-0.6b-v3-q8_0.bin`
- `third_party/whisper.cpp/build/bin/parakeet-cli`

`transcribe()` accepts the formats supported by the pinned CLI: WAV, MP3, FLAC, and OGG. With `print_segments=False` it returns `{"text": ...}`. With `print_segments=True`, it also returns segments and every token field emitted by `parakeet-cli`.

`transcribe_stream()` processes one complete audio file in overlapping windows. Its default left-context, emitted-chunk, and right-context durations are 10,000 ms, 2,000 ms, and 2,000 ms; all durations must be positive multiples of 80 ms. It returns the same result schema as `transcribe()`. Invalid timing values are reported by `parakeet-cli` as an error.

## Credits

- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) by [ggml-org](https://github.com/ggml-org) provides the Parakeet CLI.
- The Parakeet GGUF model is provided by [ggml-org/parakeet-GGUF](https://huggingface.co/ggml-org/parakeet-GGUF).

[MIT License](LICENSE)
