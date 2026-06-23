#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common-whisper.h"
#include "ggml-backend.h"
#include "parakeet.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

struct Context {
    parakeet_context * ptr = nullptr;
    explicit Context(parakeet_context * value = nullptr) : ptr(value) {}
    ~Context() { if (ptr) parakeet_free(ptr); }
    Context(const Context &) = delete;
};

struct State {
    parakeet_state * ptr = nullptr;
    explicit State(parakeet_state * value = nullptr) : ptr(value) {}
    ~State() { if (ptr) parakeet_free_state(ptr); }
    State(const State &) = delete;
};

struct ContextParams {
    parakeet_context_params value;
    ContextParams() : value(parakeet_context_default_params()) {}
};

struct FullParams {
    parakeet_full_params value;
    py::object new_segment_callback = py::none();
    py::object new_token_callback = py::none();
    py::object progress_callback = py::none();
    py::object encoder_begin_callback = py::none();
    py::object abort_callback = py::none();

    explicit FullParams(parakeet_sampling_strategy strategy = PARAKEET_SAMPLING_GREEDY)
        : value(parakeet_full_default_params(strategy)) { install_callbacks(); }

    FullParams(const FullParams & other)
        : value(other.value), new_segment_callback(other.new_segment_callback), new_token_callback(other.new_token_callback),
          progress_callback(other.progress_callback), encoder_begin_callback(other.encoder_begin_callback), abort_callback(other.abort_callback) { install_callbacks(); }
    FullParams(FullParams && other) noexcept
        : value(other.value), new_segment_callback(std::move(other.new_segment_callback)), new_token_callback(std::move(other.new_token_callback)),
          progress_callback(std::move(other.progress_callback)), encoder_begin_callback(std::move(other.encoder_begin_callback)), abort_callback(std::move(other.abort_callback)) { install_callbacks(); }
    FullParams & operator=(const FullParams & other) {
        value = other.value; new_segment_callback = other.new_segment_callback; new_token_callback = other.new_token_callback;
        progress_callback = other.progress_callback; encoder_begin_callback = other.encoder_begin_callback; abort_callback = other.abort_callback; install_callbacks(); return *this;
    }

    void install_callbacks() {
        value.new_segment_callback = new_segment;
        value.new_segment_callback_user_data = this;
        value.new_token_callback = new_token;
        value.new_token_callback_user_data = this;
        value.progress_callback = progress;
        value.progress_callback_user_data = this;
        value.encoder_begin_callback = encoder_begin;
        value.encoder_begin_callback_user_data = this;
        value.abort_callback = abort;
        value.abort_callback_user_data = this;
    }

    static py::dict token_dict(const parakeet_token_data & token) {
        py::dict result;
        result["id"] = token.id;
        result["duration_idx"] = token.duration_idx;
        result["duration_value"] = token.duration_value;
        result["frame_index"] = token.frame_index;
        result["p"] = token.p;
        result["plog"] = token.plog;
        result["t0"] = token.t0;
        result["t1"] = token.t1;
        result["is_word_start"] = token.is_word_start;
        return result;
    }

    static void new_segment(parakeet_context *, parakeet_state *, int n_new, void * user_data) {
        FullParams * self = static_cast<FullParams *>(user_data);
        if (!self || self->new_segment_callback.is_none()) return;
        py::gil_scoped_acquire gil;
        try { self->new_segment_callback(n_new); } catch (py::error_already_set & e) { e.discard_as_unraisable("parakeet segment callback"); }
    }
    static void new_token(parakeet_context *, parakeet_state *, const parakeet_token_data * token, void * user_data) {
        FullParams * self = static_cast<FullParams *>(user_data);
        if (!self || self->new_token_callback.is_none()) return;
        py::gil_scoped_acquire gil;
        try { self->new_token_callback(token_dict(*token)); } catch (py::error_already_set & e) { e.discard_as_unraisable("parakeet token callback"); }
    }
    static void progress(parakeet_context *, parakeet_state *, int value, void * user_data) {
        FullParams * self = static_cast<FullParams *>(user_data);
        if (!self || self->progress_callback.is_none()) return;
        py::gil_scoped_acquire gil;
        try { self->progress_callback(value); } catch (py::error_already_set & e) { e.discard_as_unraisable("parakeet progress callback"); }
    }
    static bool encoder_begin(parakeet_context *, parakeet_state *, void * user_data) {
        FullParams * self = static_cast<FullParams *>(user_data);
        if (!self || self->encoder_begin_callback.is_none()) return true;
        py::gil_scoped_acquire gil;
        try { return self->encoder_begin_callback().cast<bool>(); } catch (py::error_already_set & e) { e.discard_as_unraisable("parakeet encoder callback"); return false; }
    }
    static bool abort(void * user_data) {
        FullParams * self = static_cast<FullParams *>(user_data);
        if (!self || self->abort_callback.is_none()) return false;
        py::gil_scoped_acquire gil;
        try { return self->abort_callback().cast<bool>(); } catch (py::error_already_set & e) { e.discard_as_unraisable("parakeet abort callback"); return true; }
    }
};

void require(Context & context) { if (!context.ptr) throw std::runtime_error("Parakeet context is closed"); }
void require(State & state) { if (!state.ptr) throw std::runtime_error("Parakeet state is closed"); }

py::array_t<float, py::array::c_style> pcm_array(py::handle samples) {
    py::array array = py::array::ensure(samples);
    if (!array || !py::isinstance<py::array_t<float>>(array) || array.ndim() != 1 || !(array.flags() & py::array::c_style))
        throw py::type_error("samples must be a contiguous one-dimensional float32 NumPy array");
    return py::reinterpret_borrow<py::array_t<float, py::array::c_style>>(array);
}

int full(Context & ctx, FullParams & params, py::handle samples, State * state, bool chunk) {
    require(ctx);
    py::array_t<float, py::array::c_style> audio = pcm_array(samples);
    const int n = static_cast<int>(audio.size());
    int result;
    { py::gil_scoped_release release;
      result = chunk ? parakeet_chunk(ctx.ptr, state ? state->ptr : nullptr, params.value, audio.data(), n)
                     : state ? parakeet_full_with_state(ctx.ptr, state->ptr, params.value, audio.data(), n)
                             : parakeet_full(ctx.ptr, params.value, audio.data(), n); }
    return result;
}

py::dict token_result(Context & ctx, State * state, int segment, int token, int index) {
    const parakeet_token_data data = state ? parakeet_full_get_token_data_from_state(state->ptr, segment, token)
                                           : parakeet_full_get_token_data(ctx.ptr, segment, token);
    py::dict result = FullParams::token_dict(data);
    result["index"] = index;
    result["text"] = parakeet_token_to_str(ctx.ptr, data.id);
    result["word_start"] = data.is_word_start;
    return result;
}

py::dict result_dict(Context & ctx, State * state, bool segments) {
    const int count = state ? parakeet_full_n_segments_from_state(state->ptr) : parakeet_full_n_segments(ctx.ptr);
    std::string text;
    py::list values;
    for (int i = 0; i < count; ++i) {
        const char * segment_text = state ? parakeet_full_get_segment_text_from_state(state->ptr, i) : parakeet_full_get_segment_text(ctx.ptr, i);
        text += segment_text;
        if (!segments) continue;
        py::dict item;
        item["index"] = i;
        item["t0"] = state ? parakeet_full_get_segment_t0_from_state(state->ptr, i) : parakeet_full_get_segment_t0(ctx.ptr, i);
        item["t1"] = state ? parakeet_full_get_segment_t1_from_state(state->ptr, i) : parakeet_full_get_segment_t1(ctx.ptr, i);
        item["text"] = segment_text;
        py::list tokens;
        const int n_tokens = state ? parakeet_full_n_tokens_from_state(state->ptr, i) : parakeet_full_n_tokens(ctx.ptr, i);
        for (int j = 0; j < n_tokens; ++j) tokens.append(token_result(ctx, state, i, j, j));
        item["tokens"] = tokens;
        values.append(item);
    }
    py::dict result;
    result["text"] = text;
    if (segments) result["segments"] = values;
    return result;
}

struct PythonLoader {
    py::object source;
    parakeet_model_loader loader;
    explicit PythonLoader(py::object object) : source(object) {
        loader.context = this; loader.read = read; loader.eof = eof; loader.close = close;
    }
    static size_t read(void * context, void * output, size_t size) {
        PythonLoader * self = static_cast<PythonLoader *>(context); py::gil_scoped_acquire gil;
        try { py::bytes data = self->source.attr("read")(size); std::string value = data; std::memcpy(output, value.data(), value.size()); return value.size(); } catch (py::error_already_set & e) { e.discard_as_unraisable("parakeet model loader"); return 0; }
    }
    static bool eof(void * context) {
        PythonLoader * self = static_cast<PythonLoader *>(context); py::gil_scoped_acquire gil;
        try { return self->source.attr("eof")().cast<bool>(); } catch (...) { return true; }
    }
    static void close(void * context) {
        PythonLoader * self = static_cast<PythonLoader *>(context); py::gil_scoped_acquire gil;
        try { if (py::hasattr(self->source, "close")) self->source.attr("close")(); } catch (py::error_already_set & e) { e.discard_as_unraisable("parakeet model loader"); }
    }
};

py::object global_log_callback = py::none();
void log_callback(ggml_log_level level, const char * text, void *) {
    if (global_log_callback.is_none()) return;
    py::gil_scoped_acquire gil;
    try { global_log_callback(static_cast<int>(level), text); } catch (py::error_already_set & e) { e.discard_as_unraisable("parakeet log callback"); }
}

} // namespace

PYBIND11_MODULE(_parakeet_cpp, m) {
    m.doc() = "Direct bindings for the whisper.cpp Parakeet C API";
    ggml_backend_load_all();
    m.attr("SAMPLE_RATE") = PARAKEET_SAMPLE_RATE;
    m.attr("HOP_LENGTH") = PARAKEET_HOP_LENGTH;
    py::enum_<parakeet_sampling_strategy>(m, "SamplingStrategy").value("GREEDY", PARAKEET_SAMPLING_GREEDY);

    py::class_<Context>(m, "Context").def("close", [](Context & c) { if (c.ptr) { parakeet_free(c.ptr); c.ptr = nullptr; } });
    py::class_<State>(m, "State").def("close", [](State & s) { if (s.ptr) { parakeet_free_state(s.ptr); s.ptr = nullptr; } });
    py::class_<ContextParams>(m, "ContextParams")
        .def(py::init<>())
        .def_property("use_gpu", [](const ContextParams & p) { return p.value.use_gpu; }, [](ContextParams & p, bool value) { p.value.use_gpu = value; })
        .def_property("gpu_device", [](const ContextParams & p) { return p.value.gpu_device; }, [](ContextParams & p, int value) { p.value.gpu_device = value; });
    py::class_<FullParams>(m, "FullParams")
        .def(py::init<parakeet_sampling_strategy>(), py::arg("strategy") = PARAKEET_SAMPLING_GREEDY)
        .def_property("n_threads", [](const FullParams & p) { return p.value.n_threads; }, [](FullParams & p, int value) { p.value.n_threads = value; })
        .def_property("offset_ms", [](const FullParams & p) { return p.value.offset_ms; }, [](FullParams & p, int value) { p.value.offset_ms = value; })
        .def_property("duration_ms", [](const FullParams & p) { return p.value.duration_ms; }, [](FullParams & p, int value) { p.value.duration_ms = value; })
        .def_property("no_context", [](const FullParams & p) { return p.value.no_context; }, [](FullParams & p, bool value) { p.value.no_context = value; })
        .def_property("audio_ctx", [](const FullParams & p) { return p.value.audio_ctx; }, [](FullParams & p, int value) { p.value.audio_ctx = value; })
        .def_property("new_segment_callback", [](FullParams & p) { return p.new_segment_callback; }, [](FullParams & p, py::object f) { p.new_segment_callback = f; })
        .def_property("new_token_callback", [](FullParams & p) { return p.new_token_callback; }, [](FullParams & p, py::object f) { p.new_token_callback = f; })
        .def_property("progress_callback", [](FullParams & p) { return p.progress_callback; }, [](FullParams & p, py::object f) { p.progress_callback = f; })
        .def_property("encoder_begin_callback", [](FullParams & p) { return p.encoder_begin_callback; }, [](FullParams & p, py::object f) { p.encoder_begin_callback = f; })
        .def_property("abort_callback", [](FullParams & p) { return p.abort_callback; }, [](FullParams & p, py::object f) { p.abort_callback = f; });
    py::class_<parakeet_token_data>(m, "TokenData")
        .def_readonly("id", &parakeet_token_data::id).def_readonly("duration_idx", &parakeet_token_data::duration_idx)
        .def_readonly("duration_value", &parakeet_token_data::duration_value).def_readonly("frame_index", &parakeet_token_data::frame_index)
        .def_readonly("p", &parakeet_token_data::p).def_readonly("plog", &parakeet_token_data::plog)
        .def_readonly("t0", &parakeet_token_data::t0).def_readonly("t1", &parakeet_token_data::t1).def_readonly("is_word_start", &parakeet_token_data::is_word_start);
    py::class_<parakeet_timings>(m, "Timings").def_readonly("sample_ms", &parakeet_timings::sample_ms).def_readonly("encode_ms", &parakeet_timings::encode_ms).def_readonly("decode_ms", &parakeet_timings::decode_ms);

    m.def("version", [] { return parakeet_version(); });
    m.def("context_default_params", [] { return ContextParams(); });
    m.def("context_default_params_by_ref", [] { return ContextParams(); });
    // Python owns these value wrappers; the C API's pointer-free functions are
    // therefore intentionally no-ops here.
    m.def("free_context_params", [](ContextParams &) {});
    m.def("full_default_params", [](parakeet_sampling_strategy s) { return FullParams(s); }, py::arg("strategy") = PARAKEET_SAMPLING_GREEDY);
    m.def("full_default_params_by_ref", [](parakeet_sampling_strategy s) { return FullParams(s); }, py::arg("strategy") = PARAKEET_SAMPLING_GREEDY);
    m.def("free_params", [](FullParams &) {});
    m.def("init_from_file", [](const std::string & path, const ContextParams & p, bool no_state) { auto * c = no_state ? parakeet_init_from_file_with_params_no_state(path.c_str(), p.value) : parakeet_init_from_file_with_params(path.c_str(), p.value); if (!c) throw std::runtime_error("failed to load Parakeet model: " + path); return std::unique_ptr<Context>(new Context(c)); }, py::arg("path"), py::arg("params") = ContextParams(), py::arg("no_state") = false);
    m.def("init_from_buffer", [](py::buffer data, const ContextParams & p, bool no_state) { py::buffer_info b = data.request(); if (b.ndim != 1) throw py::value_error("model buffer must be one-dimensional"); auto * c = no_state ? parakeet_init_from_buffer_with_params_no_state(b.ptr, b.size * b.itemsize, p.value) : parakeet_init_from_buffer_with_params(b.ptr, b.size * b.itemsize, p.value); if (!c) throw std::runtime_error("failed to load Parakeet model buffer"); return std::unique_ptr<Context>(new Context(c)); }, py::arg("buffer"), py::arg("params") = ContextParams(), py::arg("no_state") = false);
    m.def("init_with_loader", [](py::object source, const ContextParams & p, bool no_state) { PythonLoader loader(source); auto * c = no_state ? parakeet_init_with_params_no_state(&loader.loader, p.value) : parakeet_init_with_params(&loader.loader, p.value); if (!c) throw std::runtime_error("failed to load Parakeet model from loader"); return std::unique_ptr<Context>(new Context(c)); }, py::arg("loader"), py::arg("params") = ContextParams(), py::arg("no_state") = false);
    m.def("init_state", [](Context & c) { require(c); auto * s = parakeet_init_state(c.ptr); if (!s) throw std::runtime_error("failed to initialize Parakeet state"); return std::unique_ptr<State>(new State(s)); });
    m.def("pcm_to_mel", [](Context & c, py::handle a, int threads, State * s) { require(c); auto x = pcm_array(a); py::gil_scoped_release r; return s ? parakeet_pcm_to_mel_with_state(c.ptr, s->ptr, x.data(), x.size(), threads) : parakeet_pcm_to_mel(c.ptr, x.data(), x.size(), threads); }, py::arg("context"), py::arg("samples"), py::arg("n_threads") = 1, py::arg("state") = nullptr);
    m.def("set_mel", [](Context & c, py::handle a, int n_mel, State * s) { require(c); auto x = pcm_array(a); py::gil_scoped_release r; return s ? parakeet_set_mel_with_state(c.ptr, s->ptr, x.data(), x.size(), n_mel) : parakeet_set_mel(c.ptr, x.data(), x.size(), n_mel); }, py::arg("context"), py::arg("data"), py::arg("n_mel") = 128, py::arg("state") = nullptr);
    m.def("encode", [](Context & c, int offset, int threads, State * s) { require(c); py::gil_scoped_release r; return s ? parakeet_encode_with_state(c.ptr, s->ptr, offset, threads) : parakeet_encode(c.ptr, offset, threads); }, py::arg("context"), py::arg("offset") = 0, py::arg("n_threads") = 1, py::arg("state") = nullptr);
    m.def("full", [](Context & c, FullParams & p, py::handle a, State * s) { return full(c, p, a, s, false); }, py::arg("context"), py::arg("params"), py::arg("samples"), py::arg("state") = nullptr);
    m.def("chunk", [](Context & c, State & s, FullParams & p, py::handle a) { require(s); return full(c, p, a, &s, true); });
    m.def("transcribe_file", [](Context & c, const std::string & path, FullParams & p, bool segments) { require(c); std::vector<float> pcm; std::vector<std::vector<float>> stereo; if (!read_audio_data(path, pcm, stereo, false) || pcm.empty()) throw std::runtime_error("failed to decode audio file: " + path); int status; { py::gil_scoped_release r; status = parakeet_full(c.ptr, p.value, pcm.data(), pcm.size()); } if (status) throw std::runtime_error("Parakeet inference failed"); return result_dict(c, nullptr, segments); }, py::arg("context"), py::arg("path"), py::arg("params"), py::arg("print_segments") = false);
    m.def("result", [](Context & c, bool segments, State * s) { require(c); if (s) require(*s); return result_dict(c, s, segments); }, py::arg("context"), py::arg("print_segments") = false, py::arg("state") = nullptr);

    m.def("tokenize", [](Context & c, const std::string & text) { require(c); int n = parakeet_token_count(c.ptr, text.c_str()); std::vector<parakeet_token> tokens(n); int got = parakeet_tokenize(c.ptr, text.c_str(), tokens.data(), n); if (got < 0) throw std::runtime_error("failed to tokenize text"); tokens.resize(got); return tokens; });
    m.def("token_count", [](Context & c, const std::string & text) { require(c); return parakeet_token_count(c.ptr, text.c_str()); });
    m.def("token_to_str", [](Context & c, parakeet_token t) { require(c); return parakeet_token_to_str(c.ptr, t); });
    m.def("token_to_text", [](const std::string & token, bool first) { char out[1024] = {}; int n = parakeet_token_to_text(token.c_str(), first, out, sizeof(out)); if (n < 0) throw std::runtime_error("failed to convert token to text"); return std::string(out); }, py::arg("token"), py::arg("is_first"));
    m.def("n_len", [](Context & c, State * s) { require(c); return s ? parakeet_n_len_from_state(s->ptr) : parakeet_n_len(c.ptr); }, py::arg("context"), py::arg("state") = nullptr);
    m.def("n_vocab", [](Context & c) { require(c); return parakeet_n_vocab(c.ptr); });
    m.def("n_audio_ctx", [](Context & c) { require(c); return parakeet_n_audio_ctx(c.ptr); });
    m.def("model_n_vocab", [](Context & c) { require(c); return parakeet_model_n_vocab(c.ptr); });
    m.def("model_n_audio_ctx", [](Context & c) { require(c); return parakeet_model_n_audio_ctx(c.ptr); });
    m.def("model_n_audio_state", [](Context & c) { require(c); return parakeet_model_n_audio_state(c.ptr); });
    m.def("model_n_audio_head", [](Context & c) { require(c); return parakeet_model_n_audio_head(c.ptr); });
    m.def("model_n_audio_layer", [](Context & c) { require(c); return parakeet_model_n_audio_layer(c.ptr); });
    m.def("model_n_mels", [](Context & c) { require(c); return parakeet_model_n_mels(c.ptr); });
    m.def("model_ftype", [](Context & c) { require(c); return parakeet_model_ftype(c.ptr); });
    m.def("token_blank", [](Context & c) { require(c); return parakeet_token_blank(c.ptr); }); m.def("token_unk", [](Context & c) { require(c); return parakeet_token_unk(c.ptr); }); m.def("token_bos", [](Context & c) { require(c); return parakeet_token_bos(c.ptr); });
    m.def("n_segments", [](Context & c, State * s) { require(c); return s ? parakeet_full_n_segments_from_state(s->ptr) : parakeet_full_n_segments(c.ptr); }, py::arg("context"), py::arg("state") = nullptr);
    m.def("segment_t0", [](Context & c, int i, State * s) { require(c); return s ? parakeet_full_get_segment_t0_from_state(s->ptr, i) : parakeet_full_get_segment_t0(c.ptr, i); }, py::arg("context"), py::arg("index"), py::arg("state") = nullptr);
    m.def("segment_t1", [](Context & c, int i, State * s) { require(c); return s ? parakeet_full_get_segment_t1_from_state(s->ptr, i) : parakeet_full_get_segment_t1(c.ptr, i); }, py::arg("context"), py::arg("index"), py::arg("state") = nullptr);
    m.def("segment_text", [](Context & c, int i, State * s) { require(c); return s ? parakeet_full_get_segment_text_from_state(s->ptr, i) : parakeet_full_get_segment_text(c.ptr, i); }, py::arg("context"), py::arg("index"), py::arg("state") = nullptr);
    m.def("n_tokens", [](Context & c, int i, State * s) { require(c); return s ? parakeet_full_n_tokens_from_state(s->ptr, i) : parakeet_full_n_tokens(c.ptr, i); }, py::arg("context"), py::arg("segment"), py::arg("state") = nullptr);
    m.def("token_data", [](Context & c, int i, int j, State * s) { require(c); return s ? parakeet_full_get_token_data_from_state(s->ptr, i, j) : parakeet_full_get_token_data(c.ptr, i, j); }, py::arg("context"), py::arg("segment"), py::arg("token"), py::arg("state") = nullptr);
    m.def("token_text", [](Context & c, int i, int j, State * s) { require(c); return s ? parakeet_full_get_token_text_from_state(c.ptr, s->ptr, i, j) : parakeet_full_get_token_text(c.ptr, i, j); }, py::arg("context"), py::arg("segment"), py::arg("token"), py::arg("state") = nullptr);
    m.def("token_id", [](Context & c, int i, int j, State * s) { require(c); return s ? parakeet_full_get_token_id_from_state(s->ptr, i, j) : parakeet_full_get_token_id(c.ptr, i, j); }, py::arg("context"), py::arg("segment"), py::arg("token"), py::arg("state") = nullptr);
    m.def("token_p", [](Context & c, int i, int j, State * s) { require(c); return s ? parakeet_full_get_token_p_from_state(s->ptr, i, j) : parakeet_full_get_token_p(c.ptr, i, j); }, py::arg("context"), py::arg("segment"), py::arg("token"), py::arg("state") = nullptr);
    m.def("get_logits", [](Context & c, State * s) { require(c); float * ptr = s ? parakeet_get_logits_from_state(s->ptr) : parakeet_get_logits(c.ptr); int rows = 0; int segments = s ? parakeet_full_n_segments_from_state(s->ptr) : parakeet_full_n_segments(c.ptr); for (int i = 0; i < segments; ++i) rows += s ? parakeet_full_n_tokens_from_state(s->ptr, i) : parakeet_full_n_tokens(c.ptr, i); py::array_t<float> result({rows, parakeet_n_vocab(c.ptr)}); if (ptr && rows) std::memcpy(result.mutable_data(), ptr, result.nbytes()); return result; }, py::arg("context"), py::arg("state") = nullptr);
    m.def("get_timings", [](Context & c) { require(c); return *parakeet_get_timings(c.ptr); }); m.def("print_timings", [](Context & c) { require(c); parakeet_print_timings(c.ptr); }); m.def("reset_timings", [](Context & c) { require(c); parakeet_reset_timings(c.ptr); });
    m.def("system_info", [] { return parakeet_print_system_info(); });
    m.def("log_set", [](py::object callback) { global_log_callback = callback; parakeet_log_set(callback.is_none() ? nullptr : log_callback, nullptr); }, py::arg("callback") = py::none());
}
