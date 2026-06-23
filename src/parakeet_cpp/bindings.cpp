#include <pybind11/pybind11.h>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace py = pybind11;

namespace {

struct ProcessResult {
    int status;
    std::string stdout_text;
    std::string stderr_text;
};

struct Token {
    int index;
    int id;
    int frame_index;
    int duration_idx;
    int duration_value;
    double p;
    double plog;
    long long t0;
    long long t1;
    bool word_start;
    std::string text;
};

struct Segment {
    int index;
    long long t0;
    long long t1;
    std::string text;
    std::vector<Token> tokens;
};

void require_readable_file(const std::string & path, const char * description) {
    if (path.empty() || access(path.c_str(), R_OK) != 0) {
        throw std::invalid_argument(std::string(description) + " is not readable: " + path);
    }
}

void require_executable_file(const std::string & path) {
    if (path.empty() || access(path.c_str(), X_OK) != 0) {
        throw std::invalid_argument("parakeet CLI is not executable: " + path);
    }
}

void set_nonblocking(int fd) {
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) {
        throw std::runtime_error(std::string("failed to configure process pipe: ") + std::strerror(errno));
    }
}

void drain_fd(int & fd, std::string & output) {
    char buffer[4096];
    while (true) {
        const ssize_t count = read(fd, buffer, sizeof(buffer));
        if (count > 0) {
            output.append(buffer, static_cast<std::size_t>(count));
            continue;
        }
        if (count == 0) {
            close(fd);
            fd = -1;
            return;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return;
        }
        const std::string message = std::string("failed to read process output: ") + std::strerror(errno);
        close(fd);
        fd = -1;
        throw std::runtime_error(message);
    }
}

ProcessResult run_process(const std::vector<std::string> & arguments) {
    int stdout_pipe[2];
    int stderr_pipe[2];
    if (pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0) {
        throw std::runtime_error(std::string("failed to create process pipes: ") + std::strerror(errno));
    }

    const pid_t pid = fork();
    if (pid < 0) {
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        close(stderr_pipe[0]);
        close(stderr_pipe[1]);
        throw std::runtime_error(std::string("failed to start parakeet CLI: ") + std::strerror(errno));
    }

    if (pid == 0) {
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        close(stderr_pipe[0]);
        close(stderr_pipe[1]);

        std::vector<char *> argv;
        argv.reserve(arguments.size() + 1);
        for (std::vector<std::string>::const_iterator it = arguments.begin(); it != arguments.end(); ++it) {
            argv.push_back(const_cast<char *>(it->c_str()));
        }
        argv.push_back(NULL);
        execv(argv[0], argv.data());
        dprintf(STDERR_FILENO, "failed to execute parakeet CLI: %s%c", std::strerror(errno), 10);
        _exit(127);
    }

    close(stdout_pipe[1]);
    close(stderr_pipe[1]);
    int stdout_fd = stdout_pipe[0];
    int stderr_fd = stderr_pipe[0];
    set_nonblocking(stdout_fd);
    set_nonblocking(stderr_fd);

    ProcessResult result;
    result.status = -1;
    while (stdout_fd != -1 || stderr_fd != -1) {
        struct pollfd fds[2];
        nfds_t count = 0;
        if (stdout_fd != -1) {
            fds[count].fd = stdout_fd;
            fds[count].events = POLLIN | POLLHUP;
            fds[count].revents = 0;
            ++count;
        }
        if (stderr_fd != -1) {
            fds[count].fd = stderr_fd;
            fds[count].events = POLLIN | POLLHUP;
            fds[count].revents = 0;
            ++count;
        }
        const int poll_result = poll(fds, count, -1);
        if (poll_result < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(std::string("failed while reading parakeet CLI output: ") + std::strerror(errno));
        }
        if (stdout_fd != -1) {
            drain_fd(stdout_fd, result.stdout_text);
        }
        if (stderr_fd != -1) {
            drain_fd(stderr_fd, result.stderr_text);
        }
    }

    int wait_status = 0;
    while (waitpid(pid, &wait_status, 0) == -1) {
        if (errno != EINTR) {
            throw std::runtime_error(std::string("failed to wait for parakeet CLI: ") + std::strerror(errno));
        }
    }
    result.status = wait_status;
    return result;
}

std::string trim_newlines(std::string value) {
    while (!value.empty() && (value[value.size() - 1] == 10 || value[value.size() - 1] == 13)) {
        value.erase(value.size() - 1);
    }
    return value;
}

std::vector<std::string> split_lines(const std::string & text) {
    std::vector<std::string> lines;
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line[line.size() - 1] == 13) {
            line.erase(line.size() - 1);
        }
        lines.push_back(line);
    }
    return lines;
}

long long parse_integer(const std::string & value, const char * field) {
    try {
        return std::stoll(value);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string("invalid ") + field + " in parakeet CLI output");
    }
}

double parse_float(const std::string & value, const char * field) {
    try {
        return std::stod(value);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string("invalid ") + field + " in parakeet CLI output");
    }
}

std::vector<Segment> parse_segments(const std::string & stderr_text) {
    static const std::regex segments_pattern("^Segments \\(([0-9]+)\\):$");
    static const std::regex segment_pattern("^Segment ([0-9]+): \\[(-?[0-9]+) -> (-?[0-9]+)\\] \\\"(.*)\\\"$");
    static const std::regex tokens_pattern("^Tokens \\[([0-9]+)\\]:$");
    static const std::regex token_pattern(
        "^  \\[ *([0-9]+)\\] id= *(-?[0-9]+) frame= *(-?[0-9]+) dur_idx= *(-?[0-9]+) dur_val= *(-?[0-9]+) "
        "p=([-+0-9.eE]+) plog=([-+0-9.eE]+) t0= *(-?[0-9]+) t1= *(-?[0-9]+) "
        "word_start=(true|false) \\\"(.*)\\\"$");

    const std::vector<std::string> lines = split_lines(stderr_text);
    std::smatch match;
    std::size_t position = 0;
    for (; position < lines.size(); ++position) {
        if (std::regex_match(lines[position], match, segments_pattern)) {
            break;
        }
    }
    if (position == lines.size()) {
        throw std::runtime_error("parakeet CLI did not emit segment details");
    }

    const std::size_t expected_segments = static_cast<std::size_t>(parse_integer(match[1].str(), "segment count"));
    ++position;
    std::vector<Segment> segments;
    segments.reserve(expected_segments);
    for (std::size_t segment_number = 0; segment_number < expected_segments; ++segment_number) {
        while (position < lines.size() && lines[position].empty()) {
            ++position;
        }
        if (position == lines.size() || !std::regex_match(lines[position], match, segment_pattern)) {
            throw std::runtime_error("malformed segment in parakeet CLI output");
        }
        Segment segment;
        segment.index = static_cast<int>(parse_integer(match[1].str(), "segment index"));
        segment.t0 = parse_integer(match[2].str(), "segment t0");
        segment.t1 = parse_integer(match[3].str(), "segment t1");
        segment.text = match[4].str();
        ++position;

        if (position == lines.size() || !std::regex_match(lines[position], match, tokens_pattern)) {
            throw std::runtime_error("missing token count in parakeet CLI output");
        }
        const std::size_t expected_tokens = static_cast<std::size_t>(parse_integer(match[1].str(), "token count"));
        ++position;
        segment.tokens.reserve(expected_tokens);
        for (std::size_t token_number = 0; token_number < expected_tokens; ++token_number) {
            if (position == lines.size() || !std::regex_match(lines[position], match, token_pattern)) {
                throw std::runtime_error("malformed token in parakeet CLI output");
            }
            Token token;
            token.index = static_cast<int>(parse_integer(match[1].str(), "token index"));
            token.id = static_cast<int>(parse_integer(match[2].str(), "token id"));
            token.frame_index = static_cast<int>(parse_integer(match[3].str(), "token frame index"));
            token.duration_idx = static_cast<int>(parse_integer(match[4].str(), "token duration index"));
            token.duration_value = static_cast<int>(parse_integer(match[5].str(), "token duration value"));
            token.p = parse_float(match[6].str(), "token probability");
            token.plog = parse_float(match[7].str(), "token log probability");
            token.t0 = parse_integer(match[8].str(), "token t0");
            token.t1 = parse_integer(match[9].str(), "token t1");
            token.word_start = match[10].str() == "true";
            token.text = match[11].str();
            segment.tokens.push_back(token);
            ++position;
        }
        segments.push_back(segment);
    }
    return segments;
}

py::dict as_dict(const std::string & text, const std::vector<Segment> & segments) {
    py::dict result;
    result["text"] = text;
    py::list segment_list;
    for (std::vector<Segment>::const_iterator segment = segments.begin(); segment != segments.end(); ++segment) {
        py::dict segment_dict;
        segment_dict["index"] = segment->index;
        segment_dict["t0"] = segment->t0;
        segment_dict["t1"] = segment->t1;
        segment_dict["text"] = segment->text;
        py::list token_list;
        for (std::vector<Token>::const_iterator token = segment->tokens.begin(); token != segment->tokens.end(); ++token) {
            py::dict token_dict;
            token_dict["index"] = token->index;
            token_dict["id"] = token->id;
            token_dict["frame_index"] = token->frame_index;
            token_dict["duration_idx"] = token->duration_idx;
            token_dict["duration_value"] = token->duration_value;
            token_dict["p"] = token->p;
            token_dict["plog"] = token->plog;
            token_dict["t0"] = token->t0;
            token_dict["t1"] = token->t1;
            token_dict["word_start"] = token->word_start;
            token_dict["text"] = token->text;
            token_list.append(token_dict);
        }
        segment_dict["tokens"] = token_list;
        segment_list.append(segment_dict);
    }
    result["segments"] = segment_list;
    return result;
}

class Parakeet {
public:
    Parakeet(const std::string & model_path, const std::string & cli_path)
        : model_path_(model_path), cli_path_(cli_path) {
        require_readable_file(model_path_, "model");
        require_executable_file(cli_path_);
    }

    py::dict transcribe(const std::string & audio_path, bool print_segments) const {
        return transcribe_with_options(audio_path, print_segments, false, 0, 0, 0);
    }

    py::dict transcribe_stream(
            const std::string & audio_path,
            int left_context_ms,
            int chunk_ms,
            int right_context_ms,
            bool print_segments) const {
        return transcribe_with_options(
                audio_path, print_segments, true, left_context_ms, chunk_ms, right_context_ms);
    }

private:
    py::dict transcribe_with_options(
            const std::string & audio_path,
            bool print_segments,
            bool stream,
            int left_context_ms,
            int chunk_ms,
            int right_context_ms) const {
        require_readable_file(audio_path, "audio file");
        std::vector<std::string> arguments;
        arguments.push_back(cli_path_);
        arguments.push_back("--model");
        arguments.push_back(model_path_);
        arguments.push_back("--file");
        arguments.push_back(audio_path);
        arguments.push_back("--no-prints");
        if (stream) {
            arguments.push_back("--stream");
            arguments.push_back("--left-context-ms");
            arguments.push_back(std::to_string(left_context_ms));
            arguments.push_back("--chunk-ms");
            arguments.push_back(std::to_string(chunk_ms));
            arguments.push_back("--right-context-ms");
            arguments.push_back(std::to_string(right_context_ms));
        }
        if (print_segments) {
            arguments.push_back("--print-segments");
        }

        ProcessResult process;
        {
            py::gil_scoped_release release;
            process = run_process(arguments);
        }
        if (!WIFEXITED(process.status) || WEXITSTATUS(process.status) != 0) {
            throw std::runtime_error("parakeet CLI failed: " + trim_newlines(process.stderr_text));
        }

        const std::string text = trim_newlines(process.stdout_text);
        if (!print_segments) {
            py::dict result;
            result["text"] = text;
            return result;
        }
        return as_dict(text, parse_segments(process.stderr_text));
    }

    std::string model_path_;
    std::string cli_path_;
};

} // namespace

PYBIND11_MODULE(_parakeet_cpp, module) {
    module.doc() = "C++11 bindings for the Parakeet CLI";
    py::class_<Parakeet>(module, "Parakeet")
        .def(py::init<const std::string &, const std::string &>(), py::arg("model_path"), py::arg("parakeet_cli_path"))
        .def("transcribe", &Parakeet::transcribe, py::arg("audio_path"), py::arg("print_segments") = false)
        .def(
            "transcribe_stream",
            &Parakeet::transcribe_stream,
            py::arg("audio_path"),
            py::arg("left_context_ms") = 10000,
            py::arg("chunk_ms") = 2000,
            py::arg("right_context_ms") = 2000,
            py::arg("print_segments") = false);
}
