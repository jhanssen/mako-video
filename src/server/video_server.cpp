#include "server/video_server.hpp"

#include "http/access.hpp"
#include "log/log.hpp"
#include "video/image.hpp"
#include "video/media.hpp"
#include "video/mp4.hpp"

#include <glaze/glaze.hpp>
#include <httplib.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>

namespace mako::server {

namespace {

namespace fs = std::filesystem;

constexpr auto kProgressEvery = std::chrono::seconds(10);
constexpr size_t kMaxPrompt = 16384, kMaxReferences = 12, kMaxMedia = 512ull << 20; // a reference file's bytes
constexpr double kMinSeconds = 5.0, kMaxSeconds = 15.0, kMaxArea = 1344.0 * 768 * 2;

std::string json_quoted(std::string_view s)
{
    std::string out;
    (void)glz::write_json(s, out);
    return out;
}

std::string error_body(const std::string& message, const char* type)
{
    return std::format(R"({{"error":{{"message":{},"type":{}}}}})", json_quoted(message), json_quoted(type));
}

// Base64 (a "data:...;base64," prefix allowed); nullopt on anything else.
std::optional<std::vector<uint8_t>> unbase64(std::string_view s)
{
    if (s.starts_with("data:")) {
        const auto comma = s.find(',');
        if (comma == s.npos || s.substr(0, comma).find(";base64") == s.npos)
            return std::nullopt;
        s.remove_prefix(comma + 1);
    }
    std::vector<uint8_t> out;
    out.reserve(s.size() / 4 * 3);
    uint32_t acc = 0, bits = 0;
    for (const char ch : s) {
        int v = ch >= 'A' && ch <= 'Z' ? ch - 'A' : ch >= 'a' && ch <= 'z' ? ch - 'a' + 26
              : ch >= '0' && ch <= '9' ? ch - '0' + 52 : ch == '+' || ch == '-' ? 62 : ch == '/' || ch == '_' ? 63 : -1;
        if (ch == '=')
            break;
        if (v < 0) {
            if (ch == '\n' || ch == '\r' || ch == ' ')
                continue;
            return std::nullopt;
        }
        acc = acc << 6 | uint32_t(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(uint8_t(acc >> bits));
        }
    }
    return out;
}

int64_t now_seconds()
{
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// A media value: a string of base64 (or a data: URL), or an object with "image_url", "video_url", "audio_url" or
// "data" holding one. Throws std::invalid_argument.
std::vector<uint8_t> media_bytes(const glz::generic& v, const char* what)
{
    std::string text;
    if (v.is_string()) {
        text = v.get_string();
    } else if (v.is_object()) {
        for (const char* k : {"image_url", "video_url", "audio_url", "data", "url"})
            if (v.contains(k) && v[k].is_string()) {
                text = v[k].get_string();
                break;
            }
        if (text.empty() && v.contains("file_id"))
            throw std::invalid_argument(std::format("{}: files are not stored here; send the bytes in base64", what));
    }
    if (text.empty())
        throw std::invalid_argument(std::format("{} must be a file in base64 (or a data: URL), as a string or in an "
                                                "object's image_url / video_url / audio_url",
                                                what));
    if (text.starts_with("http://") || text.starts_with("https://"))
        throw std::invalid_argument(std::format("{}: web URLs are not fetched; send the bytes in base64", what));
    const auto bytes = unbase64(text);
    if (!bytes || bytes->empty())
        throw std::invalid_argument(std::format("{} is not base64", what));
    if (bytes->size() > kMaxMedia)
        throw std::invalid_argument(std::format("{} is past {} MB", what, kMaxMedia >> 20));
    return *bytes;
}

struct Cancelled : std::runtime_error {
    Cancelled() : std::runtime_error("cancelled") {}
};

} // namespace

struct VideoServer::Impl {
    struct Job {
        std::string id, prompt, model_name;
        int64_t created_at = 0, completed_at = 0;
        std::string status = "queued"; // queued, in_progress, completed, failed
        int progress = 0;
        std::string stage, error_code, error_message;
        uint32_t width = 0, height = 0, frames = 0, steps = 0;
        double seconds = 0;
        uint64_t seed = 0;
        size_t references = 0;
        bool first = false, last = false;
        std::vector<video::Lora> loras;
        VideoModel model;
        video::Engine::Request request;
        fs::path mp4, png;
        std::atomic<bool> cancel = false;
    };

    Impl(VideoResidency& r, LoadVideo l, VideoServerOptions o) : residency(r), load(std::move(l)), opts(std::move(o))
    {
        worker = std::jthread([this] { work(); });
    }
    ~Impl()
    {
        {
            std::lock_guard lock(m);
            stopping = true;
            if (running)
                running->cancel = true;
        }
        cv.notify_all();
    }
    VideoResidency& residency;
    LoadVideo load;
    VideoServerOptions opts;
    httplib::Server http;
    std::atomic<int> port = 0;
    std::atomic<uint32_t> next_id = 0;

    std::mutex m;
    std::condition_variable cv;
    std::map<std::string, std::shared_ptr<Job>> jobs; // by id
    std::deque<std::shared_ptr<Job>> queue;
    std::shared_ptr<Job> running;
    bool stopping = false;
    std::jthread worker;

    void create(const httplib::Request& req, httplib::Response& res);
    void list(const httplib::Request& req, httplib::Response& res);
    void retrieve(const httplib::Request& req, httplib::Response& res);
    void content(const httplib::Request& req, httplib::Response& res);
    void remove(const httplib::Request& req, httplib::Response& res);
    void loras(const httplib::Request& req, httplib::Response& res);
    void model_list(httplib::Response& res);
    void status(httplib::Response& res);
    VideoModel video_model(const std::string& name) const;
    std::string job_json(const Job& j); // under m
    std::shared_ptr<Job> find(const std::string& id);
    void work();
    void run_job(const std::shared_ptr<Job>& job);
    // A reference's bytes through a temporary file (the readers take paths).
    fs::path temp_file(std::span<const uint8_t> bytes, const char* ext);
};

VideoModel VideoServer::Impl::video_model(const std::string& name) const
{
    if (!opts.video_models.empty() && (!name.empty() || !fs::is_directory(opts.video_model))) {
        const std::string want = name.empty() ? opts.video_model : name;
        auto list = video_models(opts.video_models);
        if (want.empty() && !list.empty())
            return list.front();
        for (auto& m : list)
            if (m.name == want)
                return m;
        throw std::invalid_argument(std::format("no video model named \"{}\" (GET /v1/videos/models lists them)", want));
    }
    const std::string dir = opts.video_model;
    const std::string own = fs::path(dir).lexically_normal().filename().string();
    if (!name.empty() && name != own)
        throw std::invalid_argument(std::format("no video model named \"{}\" (this server has only {})", name, own));
    return load_video_model(dir, own);
}

std::string VideoServer::Impl::job_json(const Job& j)
{
    std::string loras;
    for (const auto& l : j.loras)
        loras += std::format(R"({}{{"name":{},"scale":{:g}}})", loras.empty() ? "" : ",",
                             json_quoted(fs::path(l.path).stem().string()), l.scale);
    std::string error = "null";
    if (!j.error_message.empty())
        error = std::format(R"({{"code":{},"message":{}}})", json_quoted(j.error_code), json_quoted(j.error_message));
    return std::format(
        R"({{"id":{},"object":"video","created_at":{},"completed_at":{},"expires_at":null,"status":"{}","progress":{},)"
        R"("model":{},"seconds":"{:g}","size":"{}x{}","prompt":{},"error":{},"remixed_from_video_id":null,)"
        R"("frames":{},"steps":{},"seed":"{}","loras":[{}],"references":{},"first_frame":{},"last_frame":{},"stage":{},"path":{}}})",
        json_quoted(j.id), j.created_at, j.completed_at ? std::to_string(j.completed_at) : "null", j.status, j.progress,
        json_quoted(j.model_name), j.seconds, j.width, j.height, json_quoted(j.prompt), error, j.frames, j.steps, j.seed,
        loras, j.references, j.first, j.last, json_quoted(j.stage),
        j.status == "completed" ? json_quoted(j.mp4.string()) : "null");
}

std::shared_ptr<VideoServer::Impl::Job> VideoServer::Impl::find(const std::string& id)
{
    std::lock_guard lock(m);
    const auto it = jobs.find(id);
    return it == jobs.end() ? nullptr : it->second;
}

fs::path VideoServer::Impl::temp_file(std::span<const uint8_t> bytes, const char* ext)
{
    static std::atomic<uint32_t> n = 0;
    const fs::path dir = fs::path(opts.output_dir) / ".tmp";
    fs::create_directories(dir);
    const fs::path p = dir / std::format("in-{}-{}.{}", ::getpid(), n.fetch_add(1) + 1, ext);
    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    if (!out)
        throw std::runtime_error("could not write a temporary file in " + dir.string());
    return p;
}

void VideoServer::Impl::create(const httplib::Request& req, httplib::Response& res)
{
    auto fail = [&](int status, const std::string& message, const char* type) {
        res.status = status;
        res.set_content(error_body(message, type), "application/json");
    };
    glz::generic body;
    if (glz::read_json(body, req.body) || !body.is_object())
        return fail(400, "the body is not a JSON object", "invalid_request_error");
    for (const auto& [k, v] : body.get_object())
        if (k != "prompt" && k != "model" && k != "seconds" && k != "size" && k != "steps" && k != "seed" && k != "loras" &&
            k != "input_reference" && k != "last_frame" && k != "references" && k != "user" && k != "metadata")
            return fail(400, std::format("unknown field {}", k), "invalid_request_error");
    if (!body.contains("prompt") || !body["prompt"].is_string() || body["prompt"].get_string().empty())
        return fail(400, "prompt is required", "invalid_request_error");
    auto job = std::make_shared<Job>();
    job->prompt = body["prompt"].get_string();
    if (job->prompt.size() > kMaxPrompt)
        return fail(400, std::format("prompt is longer than {} bytes", kMaxPrompt), "invalid_request_error");
    try {
        job->model = video_model(body.contains("model") && body["model"].is_string() ? body["model"].get_string() : "");
    } catch (const std::exception& e) {
        return fail(400, e.what(), "invalid_request_error");
    }
    job->model_name = job->model.name;
    // Length: seconds as a number or a numeric string (OpenAI's), 5 to 15, snapped up to the frames the VAE decodes.
    double seconds = double(video::Engine::kMinFrames) / video::Engine::kFps;
    if (body.contains("seconds") && !body["seconds"].is_null()) {
        const auto& v = body["seconds"];
        double s = 0;
        if (v.is_number()) {
            s = v.get_number();
        } else if (v.is_string()) {
            const std::string t = v.get_string();
            if (t.empty() || std::from_chars(t.data(), t.data() + t.size(), s).ptr != t.data() + t.size())
                return fail(400, "seconds must be a number (5 to 15)", "invalid_request_error");
        } else {
            return fail(400, "seconds must be a number (5 to 15)", "invalid_request_error");
        }
        if (!(s >= kMinSeconds) || s > kMaxSeconds)
            return fail(400, std::format("seconds must be from {:g} to {:g}", kMinSeconds, kMaxSeconds), "invalid_request_error");
        seconds = s;
    }
    job->frames = video::Engine::align_frames(uint32_t(std::lround(seconds * video::Engine::kFps)));
    job->seconds = double(job->frames) / video::Engine::kFps;
    uint32_t width = 0, height = 0;
    if (body.contains("size") && body["size"].is_string() && body["size"].get_string() != "auto") {
        const std::string sz = body["size"].get_string();
        unsigned w = 0, h = 0;
        char x = 0;
        if (std::sscanf(sz.c_str(), "%u%c%u", &w, &x, &h) != 3 || x != 'x' || w % 32 || h % 32 || w < 128 || h < 128 ||
            double(w) * h > kMaxArea)
            return fail(400, "size must be WIDTHxHEIGHT, multiples of 32 from 128, at most 1344x768 pixels' area twice over (or auto)",
                        "invalid_request_error");
        width = w;
        height = h;
    } else if (body.contains("size") && !body["size"].is_null() && !body["size"].is_string()) {
        return fail(400, "size must be a string WIDTHxHEIGHT or auto", "invalid_request_error");
    }
    job->steps = job->model.steps;
    if (body.contains("steps") && !body["steps"].is_null()) {
        const auto& v = body["steps"];
        if (!v.is_number() || v.get_number() < 0 || v.get_number() > 100 || v.get_number() != std::floor(v.get_number()))
            return fail(400, "steps must be a whole number from 0 (the model's) to 100", "invalid_request_error");
        job->steps = uint32_t(v.get_number());
    }
    if (job->steps == 1)
        return fail(400, "steps: at least 2 (one forward), or 0 for the model's", "invalid_request_error");
    job->loras = job->model.loras;
    if (body.contains("loras") && !body["loras"].is_null()) {
        try {
            std::string text;
            (void)glz::write_json(body["loras"], text);
            job->loras = resolve_loras(text, lora_choices(job->model.loras, job->model.lora_dir));
        } catch (const std::exception& e) {
            return fail(400, std::format("loras: {}", e.what()), "invalid_request_error");
        }
    }
    std::optional<uint64_t> seed;
    if (body.contains("seed") && !body["seed"].is_null()) {
        const auto& v = body["seed"];
        uint64_t x = 0;
        if (v.is_number() && v.get_number() >= 0 && v.get_number() <= 9007199254740992.0 && v.get_number() == std::floor(v.get_number()))
            x = uint64_t(v.get_number());
        else if (const std::string t = v.is_string() ? v.get_string() : std::string();
                 t.empty() || std::from_chars(t.data(), t.data() + t.size(), x).ptr != t.data() + t.size())
            return fail(400, "seed must be a whole number from 0 (a string past 2^53)", "invalid_request_error");
        seed = x;
    }
    // Keyframes: OpenAI's input_reference (an image) as the first frame, last_frame as the last.
    std::vector<fs::path> temps;
    const auto cleanup = std::shared_ptr<void>(nullptr, [&](void*) {
        for (const auto& p : temps)
            fs::remove(p);
    });
    for (auto [key, first] : {std::pair{"input_reference", true}, {"last_frame", false}}) {
        if (!body.contains(key) || body[key].is_null())
            continue;
        try {
            const auto bytes = media_bytes(body[key], key);
            temps.push_back(temp_file(bytes, "img"));
            (first ? job->request.first : job->request.last) = video::read_image(temps.back().string());
            (first ? job->first : job->last) = true;
        } catch (const std::exception& e) {
            return fail(400, std::format("{}: {}", key, e.what()), "invalid_request_error");
        }
    }
    if (body.contains("references") && !body["references"].is_null()) {
        const auto& refs = body["references"];
        if (!refs.is_array())
            return fail(400, "references must be an array of {\"type\": image|video|audio, \"data\": BASE64}", "invalid_request_error");
        if (refs.get_array().size() > kMaxReferences)
            return fail(400, std::format("references: at most {}", kMaxReferences), "invalid_request_error");
        if (!job->model.references)
            return fail(400, std::format("references: {} does not take references (a Ref2VA model does)", job->model.name), "invalid_request_error");
        if (job->first || job->last)
            return fail(400, "references and keyframes do not go together", "invalid_request_error");
        size_t i = 0;
        for (const auto& r : refs.get_array()) {
            const std::string what = std::format("references[{}]", i++);
            if (!r.is_object() || !r.contains("type") || !r["type"].is_string())
                return fail(400, what + " must be an object with a type (image, video or audio)", "invalid_request_error");
            const std::string type = r["type"].get_string();
            video::Reference ref;
            try {
                const auto bytes = media_bytes(r, what.c_str());
                if (type == "image") {
                    ref.kind = video::Reference::Kind::image;
                    temps.push_back(temp_file(bytes, "img"));
                    ref.image = video::read_image(temps.back().string());
                } else if (type == "video" || type == "audio") {
                    ref.kind = type == "video" ? video::Reference::Kind::video : video::Reference::Kind::audio;
                    temps.push_back(temp_file(bytes, type == "video" ? "mp4" : "m4a"));
                    auto media = video::read_media(temps.back().string(), type == "video", true, job->seconds);
                    if (type == "video") {
                        if (!media.frames)
                            throw std::runtime_error("no video track");
                        ref.frames = media.frames;
                        ref.width = media.width;
                        ref.height = media.height;
                        ref.fps = media.fps;
                        ref.rgb = std::move(media.rgb);
                    } else if (!media.channels) {
                        throw std::runtime_error("no audio track");
                    }
                    ref.audio = std::move(media.audio);
                    ref.channels = media.channels;
                    ref.sample_rate = media.sample_rate;
                } else {
                    throw std::invalid_argument("type must be image, video or audio");
                }
            } catch (const std::exception& e) {
                return fail(400, std::format("{}: {}", what, e.what()), "invalid_request_error");
            }
            job->request.references.push_back(std::move(ref));
        }
        job->references = job->request.references.size();
        bool visual = false;
        for (const auto& r : job->request.references)
            visual |= r.kind != video::Reference::Kind::audio;
        if (!visual)
            return fail(400, "references: a soundtrack cannot be the only reference", "invalid_request_error");
    }
    // The canvas, as the engine will choose it, so the job says its size from the start.
    try {
        if (!width)
            std::tie(width, height) = job->first ? video::Engine::canvas(job->request.first->width, job->request.first->height)
                                                 : video::Engine::canvas(16, 9);
    } catch (const std::exception& e) {
        return fail(400, std::format("input_reference: {}", e.what()), "invalid_request_error");
    }
    job->width = width;
    job->height = height;
    if (!seed) {
        std::random_device rd;
        seed = (uint64_t(rd()) << 32) | rd();
    }
    job->seed = *seed;
    job->request.prompt = job->prompt;
    job->request.width = width;
    job->request.height = height;
    job->request.frames = job->frames;
    job->request.steps = job->steps;
    job->request.seed = job->seed;
    job->created_at = now_seconds();
    std::string json;
    {
        std::lock_guard lock(m);
        if (queue.size() >= opts.max_waiting + (running ? 0u : 1u))
            return fail(503, std::format("{} videos are already waiting", opts.max_waiting), "server_busy");
        job->id = std::format("video_{}_{}", job->created_at, next_id.fetch_add(1) + 1);
        job->mp4 = fs::path(opts.output_dir) / (job->id + ".mp4");
        job->png = fs::path(opts.output_dir) / (job->id + ".png");
        jobs[job->id] = job;
        queue.push_back(job);
        json = job_json(*job);
    }
    cv.notify_all();
    std::string lora_names;
    for (const auto& l : job->loras)
        lora_names += std::format("{}{} {:g}", lora_names.empty() ? ", LoRAs " : ", ", fs::path(l.path).stem().string(), l.scale);
    log::info("{}: {}x{}, {} frames ({:.2f} s), {} steps with {}{}{}{}{}, queued", job->id, width, height, job->frames,
              job->seconds, job->steps ? std::to_string(job->steps) : "the model's", job->model.name, lora_names,
              job->first ? ", a first frame" : "", job->last ? ", a last frame" : "",
              job->references ? std::format(", {} references", job->references) : "");
    res.set_content(json, "application/json");
}

void VideoServer::Impl::work()
{
    for (;;) {
        std::shared_ptr<Job> job;
        {
            std::unique_lock lock(m);
            cv.wait(lock, [&] { return stopping || !queue.empty(); });
            if (stopping)
                return;
            job = queue.front();
            queue.pop_front();
            running = job;
            job->status = "in_progress";
            job->stage = "loading the model";
        }
        run_job(job);
        {
            std::lock_guard lock(m);
            running.reset();
        }
    }
}

void VideoServer::Impl::run_job(const std::shared_ptr<Job>& job)
{
    const auto t0 = std::chrono::steady_clock::now();
    auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    auto set = [&](auto&& f) {
        std::lock_guard lock(m);
        f();
    };
    try {
        std::string key = job->model.key();
        for (const auto& l : job->loras)
            key += std::format("|{}:{}", l.path, l.scale);
        auto engine = residency.acquire(key, "video model " + job->model.name, [&] { return load(job->model, job->loras); });
        if (job->cancel)
            throw Cancelled();
        job->request.stage = [&](const std::string& what) {
            set([&] { job->stage = what; });
        };
        auto last = std::chrono::steady_clock::now();
        job->request.progress = [&](uint32_t step, uint32_t steps) {
            if (job->cancel)
                throw Cancelled();
            set([&] {
                job->stage = std::format("step {} of {}", step, steps);
                job->progress = int(std::min<uint32_t>(99, step * 95 / std::max(1u, steps)));
            });
            const auto now = std::chrono::steady_clock::now();
            if (now - last >= kProgressEvery || step == steps) {
                last = now;
                log::info("{}: step {} of {} ({:.1f} s a step so far)", job->id, step, steps, secs() / step);
            }
        };
        const auto res = engine->generate(job->request);
        set([&] {
            job->stage = "writing the file";
            job->progress = 99;
        });
        fs::create_directories(job->mp4.parent_path());
        video::write_mp4(job->mp4.string(), res.width, res.height, res.fps, res.frames, res.rgb, res.audio, res.sample_rate);
        const uint64_t frame = uint64_t(res.width) * res.height * 3;
        video::write_png(job->png.string(), res.width, res.height, std::span(res.rgb).subspan(frame * (res.frames / 2), frame));
        set([&] {
            job->width = res.width;
            job->height = res.height;
            job->status = "completed";
            job->stage = "done";
            job->progress = 100;
            job->completed_at = now_seconds();
        });
        log::info("{}: {}x{}, {} frames in {:.1f} s (seed {}): {}", job->id, res.width, res.height, res.frames, secs(),
                  job->seed, job->mp4.string());
    } catch (const Cancelled&) {
        log::info("{}: cancelled after {:.1f} s", job->id, secs());
        set([&] {
            job->status = "failed";
            job->stage = "cancelled";
            job->error_code = "cancelled";
            job->error_message = "cancelled";
            job->completed_at = now_seconds();
        });
    } catch (const std::exception& e) {
        log::error("{}: {}", job->id, e.what());
        set([&] {
            job->status = "failed";
            job->stage = "failed";
            job->error_code = "server_error";
            job->error_message = e.what();
            job->completed_at = now_seconds();
        });
    }
    job->request = {}; // the keyframes' and references' bytes
}

void VideoServer::Impl::list(const httplib::Request& req, httplib::Response& res)
{
    size_t limit = 20;
    if (req.has_param("limit")) {
        const std::string t = req.get_param_value("limit");
        if (t.empty() || std::from_chars(t.data(), t.data() + t.size(), limit).ptr != t.data() + t.size() || limit > 100) {
            res.status = 400;
            res.set_content(error_body("limit must be a whole number up to 100", "invalid_request_error"), "application/json");
            return;
        }
    }
    const std::string after = req.has_param("after") ? req.get_param_value("after") : "";
    const bool desc = req.has_param("order") && req.get_param_value("order") == "desc";
    std::lock_guard lock(m);
    std::vector<std::shared_ptr<Job>> all;
    for (const auto& [id, j] : jobs)
        all.push_back(j);
    std::ranges::sort(all, [&](const auto& a, const auto& b) {
        return desc ? std::tie(b->created_at, b->id) < std::tie(a->created_at, a->id)
                    : std::tie(a->created_at, a->id) < std::tie(b->created_at, b->id);
    });
    size_t start = 0;
    if (!after.empty())
        for (size_t i = 0; i < all.size(); ++i)
            if (all[i]->id == after) {
                start = i + 1;
                break;
            }
    std::string data;
    size_t n = 0;
    for (size_t i = start; i < all.size() && n < limit; ++i, ++n)
        data += (data.empty() ? "" : ",") + job_json(*all[i]);
    const bool more = start + n < all.size();
    res.set_content(std::format(R"({{"object":"list","data":[{}],"first_id":{},"last_id":{},"has_more":{}}})", data,
                                n ? json_quoted(all[start]->id) : "null", n ? json_quoted(all[start + n - 1]->id) : "null", more),
                    "application/json");
}

void VideoServer::Impl::retrieve(const httplib::Request& req, httplib::Response& res)
{
    const auto job = find(req.matches[1]);
    if (!job) {
        res.status = 404;
        res.set_content(error_body(std::format("no video {}", req.matches[1].str()), "not_found_error"), "application/json");
        return;
    }
    std::lock_guard lock(m);
    res.set_content(job_json(*job), "application/json");
}

void VideoServer::Impl::content(const httplib::Request& req, httplib::Response& res)
{
    const auto job = find(req.matches[1]);
    if (!job) {
        res.status = 404;
        res.set_content(error_body(std::format("no video {}", req.matches[1].str()), "not_found_error"), "application/json");
        return;
    }
    const std::string variant = req.has_param("variant") ? req.get_param_value("variant") : "video";
    if (variant != "video" && variant != "thumbnail") {
        res.status = 400;
        res.set_content(error_body("variant: video or thumbnail", "invalid_request_error"), "application/json");
        return;
    }
    fs::path file;
    {
        std::lock_guard lock(m);
        if (job->status != "completed") {
            res.status = 409;
            res.set_content(error_body(std::format("video {} is {} ({})", job->id, job->status, job->stage), "invalid_request_error"),
                            "application/json");
            return;
        }
        file = variant == "video" ? job->mp4 : job->png;
    }
    std::ifstream in(file, std::ios::binary);
    std::string bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    if (!in || bytes.empty()) {
        res.status = 404;
        res.set_content(error_body(std::format("{} is gone", file.string()), "not_found_error"), "application/json");
        return;
    }
    res.set_header("Content-Disposition", std::format("attachment; filename=\"{}\"", file.filename().string()));
    res.set_content(std::move(bytes), variant == "video" ? "video/mp4" : "image/png");
}

void VideoServer::Impl::remove(const httplib::Request& req, httplib::Response& res)
{
    const std::string id = req.matches[1];
    std::shared_ptr<Job> job;
    {
        std::lock_guard lock(m);
        const auto it = jobs.find(id);
        if (it != jobs.end()) {
            job = it->second;
            jobs.erase(it);
            std::erase(queue, job);
            job->cancel = true;
        }
    }
    if (!job) {
        res.status = 404;
        res.set_content(error_body(std::format("no video {}", id), "not_found_error"), "application/json");
        return;
    }
    std::error_code ec;
    fs::remove(job->mp4, ec);
    fs::remove(job->png, ec);
    log::info("{}: deleted", id);
    res.set_content(std::format(R"({{"id":{},"object":"video.deleted","deleted":true}})", json_quoted(id)), "application/json");
}

void VideoServer::Impl::model_list(httplib::Response& res)
{
    std::vector<VideoModel> list;
    if (!opts.video_models.empty())
        list = video_models(opts.video_models);
    std::string def, loaded = residency.loaded();
    try {
        const VideoModel d = video_model("");
        def = d.name;
        if (list.empty())
            list.push_back(d);
    } catch (const std::exception&) {
    }
    std::string data;
    for (const auto& m : list) {
        std::string l;
        for (const auto& x : m.loras)
            l += std::format(R"({}{{"name":{},"scale":{:g}}})", l.empty() ? "" : ",", json_quoted(fs::path(x.path).stem().string()), x.scale);
        data += std::format(R"({}{{"id":{},"family":"MiniMax-H3","references":{},"default":{},"loaded":{},"steps":{},"loras":[{}])",
                            data.empty() ? "" : ",", json_quoted(m.name), m.references, m.name == def,
                            loaded == "video model " + m.name, m.steps ? m.steps : 50, l);
        if (m.options.sparse > 0)
            data += std::format(R"(,"sparse":{:g})", m.options.sparse);
        if (!m.description.empty())
            data += std::format(R"(,"description":{})", json_quoted(m.description));
        if (!m.prompting.empty())
            data += std::format(R"(,"prompting":{})", json_quoted(m.prompting));
        data += "}";
    }
    res.set_content(std::format(R"({{"object":"list","data":[{}]}})", data), "application/json");
}

void VideoServer::Impl::loras(const httplib::Request& req, httplib::Response& res)
{
    VideoModel model;
    try {
        model = video_model(req.has_param("model") ? req.get_param_value("model") : "");
    } catch (const std::exception& e) {
        res.status = 400;
        res.set_content(error_body(e.what(), "invalid_request_error"), "application/json");
        return;
    }
    std::string data;
    for (const auto& c : lora_choices(model.loras, model.lora_dir)) {
        data += std::format(R"({}{{"name":{})", data.empty() ? "" : ",", json_quoted(c.name));
        if (c.scale)
            data += std::format(R"(,"default_scale":{:g})", *c.scale);
        const auto notes = lora_notes(c.path);
        if (notes.steps)
            data += std::format(R"(,"steps":{})", notes.steps);
        if (!notes.trigger.empty())
            data += std::format(R"(,"trigger":{})", json_quoted(notes.trigger));
        if (!notes.description.empty())
            data += std::format(R"(,"description":{})", json_quoted(notes.description));
        data += "}";
    }
    res.set_content(std::format(R"({{"object":"list","data":[{}]}})", data), "application/json");
}

void VideoServer::Impl::status(httplib::Response& res)
{
    std::lock_guard lock(m);
    res.set_content(std::format(R"({{"model":{},"state":"{}","queued":{},"running":{}}})", json_quoted(residency.loaded()),
                                residency.state(), queue.size(),
                                running ? std::format(R"({{"id":{},"stage":{},"progress":{}}})", json_quoted(running->id),
                                                      json_quoted(running->stage), running->progress)
                                        : "null"),
                    "application/json");
}

VideoServer::VideoServer(VideoResidency& residency, LoadVideo load, VideoServerOptions opts)
    : impl_(std::make_unique<Impl>(residency, std::move(load), std::move(opts)))
{
    auto& h = impl_->http;
    h.set_payload_max_length(size_t(kMaxMedia) * 2);
    h.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        const auto& keys = impl_->opts.api_keys;
        if (req.path == "/health" ||
            http::allowed(req.get_header_value("Authorization"), req.get_header_value("x-api-key"),
                          req.get_header_value("Host"), req.get_header_value("Origin"), keys))
            return httplib::Server::HandlerResponse::Unhandled;
        res.status = keys.empty() ? 403 : 401;
        res.set_content(error_body(keys.empty() ? "without an API key, requests from web pages or other hosts are refused"
                                                : "invalid API key",
                                   keys.empty() ? "permission_denied" : "invalid_api_key"),
                        "application/json");
        return httplib::Server::HandlerResponse::Handled;
    });
    for (const char* prefix : {"/v1", ""}) {
        const std::string p = prefix;
        h.Post(p + "/videos", [this](const httplib::Request& req, httplib::Response& res) { impl_->create(req, res); });
        h.Get(p + "/videos", [this](const httplib::Request& req, httplib::Response& res) { impl_->list(req, res); });
        h.Get(p + "/videos/models", [this](const httplib::Request&, httplib::Response& res) { impl_->model_list(res); });
        h.Get(p + "/videos/loras", [this](const httplib::Request& req, httplib::Response& res) { impl_->loras(req, res); });
        h.Get(p + R"(/videos/([^/]+)/content)", [this](const httplib::Request& req, httplib::Response& res) { impl_->content(req, res); });
        h.Get(p + R"(/videos/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) { impl_->retrieve(req, res); });
        h.Delete(p + R"(/videos/([^/]+))", [this](const httplib::Request& req, httplib::Response& res) { impl_->remove(req, res); });
    }
    h.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });
    h.Get("/v1/status", [this](const httplib::Request&, httplib::Response& res) { impl_->status(res); });
}

VideoServer::~VideoServer() = default;

bool VideoServer::run()
{
    auto& im = *impl_;
    const int port = im.opts.port ? (im.http.bind_to_port(im.opts.host, im.opts.port) ? im.opts.port : -1)
                                  : im.http.bind_to_any_port(im.opts.host);
    if (port < 0)
        return false;
    im.port = port;
    return im.http.listen_after_bind();
}

void VideoServer::stop() { impl_->http.stop(); }

int VideoServer::port() const { return impl_->port; }

VideoModel VideoServer::model(const std::string& name) const { return impl_->video_model(name); }

} // namespace mako::server
