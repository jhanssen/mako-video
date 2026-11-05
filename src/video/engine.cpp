#include "video/engine.hpp"

#include "log/log.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <random>
#include <stdexcept>

namespace mako::video {

namespace {

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

} // namespace

Engine::Engine(const std::string& dir, mtl::Context& ctx, const Options& opts) : ctx_(ctx), opts_(opts), dir_(dir)
{
    if (!std::filesystem::exists(dir + "/model_index.json"))
        throw std::runtime_error(dir + ": no model_index.json");
    const std::string tokenizer = std::filesystem::exists(dir + "/tokenizer/tokenizer.json")
                                      ? dir + "/tokenizer/tokenizer.json"
                                      : dir + "/processor/tokenizer.json";
    auto t = text::Tokenizer::load_json(tokenizer);
    if (!t)
        throw std::runtime_error(t.error());
    tok_ = std::move(*t);
}

uint32_t Engine::align_frames(uint32_t frames)
{
    if (!frames)
        throw std::runtime_error("frames must be positive");
    while (frames % kClipFrames != kClipLatents)
        ++frames;
    return frames;
}

uint32_t Engine::audio_latents(uint32_t frames)
{
    return uint32_t(std::lround(double(frames) / kFps * kAudioLatentsPerSecond));
}

std::pair<uint32_t, uint32_t> Engine::canvas(double aspect_w, double aspect_h)
{
    constexpr double kShort = 768, kMaxPixels = 768 * 1344;
    static constexpr uint32_t kMultiple = 32;
    if (!(aspect_w > 0) || !(aspect_h > 0))
        throw std::runtime_error("the aspect ratio must be positive");
    const double ratio = aspect_w / aspect_h;
    if (ratio < 0.25 || ratio > 4.0)
        throw std::runtime_error("aspect ratios from 1:4 to 4:1");
    double w = ratio >= 1.0 ? kShort * ratio : kShort, h = ratio >= 1.0 ? kShort : kShort / ratio;
    if (w * h > kMaxPixels) {
        const double s = std::sqrt(kMaxPixels / (w * h));
        w *= s;
        h *= s;
    }
    auto snap = [](double v) { return std::max<uint32_t>(kMultiple, uint32_t(std::lround(v / kMultiple)) * kMultiple); };
    return {snap(w), snap(h)};
}

std::vector<float> Engine::sigmas(uint32_t steps, float shift)
{
    if (steps < 2)
        throw std::runtime_error("at least 2 schedule points");
    if (!(shift > 0))
        throw std::runtime_error("the shift must be positive");
    // linspace(1, 0, steps) as torch computes it in float (the two halves from either end), then the shift
    // s * x / (1 + (s - 1) * x) in float, consecutive duplicates dropped.
    std::vector<float> s;
    const float step = (0.0f - 1.0f) / float(steps - 1);
    for (uint32_t i = 0; i < steps; ++i) {
        const float base = i < steps / 2 ? 1.0f + float(i) * step : 0.0f - float(steps - 1 - i) * step;
        const float v = shift * base / (1.0f + (shift - 1.0f) * base);
        if (s.empty() || s.back() != v)
            s.push_back(v);
    }
    return s;
}

models::QwenText& Engine::text_encoder()
{
    if (!text_) {
        const auto t0 = Clock::now();
        const std::string dir = opts_.text_encoder.empty() ? dir_ + "/text_encoder" : opts_.text_encoder;
        // Layer 50's output conditions the transformer: the layers past it are not loaded.
        text_ = std::make_unique<models::QwenText>(dir, ctx_, opts_.telemetry,
                                                   models::QwenText::Options{.layers = kTextLayer});
        if (text_->width() != H3::kText)
            throw std::runtime_error(std::format("text encoder: {} wide, not {}", text_->width(), H3::kText));
        log::info("text encoder: {} layers in {:.1f} s", text_->layers(), seconds_since(t0));
    }
    return *text_;
}

H3& Engine::transformer()
{
    if (!dit_) {
        const auto t0 = Clock::now();
        dit_ = std::make_unique<H3>(opts_.transformer.empty() ? dir_ + "/transformer" : opts_.transformer, ctx_,
                                    opts_.telemetry, opts_.int8);
        log::info("transformer loaded in {:.1f} s", seconds_since(t0));
        for (const Lora& l : loras_)
            dit_->add_lora(l.path, l.scale, true);
    }
    return *dit_;
}

void Engine::add_lora(const Lora& l)
{
    if (dit_)
        dit_->add_lora(l.path, l.scale, true);
    else
        loras_.push_back(l);
}

VideoVae& Engine::video_vae()
{
    if (!vae_) {
        const auto t0 = Clock::now();
        vae_ = std::make_unique<VideoVae>(dir_ + "/video_vae", ctx_, opts_.telemetry);
        log::info("video VAE loaded in {:.1f} s", seconds_since(t0));
    }
    return *vae_;
}

AudioVae& Engine::audio_vae()
{
    if (!avae_) {
        const auto t0 = Clock::now();
        avae_ = std::make_unique<AudioVae>(dir_ + "/audio_vae", ctx_, opts_.telemetry);
        log::info("audio VAE loaded in {:.1f} s", seconds_since(t0));
    }
    return *avae_;
}

std::vector<float> Engine::voxels(const Latents& l)
{
    // A row holds a 2x2 patch of a latent frame as [channel][dy][dx]; voxels are [frame][y][x][channel].
    const uint32_t F = l.latent_frames, h = l.latent_h, w = l.latent_w, ph = h / 2, pw = w / 2, C = H3::kAudioChannels;
    (void)C;
    std::vector<float> v(uint64_t(F) * h * w * VideoVae::kLatent);
    for (uint32_t f = 0; f < F; ++f)
        for (uint32_t py = 0; py < ph; ++py)
            for (uint32_t px = 0; px < pw; ++px) {
                const float* row = &l.video[((uint64_t(f) * ph + py) * pw + px) * H3::kVideoPatch];
                for (uint32_t c = 0; c < VideoVae::kLatent; ++c)
                    for (uint32_t dy = 0; dy < 2; ++dy)
                        for (uint32_t dx = 0; dx < 2; ++dx)
                            v[((uint64_t(f) * h + 2 * py + dy) * w + 2 * px + dx) * VideoVae::kLatent + c] =
                                row[(c * 2 + dy) * 2 + dx];
            }
    return v;
}

void Engine::decode_video(const Latents& l, const VideoVae::FrameSink& sink)
{
    video_vae().decode(voxels(l), l.latent_frames, l.latent_h, l.latent_w, sink);
}

std::vector<float> Engine::decode_audio(const Latents& l) { return audio_vae().decode(l.audio, l.audio_latents); }

Engine::Result Engine::generate(const Request& r)
{
    const Latents l = denoise(r);
    Result res;
    res.width = l.width;
    res.height = l.height;
    res.frames = l.frames;
    res.rgb.resize(uint64_t(l.frames) * l.height * l.width * 3);
    const auto t0 = Clock::now();
    decode_video(l, [&](uint32_t i, std::span<const float> rgb) {
        if (i >= l.frames)
            throw std::runtime_error("the video VAE decoded more frames than asked for");
        uint8_t* dst = &res.rgb[uint64_t(i) * l.height * l.width * 3];
        for (size_t k = 0; k < rgb.size(); ++k)
            dst[k] = uint8_t(std::lround(std::clamp(rgb[k], 0.0f, 1.0f) * 255.0f));
    });
    log::info("video decoded in {:.1f} s", seconds_since(t0));
    const auto t1 = Clock::now();
    res.audio = decode_audio(l);
    log::info("audio decoded in {:.1f} s", seconds_since(t1));
    return res;
}

std::vector<float> Engine::prompt_embeddings(const std::string& prompt)
{
    // The prompt as it is: no chat template, no special tokens around it (the ones written in it are recognized).
    const auto ids = tok_->encode(prompt, true);
    if (ids.empty())
        throw std::runtime_error("the prompt has no tokens");
    const uint32_t taps[1] = {kTextLayer};
    auto out = text_encoder().encode(ids, 0, taps);
    if (!opts_.keep_text_encoder)
        text_.reset();
    return out;
}

Engine::Latents Engine::denoise(const Request& r)
{
    Latents L;
    if ((r.width == 0) != (r.height == 0))
        throw std::runtime_error("width and height go together");
    if (r.width == 0)
        std::tie(L.width, L.height) = canvas(16, 9);
    else
        std::tie(L.width, L.height) = std::pair{r.width, r.height};
    if (L.width % 32 || L.height % 32)
        throw std::runtime_error("width and height must be multiples of 32");
    L.frames = align_frames(r.frames);
    if (L.frames < kMinFrames || L.frames > kMaxFrames)
        throw std::runtime_error(std::format("{} frames (rounded up to 17 n + 5) is outside 5 to 15 seconds at 24 fps "
                                             "({} to {} frames)", r.frames, kMinFrames, kMaxFrames));
    L.latent_frames = latent_frames(L.frames);
    L.latent_h = L.height / 16;
    L.latent_w = L.width / 16;
    L.audio_latents = audio_latents(L.frames);
    const uint32_t steps = r.steps ? r.steps : default_steps();
    const auto sig_v = sigmas(steps, r.video_shift), sig_a = sigmas(steps, r.audio_shift);
    if (sig_v.size() != sig_a.size())
        throw std::runtime_error("the video and audio schedules differ in length");

    const auto text = prompt_embeddings(r.prompt);
    H3& dit = transformer();
    H3::Layout layout;
    layout.text = uint32_t(text.size() / H3::kText);
    layout.frames = L.latent_frames;
    layout.h = L.latent_h;
    layout.w = L.latent_w;
    layout.audio = L.audio_latents;
    const uint32_t V = layout.video_rows(), A = layout.audio_rows();
    L.video.resize(uint64_t(V) * H3::kVideoPatch);
    L.audio.resize(uint64_t(A) * H3::kAudioChannels);
    if (!r.noise_video.empty() || !r.noise_audio.empty()) {
        if (r.noise_video.size() != L.video.size() || r.noise_audio.size() != L.audio.size())
            throw std::runtime_error(std::format("noise: video rows [{}][96] and audio rows [{}][32]", V, A));
        std::ranges::copy(r.noise_video, L.video.begin());
        std::ranges::copy(r.noise_audio, L.audio.begin());
    } else {
        std::mt19937_64 rng(r.seed);
        std::normal_distribution<float> nd;
        for (float& v : L.video)
            v = nd(rng);
        for (float& v : L.audio)
            v = nd(rng);
    }
    dit.begin(text, layout);
    const uint32_t n = uint32_t(sig_v.size()) - 1;
    // Euler as the reference steps it: the velocity points at the data (x0 = x + sigma v, sigma from the timestep
    // the transformer saw), the next sample the blend r x + (1 - r) x0 with r the next sigma over this one.
    auto euler = [](std::vector<float>& x, std::span<const float> v, float t, float sigma, float sigma_next) {
        const float st = 1.0f - t, ratio = sigma_next / sigma;
        for (size_t i = 0; i < x.size(); ++i) {
            const float x0 = x[i] + st * v[i];
            x[i] = ratio * x[i] + (1.0f - ratio) * x0;
        }
    };
    for (uint32_t i = 0; i < n; ++i) {
        const float tv = 1.0f - sig_v[i], ta = 1.0f - sig_a[i];
        const auto vel = dit.step(L.video, L.audio, tv, ta);
        euler(L.video, vel.video, tv, sig_v[i], sig_v[i + 1]);
        euler(L.audio, vel.audio, ta, sig_a[i], sig_a[i + 1]);
        if (r.progress)
            r.progress(i + 1, n);
    }
    return L;
}

} // namespace mako::video
