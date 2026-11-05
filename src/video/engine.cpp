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
                const float* row = &l.video[(l.condition_rows + (uint64_t(f) * ph + py) * pw + px) * H3::kVideoPatch];
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

models::QwenVision& Engine::vision()
{
    if (!vision_) {
        const auto t0 = Clock::now();
        vision_ = std::make_unique<models::QwenVision>(
            opts_.text_encoder.empty() ? dir_ + "/text_encoder" : opts_.text_encoder, ctx_, opts_.telemetry);
        log::info("vision tower loaded in {:.1f} s", seconds_since(t0));
    }
    return *vision_;
}

Image Engine::prepare_keyframe(const Image& img, uint32_t width, uint32_t height, bool first)
{
    if (img.width == width && img.height == height)
        return img;
    if (first)
        return resize_lanczos(img, width, height);
    const double scale = std::max(double(width) / img.width, double(height) / img.height);
    const uint32_t rw = std::max<uint32_t>(width, uint32_t(std::lround(img.width * scale)));
    const uint32_t rh = std::max<uint32_t>(height, uint32_t(std::lround(img.height * scale)));
    const Image big = resize_lanczos(img, rw, rh);
    const uint32_t left = (rw - width) / 2, top = (rh - height) / 2;
    Image out{width, height, std::vector<uint8_t>(uint64_t(width) * height * 3)};
    for (uint32_t y = 0; y < height; ++y)
        std::copy_n(&big.rgb[(uint64_t(top + y) * rw + left) * 3], uint64_t(width) * 3, &out.rgb[uint64_t(y) * width * 3]);
    return out;
}

std::vector<float> Engine::prompt_embeddings(const std::string& prompt, std::span<const Image> keyframes,
                                             std::vector<uint8_t>* tags)
{
    // The prompt as it is: no chat template, no special tokens around it (the ones written in it are recognized);
    // each keyframe before it as a "<Picture i>: " label and a vision block (start, the merged tokens' pads, end),
    // the block's rows tagged as video for the transformer's modulation.
    std::vector<int32_t> ids;
    std::vector<uint8_t> tg;
    std::vector<models::QwenVision::Features> feats;
    std::vector<models::QwenText::Vision> blocks;
    auto special = [&](const char* name) {
        const auto v = tok_->encode(name, true);
        if (v.size() != 1)
            throw std::runtime_error(std::format("the tokenizer has no {} token", name));
        return v[0];
    };
    for (size_t i = 0; i < keyframes.size(); ++i) {
        const Image& k = keyframes[i];
        feats.push_back(vision().encode(k.rgb, k.height, k.width));
        const auto label = tok_->encode(std::format("<Picture {}>: ", i + 1), false);
        ids.insert(ids.end(), label.begin(), label.end());
        tg.insert(tg.end(), label.size(), uint8_t(H3::kTextTag));
        const uint32_t n = feats.back().rows * feats.back().cols;
        ids.push_back(special("<|vision_start|>"));
        models::QwenText::Vision b;
        b.first = uint32_t(ids.size());
        b.rows = feats.back().rows;
        b.cols = feats.back().cols;
        ids.insert(ids.end(), n, special("<|image_pad|>"));
        ids.push_back(special("<|vision_end|>"));
        tg.insert(tg.end(), n + 2, uint8_t(H3::kVideoTag));
        blocks.push_back(std::move(b));
    }
    for (size_t i = 0; i < feats.size(); ++i) { // the features' storage is final now
        blocks[i].embeds = feats[i].embeds;
        for (const auto& d : feats[i].deepstack)
            blocks[i].deepstack.emplace_back(d);
    }
    const auto prompt_ids = tok_->encode(prompt, true);
    if (prompt_ids.empty())
        throw std::runtime_error("the prompt has no tokens");
    ids.insert(ids.end(), prompt_ids.begin(), prompt_ids.end());
    tg.insert(tg.end(), prompt_ids.size(), uint8_t(H3::kTextTag));
    const uint32_t taps[1] = {kTextLayer};
    auto out = text_encoder().encode(ids, 0, taps, false, blocks);
    if (!opts_.keep_text_encoder)
        free_text_encoder();
    if (tags)
        *tags = std::move(tg);
    return out;
}

Engine::Latents Engine::denoise(const Request& r)
{
    Latents L;
    if ((r.width == 0) != (r.height == 0))
        throw std::runtime_error("width and height go together");
    std::vector<Image> keyframes;
    std::vector<uint8_t> anchors; // per keyframe: at the last generated frame (else the first)
    for (const auto* k : {&r.first, &r.last})
        if (k->has_value()) {
            keyframes.push_back(**k);
            anchors.push_back(k == &r.last);
        }
    if (r.width == 0)
        std::tie(L.width, L.height) = keyframes.empty() ? canvas(16, 9) : canvas(keyframes[0].width, keyframes[0].height);
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

    for (size_t i = 0; i < keyframes.size(); ++i)
        keyframes[i] = prepare_keyframe(keyframes[i], L.width, L.height, i == 0);
    std::vector<uint8_t> tags;
    const auto text = prompt_embeddings(r.prompt, keyframes, &tags);
    std::mt19937_64 rng(r.seed);
    std::normal_distribution<float> nd;
    // The keyframes' latents, noised to t = 0.999 and packed into rows (a 2x2 patch's [24][2][2]), ahead of the
    // generated rows; their noise is drawn before the generated rows'.
    const uint32_t ph = L.latent_h / 2, pw = L.latent_w / 2, rpf = ph * pw;
    std::vector<float> cond;
    if (!r.keyframe_noise.empty() && r.keyframe_noise.size() != keyframes.size())
        throw std::runtime_error("keyframe noise: one per keyframe");
    for (size_t i = 0; i < keyframes.size(); ++i) {
        const auto z = video_vae().encode(keyframes[i].rgb, L.height, L.width);
        std::vector<float> noise(z.size());
        if (!r.keyframe_noise.empty()) {
            if (r.keyframe_noise[i].size() != z.size())
                throw std::runtime_error("keyframe noise: [latent_h][latent_w][24]");
            std::ranges::copy(r.keyframe_noise[i], noise.begin());
        } else {
            for (float& v : noise)
                v = nd(rng);
        }
        const float t = 0.999f, u = float(1.0 - 0.999);
        for (uint32_t py = 0; py < ph; ++py)
            for (uint32_t px = 0; px < pw; ++px)
                for (uint32_t c = 0; c < VideoVae::kLatent; ++c)
                    for (uint32_t dy = 0; dy < 2; ++dy)
                        for (uint32_t dx = 0; dx < 2; ++dx) {
                            const uint64_t v = ((uint64_t(2 * py + dy)) * L.latent_w + 2 * px + dx) * VideoVae::kLatent + c;
                            cond.push_back(t * z[v] + u * noise[v]);
                        }
    }
    H3& dit = transformer();
    H3::Layout layout;
    layout.text = uint32_t(text.size() / H3::kText);
    layout.frames = L.latent_frames;
    layout.h = L.latent_h;
    layout.w = L.latent_w;
    layout.audio = L.audio_latents;
    layout.cond_frames = uint32_t(keyframes.size());
    layout.cond_last = anchors;
    layout.text_tags = tags;
    const uint32_t V = layout.video_rows(), A = layout.audio_rows(), C = layout.cond_rows();
    L.condition_rows = C;
    L.video.resize(uint64_t(V) * H3::kVideoPatch);
    L.audio.resize(uint64_t(A) * H3::kAudioChannels);
    if (cond.size() != uint64_t(C) * H3::kVideoPatch)
        throw std::runtime_error("the keyframe rows do not match the layout");
    std::ranges::copy(cond, L.video.begin());
    const std::span<float> gen(L.video.data() + uint64_t(C) * H3::kVideoPatch, uint64_t(V - C) * H3::kVideoPatch);
    if (!r.noise_video.empty() || !r.noise_audio.empty()) {
        if (r.noise_video.size() != gen.size() || r.noise_audio.size() != L.audio.size())
            throw std::runtime_error(std::format("noise: video rows [{}][96] and audio rows [{}][32]", V - C, A));
        std::ranges::copy(r.noise_video, gen.begin());
        std::ranges::copy(r.noise_audio, L.audio.begin());
    } else {
        for (float& v : gen)
            v = nd(rng);
        for (float& v : L.audio)
            v = nd(rng);
    }
    (void)rpf;
    dit.begin(text, layout);
    const uint32_t n = uint32_t(sig_v.size()) - 1;
    // Euler as the reference steps it: the velocity points at the data (x0 = x + sigma v, sigma from the timestep
    // the transformer saw), the next sample the blend r x + (1 - r) x0 with r the next sigma over this one.
    // The keyframe rows ride through unchanged: only the generated rows step.
    auto euler = [](std::span<float> x, std::span<const float> v, float t, float sigma, float sigma_next) {
        const float st = 1.0f - t, ratio = sigma_next / sigma;
        for (size_t i = 0; i < x.size(); ++i) {
            const float x0 = x[i] + st * v[i];
            x[i] = ratio * x[i] + (1.0f - ratio) * x0;
        }
    };
    for (uint32_t i = 0; i < n; ++i) {
        const float tv = 1.0f - sig_v[i], ta = 1.0f - sig_a[i];
        const auto vel = dit.step(L.video, L.audio, tv, ta);
        euler(gen, std::span(vel.video).subspan(uint64_t(C) * H3::kVideoPatch), tv, sig_v[i], sig_v[i + 1]);
        euler(L.audio, vel.audio, ta, sig_a[i], sig_a[i + 1]);
        if (r.progress)
            r.progress(i + 1, n);
    }
    return L;
}

} // namespace mako::video
