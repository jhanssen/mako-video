#include "video/engine.hpp"

#include "log/log.hpp"
#include "video/audio.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <random>
#include <stdexcept>

#include <sys/sysctl.h>

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
        dit_->set_sparse(opts_.sparse, opts_.sparse_local);
        for (const Lora& l : loras_)
            dit_->add_lora(l.path, l.scale, true);
    }
    return *dit_;
}

void Engine::add_lora(const Lora& l)
{
    loras_.push_back(l);
    if (dit_)
        dit_->add_lora(l.path, l.scale, true);
}

bool Engine::keeps_transformer()
{
    if (opts_.keep_transformer)
        return *opts_.keep_transformer;
    if (keeps_transformer_)
        return *keeps_transformer_;
    // The weights' files, the transformer's and the text encoder's, against the machine's memory.
    auto bytes = [](const std::filesystem::path& p) {
        uint64_t n = 0;
        std::error_code ec;
        if (std::filesystem::is_regular_file(p, ec))
            return uint64_t(std::filesystem::file_size(p, ec));
        for (const auto& e : std::filesystem::recursive_directory_iterator(p, ec))
            if (e.is_regular_file(ec) && e.path().extension() == ".safetensors")
                n += e.file_size(ec);
        return n;
    };
    const uint64_t t = bytes(opts_.transformer.empty() ? dir_ + "/transformer" : opts_.transformer);
    const uint64_t te = bytes(opts_.text_encoder.empty() ? dir_ + "/text_encoder" : opts_.text_encoder);
    uint64_t mem = 0;
    size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) != 0)
        mem = 0;
    keeps_transformer_ = double(mem) >= 1.35 * double(t + te);
    log::info("the transformer ({:.0f} GB) and the text encoder ({:.0f} GB) {} in this machine's {:.0f} GB: the transformer "
              "{}",
              double(t) / 1e9, double(te) / 1e9, *keeps_transformer_ ? "fit together" : "do not fit together with room",
              double(mem) / 1e9, *keeps_transformer_ ? "stays loaded across prompts" : "is freed for each prompt and loaded again");
    return *keeps_transformer_;
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

std::vector<float> Engine::decode_audio(const Latents& l)
{
    return audio_vae().decode(std::span(l.audio).subspan(uint64_t(l.condition_audio_rows) * H3::kAudioChannels),
                              l.audio_latents);
}

Engine::Result Engine::generate(const Request& r)
{
    const Latents l = denoise(r);
    Result res;
    res.width = l.width;
    res.height = l.height;
    res.frames = l.frames;
    res.rgb.resize(uint64_t(l.frames) * l.height * l.width * 3);
    const auto t0 = Clock::now();
    if (r.stage)
        r.stage("decoding the video");
    decode_video(l, [&](uint32_t i, std::span<const float> rgb) {
        if (i >= l.frames)
            throw std::runtime_error("the video VAE decoded more frames than asked for");
        uint8_t* dst = &res.rgb[uint64_t(i) * l.height * l.width * 3];
        for (size_t k = 0; k < rgb.size(); ++k)
            dst[k] = uint8_t(std::lround(std::clamp(rgb[k], 0.0f, 1.0f) * 255.0f));
    });
    log::info("video decoded in {:.1f} s", seconds_since(t0));
    const auto t1 = Clock::now();
    if (r.stage)
        r.stage("decoding the audio");
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
    for (size_t i = 0; i < keyframes.size(); ++i) {
        const Image& k = keyframes[i];
        feats.push_back(vision().encode(k.rgb, k.height, k.width));
        const auto label = tok_->encode(std::format("<Picture {}>: ", i + 1), false);
        ids.insert(ids.end(), label.begin(), label.end());
        tg.insert(tg.end(), label.size(), uint8_t(H3::kTextTag));
        blocks.push_back(vision_block(ids, tg, feats.back(), "<|image_pad|>"));
    }
    auto out = encode_presentation(prompt, ids, tg, std::move(blocks), feats);
    if (tags)
        *tags = std::move(tg);
    return out;
}

models::QwenText::Vision Engine::vision_block(std::vector<int32_t>& ids, std::vector<uint8_t>& tg,
                                              const models::QwenVision::Features& f, const char* pad)
{
    auto special = [&](const char* name) {
        const auto v = tok_->encode(name, true);
        if (v.size() != 1)
            throw std::runtime_error(std::format("the tokenizer has no {} token", name));
        return v[0];
    };
    const uint32_t n = f.rows * f.cols;
    ids.push_back(special("<|vision_start|>"));
    models::QwenText::Vision b;
    b.first = uint32_t(ids.size());
    b.rows = f.rows;
    b.cols = f.cols;
    ids.insert(ids.end(), n, special(pad));
    ids.push_back(special("<|vision_end|>"));
    tg.insert(tg.end(), n + 2, uint8_t(H3::kVideoTag));
    return b;
}

std::vector<float> Engine::encode_presentation(const std::string& prompt, std::vector<int32_t>& ids,
                                               std::vector<uint8_t>& tg, std::vector<models::QwenText::Vision> blocks,
                                               const std::vector<models::QwenVision::Features>& feats)
{
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
    return out;
}

void Engine::sample_frames(uint32_t frames, std::vector<uint32_t>& indices, std::vector<double>& stamps)
{
    constexpr double kStride = double(kFps) / 2.0; // read at 2 fps
    indices.clear();
    stamps.clear();
    for (double cursor = 0.0; uint32_t(std::nearbyint(cursor)) < frames; cursor += kStride) {
        const uint32_t i = uint32_t(std::nearbyint(cursor));
        if (indices.empty() || i > indices.back())
            indices.push_back(i);
    }
    if (indices.size() < 2)
        throw std::runtime_error(std::format("a reference video must run at least {} frames at 24 fps ({:.2g} s)",
                                             uint32_t(kStride) + 1, (kStride + 1) / kFps));
    std::vector<double> ts;
    for (size_t i = 0; i < indices.size(); ++i)
        ts.push_back(double(i) / 2.0);
    if (ts.size() % 2)
        ts.push_back(ts.back());
    for (size_t i = 0; i < ts.size(); i += 2)
        stamps.push_back((ts[i] + ts[i + 1]) / 2.0);
}

std::pair<uint32_t, uint32_t> Engine::vision_video_size(uint32_t n, uint32_t h, uint32_t w)
{
    constexpr double kFactor = 32, kMin = 4096, kMax = 25165824;
    double hh = h, ww = w;
    if (hh < kFactor || ww < kFactor) {
        const double scale = std::max(kFactor / hh, kFactor / ww);
        hh = std::floor(hh * scale);
        ww = std::floor(ww * scale);
    }
    if (std::max(hh, ww) / std::min(hh, ww) > 200)
        throw std::runtime_error("a reference video's aspect ratio must be under 200");
    double hb = std::nearbyint(hh / kFactor) * kFactor, wb = std::nearbyint(ww / kFactor) * kFactor;
    const double tb = std::nearbyint(n / 2.0) * 2.0;
    if (tb * hb * wb > kMax) {
        const double beta = std::sqrt(n * hh * ww / kMax);
        hb = std::max(kFactor, std::floor(hh / beta / kFactor) * kFactor);
        wb = std::max(kFactor, std::floor(ww / beta / kFactor) * kFactor);
    } else if (tb * hb * wb < kMin) {
        const double beta = std::sqrt(kMin / (n * hh * ww));
        hb = std::ceil(hh * beta / kFactor) * kFactor;
        wb = std::ceil(ww * beta / kFactor) * kFactor;
    }
    return {uint32_t(hb), uint32_t(wb)};
}

Image Engine::normalize_reference_image(const Image& img, uint32_t short_edge)
{
    if (!img.width || !img.height || !short_edge)
        throw std::runtime_error("an empty reference picture");
    const double scale = double(short_edge) / std::min(img.width, img.height);
    auto side = [&](uint32_t v) { return std::max<uint32_t>(32, uint32_t(std::nearbyint(v * scale / 32.0)) * 32); };
    return resize_lanczos(img, side(img.width), side(img.height));
}

void Engine::normalize_reference_video(Reference& r, uint32_t frames)
{
    if (r.kind != Reference::Kind::video || !r.frames || !r.width || !r.height ||
        r.rgb.size() != uint64_t(r.frames) * r.height * r.width * 3)
        throw std::runtime_error("a reference video needs frames [frames][height][width][3]");
    if (!(r.fps > 0))
        throw std::runtime_error("a reference video needs a positive frame rate");
    // Every source frame's slots on the 24 fps grid (held until the next frame's), the frames kept up to `frames`.
    const double scale = double(kFps) / r.fps;
    const uint64_t end = r.fps == kFps ? r.frames : uint64_t(std::floor(r.frames * scale + 0.5));
    std::vector<uint32_t> source; // the source frame of every output frame
    for (uint32_t i = 0; i < r.frames && source.size() < frames; ++i) {
        const uint64_t s0 = r.fps == kFps ? i : uint64_t(std::floor(i * scale + 0.5));
        const uint64_t s1 = i + 1 < r.frames ? (r.fps == kFps ? i + 1 : uint64_t(std::floor((i + 1) * scale + 0.5))) : end;
        for (uint64_t s = s0; s < s1 && source.size() < frames; ++s)
            source.push_back(i);
    }
    if (source.empty())
        throw std::runtime_error("a reference video with no frames");
    const auto [cw, ch] = canvas(r.width, r.height);
    std::vector<uint8_t> out;
    out.reserve(uint64_t(source.size()) * ch * cw * 3);
    std::vector<uint8_t> last; // the source frame last put on the canvas, reused while it is held
    uint32_t last_src = UINT32_MAX;
    for (uint32_t src : source) {
        if (src != last_src) {
            const Image f{r.width, r.height, std::vector<uint8_t>(r.rgb.begin() + uint64_t(src) * r.height * r.width * 3,
                                                                 r.rgb.begin() + uint64_t(src + 1) * r.height * r.width * 3)};
            last = resize_lanczos(f, cw, ch).rgb;
            last_src = src;
        }
        out.insert(out.end(), last.begin(), last.end());
    }
    r.rgb = std::move(out);
    r.frames = uint32_t(source.size());
    r.width = cw;
    r.height = ch;
    r.fps = kFps;
}

std::vector<float> Engine::normalize_reference_audio(const Reference& r, uint32_t frames)
{
    if (!r.has_audio() || !r.sample_rate || (r.channels != 1 && r.channels != 2) ||
        r.audio.size() % r.channels)
        throw std::runtime_error("a reference soundtrack must be mono or stereo [channels][samples] at a sample rate");
    const uint64_t n = r.audio.size() / r.channels;
    const uint64_t keep = std::min<uint64_t>(n, uint64_t(double(frames) / kFps * r.sample_rate));
    std::vector<float> out;
    for (uint32_t ch = 0; ch < 2; ++ch) {
        const std::span<const float> src(&r.audio[uint64_t(std::min(ch, r.channels - 1)) * n], keep);
        const auto v = resample(src, r.sample_rate, kAudioRate);
        if (ch == 0)
            out.reserve(2 * v.size());
        out.insert(out.end(), v.begin(), v.end());
    }
    return out;
}

std::vector<float> Engine::reference_embeddings(const std::string& prompt, std::span<const Reference> refs,
                                                std::vector<uint8_t>* tags, std::vector<int32_t>* out_ids)
{
    std::vector<int32_t> ids;
    std::vector<uint8_t> tg;
    std::vector<models::QwenVision::Features> feats;
    std::vector<models::QwenText::Vision> blocks;
    auto text = [&](const std::string& t) {
        const auto v = tok_->encode(t, false);
        ids.insert(ids.end(), v.begin(), v.end());
        tg.insert(tg.end(), v.size(), uint8_t(H3::kTextTag));
    };
    uint32_t pictures = 0, videos = 0, audios = 0;
    for (const Reference& r : refs) {
        if (r.has_audio())
            text(std::format("<Audio {}>: ", ++audios));
        if (r.kind == Reference::Kind::image) {
            text(std::format("<Picture {}>: ", ++pictures));
            feats.push_back(vision().encode(r.image.rgb, r.image.height, r.image.width));
            blocks.push_back(vision_block(ids, tg, feats.back(), "<|image_pad|>"));
        } else if (r.kind == Reference::Kind::video) {
            text(std::format("<Video {}>: ", ++videos));
            std::vector<uint32_t> idx;
            std::vector<double> stamps;
            sample_frames(r.frames, idx, stamps);
            const auto [vh, vw] = vision_video_size(uint32_t(idx.size()), r.height, r.width);
            std::vector<Image> sampled;
            for (uint32_t i : idx) {
                Image f{r.width, r.height, std::vector<uint8_t>(r.rgb.begin() + uint64_t(i) * r.height * r.width * 3,
                                                               r.rgb.begin() + uint64_t(i + 1) * r.height * r.width * 3)};
                sampled.push_back(vh == r.height && vw == r.width ? std::move(f) : resize_bicubic(f, vw, vh));
            }
            for (size_t p = 0; p < stamps.size(); ++p) {
                const Image& a = sampled[2 * p];
                const Image& b = sampled[std::min(2 * p + 1, sampled.size() - 1)];
                text(std::format("<{:.1f} seconds>", stamps[p]));
                feats.push_back(vision().encode(a.rgb, b.rgb, vh, vw));
                blocks.push_back(vision_block(ids, tg, feats.back(), "<|video_pad|>"));
            }
        }
    }
    auto out = encode_presentation(prompt, ids, tg, std::move(blocks), feats);
    if (tags)
        *tags = std::move(tg);
    if (out_ids)
        *out_ids = std::move(ids);
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

    if (!r.references.empty() && !keyframes.empty())
        throw std::runtime_error("keyframes and references do not go together");
    for (size_t i = 0; i < keyframes.size(); ++i)
        keyframes[i] = prepare_keyframe(keyframes[i], L.width, L.height, i == 0);
    // The references as the model reads them: pictures at a 2048 short edge, videos at 24 fps on their own canvas
    // and no longer than the generation, soundtracks at 32 kHz.
    std::vector<Reference> refs;
    for (const Reference& src : r.references) {
        Reference ref;
        ref.kind = src.kind;
        if (src.kind == Reference::Kind::image) {
            ref.image = normalize_reference_image(src.image, opts_.reference_short_edge);
        } else if (src.kind == Reference::Kind::video) {
            ref = src;
            ref.audio.clear();
            normalize_reference_video(ref, L.frames);
        } else if (!src.has_audio()) {
            throw std::runtime_error("an audio reference without samples");
        }
        if (src.has_audio()) {
            ref.audio = normalize_reference_audio(src, L.frames);
            ref.channels = 2;
            ref.sample_rate = kAudioRate;
        }
        refs.push_back(std::move(ref));
    }
    auto stage = [&](const char* what) {
        if (r.stage)
            r.stage(what);
    };
    // The transformer makes room for the text encoder unless both fit.
    if (dit_ && !opts_.keep_text_encoder && !keeps_transformer()) {
        log::info("transformer freed for the text encoder");
        dit_.reset();
    }
    stage("encoding the prompt");
    std::vector<uint8_t> tags;
    const auto text = refs.empty() ? prompt_embeddings(r.prompt, keyframes, &tags) : reference_embeddings(r.prompt, refs, &tags);
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
    // The references' latents: a picture's or video's through the video VAE (noised to t = 0.999, their noise drawn
    // in order before the generated rows'), a soundtrack's through the audio VAE's encoder, clean.
    std::vector<H3::Layout::Reference> ref_layout;
    std::vector<float> cond_audio;
    if (!refs.empty())
        stage("encoding the references");
    if (!r.reference_noise.empty()) {
        size_t visual = 0;
        for (const Reference& ref : refs)
            visual += ref.kind != Reference::Kind::audio;
        if (r.reference_noise.size() != visual)
            throw std::runtime_error("reference noise: one per picture or video");
    }
    for (size_t i = 0, vi = 0; i < refs.size(); ++i) {
        const Reference& ref = refs[i];
        H3::Layout::Reference lr;
        if (ref.kind == Reference::Kind::image) {
            lr.kind = H3::Layout::Reference::Kind::image;
            lr.frames = 1;
            lr.h = ref.image.height / 16;
            lr.w = ref.image.width / 16;
        } else if (ref.kind == Reference::Kind::video) {
            lr.kind = H3::Layout::Reference::Kind::video;
            lr.h = ref.height / 16;
            lr.w = ref.width / 16;
        } else {
            lr.kind = H3::Layout::Reference::Kind::audio;
        }
        if (ref.kind != Reference::Kind::audio) {
            std::vector<float> z;
            uint32_t frames_enc = 1;
            if (ref.kind == Reference::Kind::image) {
                z = video_vae().encode(ref.image.rgb, ref.image.height, ref.image.width);
            } else {
                // Snapped down to 17 n + 5 frames (at least 22) so the VAE encodes whole clips; a shorter video is
                // encoded whole, its last clip padded.
                const int64_t F = ref.frames;
                const int64_t n = F >= int64_t(kClipLatents) ? (F - int64_t(kClipLatents)) / int64_t(kClipFrames) : 0;
                frames_enc = uint32_t(std::min<int64_t>(F, std::max<int64_t>(1, n) * kClipFrames + kClipLatents));
                z = video_vae().encode_video(std::span(ref.rgb).first(uint64_t(frames_enc) * ref.height * ref.width * 3),
                                             frames_enc, ref.height, ref.width);
            }
            const uint32_t lf = uint32_t(z.size() / (uint64_t(lr.h) * lr.w * VideoVae::kLatent));
            lr.frames = lf;
            std::vector<float> noise(z.size());
            if (!r.reference_noise.empty()) {
                if (r.reference_noise[vi].size() != z.size())
                    throw std::runtime_error("reference noise: [latent frames][latent_h][latent_w][24] of the reference");
                std::ranges::copy(r.reference_noise[vi], noise.begin());
            } else {
                for (float& v : noise)
                    v = nd(rng);
            }
            ++vi;
            const float t = 0.999f, u = float(1.0 - 0.999);
            const uint32_t lh = lr.h, lw = lr.w, ph_ = lh / 2, pw_ = lw / 2;
            for (uint32_t f = 0; f < lf; ++f)
                for (uint32_t py = 0; py < ph_; ++py)
                    for (uint32_t px = 0; px < pw_; ++px)
                        for (uint32_t c = 0; c < VideoVae::kLatent; ++c)
                            for (uint32_t dy = 0; dy < 2; ++dy)
                                for (uint32_t dx = 0; dx < 2; ++dx) {
                                    const uint64_t v = ((uint64_t(f) * lh + 2 * py + dy) * lw + 2 * px + dx) * VideoVae::kLatent + c;
                                    cond.push_back(t * z[v] + u * noise[v]);
                                }
        }
        if (ref.has_audio()) {
            const size_t half = ref.audio.size() / 2;
            const auto rows = audio_vae().encode(std::span(ref.audio).first(half), std::span(ref.audio).subspan(half));
            lr.audio = uint32_t(rows.size() / (2 * H3::kAudioChannels));
            cond_audio.insert(cond_audio.end(), rows.begin(), rows.end());
        }
        ref_layout.push_back(lr);
    }
    if (!dit_)
        stage("loading the transformer");
    H3& dit = transformer();
    H3::Layout layout;
    layout.text = uint32_t(text.size() / H3::kText);
    layout.frames = L.latent_frames;
    layout.h = L.latent_h;
    layout.w = L.latent_w;
    layout.audio = L.audio_latents;
    layout.cond_frames = uint32_t(keyframes.size());
    layout.cond_last = anchors;
    layout.refs = ref_layout;
    layout.text_tags = tags;
    const uint32_t V = layout.video_rows(), A = layout.audio_rows(), C = layout.cond_rows(), CA = layout.cond_audio_rows();
    L.condition_rows = C;
    L.condition_audio_rows = CA;
    L.video.resize(uint64_t(V) * H3::kVideoPatch);
    L.audio.resize(uint64_t(A) * H3::kAudioChannels);
    if (cond.size() != uint64_t(C) * H3::kVideoPatch || cond_audio.size() != uint64_t(CA) * H3::kAudioChannels)
        throw std::runtime_error("the conditioning rows do not match the layout");
    std::ranges::copy(cond, L.video.begin());
    std::ranges::copy(cond_audio, L.audio.begin());
    const std::span<float> gen(L.video.data() + uint64_t(C) * H3::kVideoPatch, uint64_t(V - C) * H3::kVideoPatch);
    const std::span<float> gen_audio(L.audio.data() + uint64_t(CA) * H3::kAudioChannels, uint64_t(A - CA) * H3::kAudioChannels);
    if (!r.noise_video.empty() || !r.noise_audio.empty()) {
        if (r.noise_video.size() != gen.size() || r.noise_audio.size() != gen_audio.size())
            throw std::runtime_error(std::format("noise: video rows [{}][96] and audio rows [{}][32]", V - C, A - CA));
        std::ranges::copy(r.noise_video, gen.begin());
        std::ranges::copy(r.noise_audio, gen_audio.begin());
    } else {
        for (float& v : gen)
            v = nd(rng);
        for (float& v : gen_audio)
            v = nd(rng);
    }
    (void)rpf;
    stage("denoising");
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
        euler(gen_audio, std::span(vel.audio).subspan(uint64_t(CA) * H3::kAudioChannels), ta, sig_a[i], sig_a[i + 1]);
        if (r.progress)
            r.progress(i + 1, n);
    }
    return L;
}

} // namespace mako::video
