// MiniMax-H3 against the Diffusers implementation's tensors (tools/dump_h3_ref.py: the text encoder in fp32, the
// transformer with its BF16 weights computed in fp32, 256x256, 124 frames, 2 steps); an optional fourth argument
// is the transformer's int8 matrices (none, all, or some of qkv, out, fc1, fc2) to see what quantization costs:
//   tokens    the prompt's and other strings' token ids
//   schedule  the two schedules' sigmas and timesteps
//   text      the text encoder's 50th layer output
//   dit       the transformer on the reference's text and latents: the stages of both steps and their velocities
//   latents   the denoising loop from the reference's noise to its final latents
//   h3_ref_test <model dir> <reference dir> <mode>
#include "video/engine.hpp"
#include "mtl/context.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <print>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

int failures = 0;

void check(bool ok, const std::string& what)
{
    std::println("  {:4} {}", ok ? "ok" : "FAIL", what);
    failures += !ok;
}

std::string read_file(const fs::path& p)
{
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::vector<float> read_f32(const fs::path& p)
{
    const std::string s = read_file(p);
    if (s.empty())
        throw std::runtime_error("cannot read " + p.string());
    std::vector<float> v(s.size() / 4);
    std::memcpy(v.data(), s.data(), v.size() * 4);
    return v;
}

// Relative RMS error over everything, and the lowest cosine similarity of a row of `width`.
struct Diff {
    double rel = 0, min_cos = 1;
};
Diff diff(std::span<const float> got, std::span<const float> want, size_t width)
{
    Diff d;
    double e = 0, r = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        e += (double(got[i]) - want[i]) * (double(got[i]) - want[i]);
        r += double(want[i]) * want[i];
    }
    d.rel = std::sqrt(e / r);
    for (size_t row = 0; row < want.size() / width; ++row) {
        double gw = 0, gg = 0, ww = 0;
        for (size_t i = row * width; i < (row + 1) * width; ++i) {
            gw += double(got[i]) * want[i];
            gg += double(got[i]) * got[i];
            ww += double(want[i]) * want[i];
        }
        d.min_cos = std::min(d.min_cos, ww > 0 && gg > 0 ? gw / std::sqrt(gg * ww) : 1.0);
    }
    return d;
}

void compare(const std::string& what, std::span<const float> got, const std::vector<float>& want, size_t width,
             double tol)
{
    if (got.size() != want.size()) {
        check(false, std::format("{}: {} values, the reference {}", what, got.size(), want.size()));
        return;
    }
    const Diff d = diff(got, want, width);
    check(d.rel < tol, std::format("{}: {} rows, rel error {:.2e}, lowest row cosine {:.6f}", what, want.size() / width,
                                   d.rel, d.min_cos));
}

std::vector<float> floats_of(const glz::generic& a)
{
    std::vector<float> v;
    for (const auto& x : a.get_array())
        v.push_back(float(x.get_number()));
    return v;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4 || !fs::exists(fs::path(argv[1]) / "model_index.json") || !fs::exists(fs::path(argv[2]) / "manifest.json")) {
        std::println("model {} or reference {} not found: skipped", argc > 1 ? argv[1] : "", argc > 2 ? argv[2] : "");
        return 77;
    }
    const fs::path model = argv[1], ref = argv[2];
    const std::string mode = argv[3];
    glz::generic m;
    if (glz::read_json(m, read_file(ref / "manifest.json"))) {
        std::println(stderr, "bad manifest");
        return 1;
    }
    std::println("{}", mode);
    using mako::video::Engine;
    using mako::video::H3;
    const std::string prompt = m["prompt"].get_string();
    std::vector<int32_t> ids;
    for (const auto& v : m["ids"].get_array())
        ids.push_back(int32_t(v.get_number()));
    const uint32_t steps = uint32_t(m["steps"].get_number()), size = uint32_t(m["size"].get_number());
    const uint32_t frames = uint32_t(m["num_frames"].get_number());
    mako::mtl::Context ctx;
    mako::video::Options opts;
    if (argc > 4) { // the transformer's int8 matrices (H3::parse_int8), to see what the quantization costs
        const auto m = H3::parse_int8(argv[4]);
        if (!m) {
            std::println(stderr, "int8: none, all, or a comma list of qkv, out, fc1, fc2");
            return 2;
        }
        opts.int8 = *m;
    }
    Engine engine(model.string(), ctx, opts);

    if (mode == "tokens") {
        const auto got = engine.tokenizer().encode(prompt, true);
        check(got == ids, std::format("the prompt's {} token ids (the reference's {})", got.size(), ids.size()));
        for (const auto& c : m["token_cases"].get_array()) {
            std::vector<int32_t> want;
            for (const auto& v : c["ids"].get_array())
                want.push_back(int32_t(v.get_number()));
            const std::string text = c["text"].get_string();
            const auto g = engine.tokenizer().encode(text, true);
            check(g == want, std::format("\"{}\": {} ids", text.substr(0, 40), g.size()));
        }
    } else if (mode == "schedule") {
        const auto vs = floats_of(m["video_sigmas"]), as = floats_of(m["audio_sigmas"]);
        const auto vt = floats_of(m["video_timesteps"]), at = floats_of(m["audio_timesteps"]);
        const auto sv = Engine::sigmas(steps, Engine::kVideoShift), sa = Engine::sigmas(steps, Engine::kAudioShift);
        check(sv == vs, std::format("video sigmas of {} points", steps));
        check(sa == as, std::format("audio sigmas of {} points", steps));
        bool ok = sv.size() == vt.size() + 1 && sa.size() == at.size() + 1;
        for (size_t i = 0; ok && i < vt.size(); ++i)
            ok = 1.0f - sv[i] == vt[i] && 1.0f - sa[i] == at[i];
        check(ok, "timesteps 1 - sigma");
        const auto s50 = Engine::sigmas(50, 12.0f);
        check(s50.size() == 50, std::format("50 points keep {} after the shift", s50.size()));
    } else if (mode == "text") {
        const auto got = engine.prompt_embeddings(prompt);
        const auto want = read_f32(ref / "text_embeds.f32");
        compare("layer 50", got, want, H3::kText, 0.03);
    } else if (mode == "dit" || mode == "latents") {
        const uint32_t lat_f = uint32_t(m["latent_frames"].get_number()), lat_h = uint32_t(m["latent_height"].get_number()),
                       lat_w = uint32_t(m["latent_width"].get_number()), n_audio = uint32_t(m["num_audio_latents"].get_number());
        const auto text = read_f32(ref / "text_embeds.f32");
        const uint32_t T = uint32_t(text.size() / H3::kText);
        H3::Layout layout{.text = T, .frames = lat_f, .h = lat_h, .w = lat_w, .audio = n_audio};
        const uint32_t S = layout.rows();
        check(S == uint32_t(m["seq_len"].get_number()), std::format("{} rows", S));
        const auto vt = floats_of(m["video_timesteps"]), at = floats_of(m["audio_timesteps"]);
        if (mode == "latents") {
            Engine::Request r;
            r.prompt = prompt;
            r.width = r.height = size;
            r.frames = frames;
            r.steps = steps;
            const auto nv = read_f32(ref / "noise_video_rows.f32"), na = read_f32(ref / "noise_audio_rows.f32");
            r.noise_video = nv;
            r.noise_audio = na;
            const auto L = engine.denoise(r);
            // The last step's x0 = x + sigma v cancels most of x, which amplifies the velocities' error (about 1e-2).
            compare("final video rows", L.video, read_f32(ref / "final_video_rows.f32"), H3::kVideoPatch, 0.06);
            compare("final audio rows", L.audio, read_f32(ref / "final_audio_rows.f32"), H3::kAudioChannels, 0.03);
            return failures ? 1 : 0;
        }
        H3& dit = engine.transformer();
        std::map<std::string, std::vector<float>> probes;
        dit.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            probes[layer < 0 ? name : std::format("{}{}", name, layer)] = std::vector<float>(data, data + n);
        });
        dit.begin(text, layout);
        compare("text after the refiner", probes["txt"], read_f32(ref / "s0_txt.f32"), H3::kDim, 5e-3);
        for (uint32_t i = 0; i < vt.size(); ++i) {
            probes.clear();
            const auto vin = read_f32(ref / std::format("step{}_video_rows.f32", i));
            const auto ain = read_f32(ref / std::format("step{}_audio_rows.f32", i));
            const auto vel = dit.step(vin, ain, vt[i], at[i]);
            std::println(" step {} (t video {}, audio {})", i, vt[i], at[i]);
            const auto temb = read_f32(ref / std::format("s{}_temb.f32", i));
            const uint32_t U = uint32_t(temb.size() / H3::kTimeDim);
            compare("time embedding", probes["temb"], temb, H3::kTimeDim, 1e-4);
            // The reference's modulation: [6][U * 3][5376] chunks; mako's the projection's rows [U][3][6][5376].
            {
                const auto want = read_f32(ref / std::format("s{}_mod0.f32", i));
                const auto& got = probes["mod0"];
                std::vector<float> g(want.size());
                const uint32_t D = H3::kDim;
                if (got.size() == uint64_t(U) * 18 * D)
                    for (uint32_t c = 0; c < 6; ++c)
                        for (uint32_t u = 0; u < U; ++u)
                            for (uint32_t md = 0; md < 3; ++md)
                                std::copy_n(got.begin() + (uint64_t(u) * 18 + md * 6 + c) * D, D,
                                            g.begin() + (uint64_t(c) * U * 3 + u * 3 + md) * D);
                compare("block 0 modulation", g, want, D, 5e-3);
            }
            compare("packed input", probes["x0"], read_f32(ref / std::format("s{}_x0.f32", i)), H3::kDim, 5e-3);
            for (int b : {0, 1, 24, 49})
                compare(std::format("block {}", b), probes[std::format("block{}", b)],
                        read_f32(ref / std::format("s{}_block{}.f32", i, b)), H3::kDim, b < 40 ? 1e-2 : 4e-2);
            compare("final norm", probes["norm_out"], read_f32(ref / std::format("s{}_norm_out.f32", i)), H3::kDim, 6e-2);
            compare("video velocity", vel.video, read_f32(ref / std::format("step{}_pred_video.f32", i)), H3::kVideoPatch, 2e-2);
            compare("audio velocity", vel.audio, read_f32(ref / std::format("step{}_pred_audio.f32", i)), H3::kAudioChannels, 2e-2);
        }
    } else if (mode == "encode") {
        // The keyframes (already on the canvas) through the VAE encoder: the posterior's mean and standard deviation
        // against the fp32 reference, then the normalized sample with the reference's noise, and with the torch
        // generator reproduced.
        auto& vae = engine.video_vae();
        auto hwc = [&](const std::vector<float>& chw, uint32_t h, uint32_t w) { // [24][h][w] -> [h][w][24]
            std::vector<float> o(chw.size());
            for (uint32_t c = 0; c < 24; ++c)
                for (uint32_t y = 0; y < h; ++y)
                    for (uint32_t x = 0; x < w; ++x)
                        o[(uint64_t(y) * w + x) * 24 + c] = chw[(uint64_t(c) * h + y) * w + x];
            return o;
        };
        for (int i = 0; i < 2; ++i) {
            const auto rgb = read_file(ref / std::format("fl_keyframe{}.u8", i));
            const uint32_t h = size, w = size;
            check(rgb.size() == uint64_t(h) * w * 3, std::format("keyframe {}: {}x{} pixels", i, w, h));
            std::vector<float> mean, sd;
            vae.encode_moments(std::span(reinterpret_cast<const uint8_t*>(rgb.data()), rgb.size()), h, w, mean, sd);
            compare(std::format("keyframe {} posterior mean", i), mean, hwc(read_f32(ref / std::format("fl_cond_mean{}.f32", i)), h / 16, w / 16), 24, 2e-2);
            compare(std::format("keyframe {} posterior sd", i), sd, hwc(read_f32(ref / std::format("fl_cond_std{}.f32", i)), h / 16, w / 16), 24, 2e-2);
            const auto noise = hwc(read_f32(ref / std::format("fl_cond_sample_noise{}.f32", i)), h / 16, w / 16);
            const auto want = hwc(read_f32(ref / std::format("fl_cond_latents{}.f32", i)), h / 16, w / 16);
            const auto got = vae.encode(std::span(reinterpret_cast<const uint8_t*>(rgb.data()), rgb.size()), h, w, noise);
            compare(std::format("keyframe {} latents (reference noise)", i), got, want, 24, 2e-2);
            const auto own = vae.encode(std::span(reinterpret_cast<const uint8_t*>(rgb.data()), rgb.size()), h, w);
            compare(std::format("keyframe {} latents (torch generator reproduced)", i), own, want, 24, 2e-2);
        }
    } else if (mode == "vae" || mode == "audio") {
        Engine::Latents L;
        L.width = L.height = size;
        L.frames = frames;
        L.latent_frames = uint32_t(m["latent_frames"].get_number());
        L.latent_h = uint32_t(m["latent_height"].get_number());
        L.latent_w = uint32_t(m["latent_width"].get_number());
        L.audio_latents = uint32_t(m["num_audio_latents"].get_number());
        L.video = read_f32(ref / "final_video_rows.f32");
        L.audio = read_f32(ref / "final_audio_rows.f32");
        if (mode == "vae") {
            const auto want = read_f32(ref / "video.f32");
            std::vector<float> got;
            uint32_t n = 0;
            engine.decode_video(L, [&](uint32_t i, std::span<const float> rgb) {
                check(i == n, std::format("frame {} in order", i));
                ++n;
                got.insert(got.end(), rgb.begin(), rgb.end());
            });
            check(n == frames, std::format("{} frames decoded", n));
            compare("video frames", got, want, uint64_t(size) * size * 3, 1e-2);
            // The worst pixel, in 8-bit steps.
            float worst = 0;
            for (size_t i = 0; i < std::min(got.size(), want.size()); ++i)
                worst = std::max(worst, std::fabs(got[i] - want[i]));
            check(worst * 255.0f < 4.0f, std::format("worst pixel {:.2f} of 255 off", worst * 255.0f));
        } else {
            const auto want = read_f32(ref / "audio.f32");
            const auto got = engine.decode_audio(L);
            compare("audio samples", got, want, want.size() / 2, 1e-3);
        }
    } else {
        std::println(stderr, "unknown mode {}", mode);
        return 2;
    }
    return failures ? 1 : 0;
}
