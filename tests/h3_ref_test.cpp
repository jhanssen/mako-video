// MiniMax-H3 against the Diffusers implementation's tensors (tools/dump_h3_ref.py: the text encoder in fp32, the
// transformer with its BF16 weights computed in fp32, 256x256, 124 frames, 2 steps); an optional fourth argument
// is the transformer's int8 matrices (none, all, or some of qkv, out, fc1, fc2) to see what quantization costs, an
// optional fifth the sparse attention's share of cubes (1 selects every cube and must match the dense model):
//   tokens    the prompt's and other strings' token ids
//   schedule  the two schedules' sigmas and timesteps
//   text      the text encoder's 50th layer output
//   dit       the transformer on the reference's text and latents: the stages of both steps and their velocities
//   latents   the denoising loop from the reference's noise to its final latents
//   encode, vision, resize, fl2va, fl2va_latents  the fl2va stages (keyframes)
//   ref_audio, ref_image, ref_video  the audio and video VAEs encoding the ref2va references
//   ports      torchvision's bicubic and torchaudio's resampler ports against tools/check_ports.py's outputs
//   ref_layout the ref2va sequence's rotary positions and tags
//   ref_vision the vision tower on the ref2va picture and the video's frame pair
//   ref_text   the ref2va presentation and the text encoder over it
//   ref_dit    the Ref2VA transformer's two steps on the reference's rows
//   ref2va     the engine end to end from the raw references
//   h3_ref_test <model dir> <reference dir> <mode>
#include "models/qwen_vision.hpp"
#include "video/audio.hpp"
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
#include <tuple>
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
    if (argc > 5) // the sparse attention's share of cubes (1: every cube, which must give the dense result)
        opts.sparse = std::stof(argv[5]);
    if (argc > 6) // the cubes kept regardless of score (0, 1 or 3)
        opts.sparse_local = uint32_t(std::stoul(argv[6]));
    const bool ref2va = mode.starts_with("ref_") || mode == "ref2va";
    if (ref2va) { // the references' transformer, beside the model's directory, at the dump's picture resolution
        const fs::path t = model.parent_path() / "Ref2VA" / "transformer";
        if ((mode == "ref_dit" || mode == "ref2va") && !fs::exists(t / "config.json")) {
            std::println("the Ref2VA transformer {} not found: skipped", t.string());
            return 77;
        }
        opts.transformer = t.string();
        opts.reference_short_edge = uint32_t(m["rf_ref_short_edge"].get_number());
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
    } else if (mode == "resize") {
        // The keyframes onto the canvas: the first stretched, the second cover-cropped, against PIL's Lanczos.
        for (int i = 0; i < 2; ++i) {
            const auto src = read_file(ref / (i == 0 ? "fl_first.u8" : "fl_last.u8"));
            const auto& sz = m[i == 0 ? "fl_first_size" : "fl_last_size"].get_array();
            mako::video::Image img{uint32_t(sz[0].get_number()), uint32_t(sz[1].get_number()),
                              std::vector<uint8_t>(src.begin(), src.end())};
            const auto got = Engine::prepare_keyframe(img, size, size, i == 0);
            const auto want = read_file(ref / std::format("fl_keyframe{}.u8", i));
            check(got.rgb.size() == want.size(), std::format("keyframe {}: {} bytes", i, got.rgb.size()));
            double sum = 0;
            int worst = 0;
            for (size_t k = 0; k < want.size() && k < got.rgb.size(); ++k) {
                const int d = std::abs(int(got.rgb[k]) - int(uint8_t(want[k])));
                sum += d;
                worst = std::max(worst, d);
            }
            const double mean = sum / double(want.size());
            check(mean < 0.5 && worst <= 8, std::format("keyframe {} ({}): mean |diff| {:.3f} of 255, worst {}", i,
                                                         i == 0 ? "stretched" : "cover-cropped", mean, worst));
        }
    } else if (mode == "fl2va" || mode == "fl2va_latents") {
        // The transformer with two keyframes' rows in the sequence: from the reference's text embeddings, tags and
        // per-step rows (fl2va), or the engine end to end from the keyframes (fl2va_latents).
        const uint32_t lat_f = uint32_t(m["latent_frames"].get_number()), lat_h = uint32_t(m["latent_height"].get_number()),
                       lat_w = uint32_t(m["latent_width"].get_number()), n_audio = uint32_t(m["num_audio_latents"].get_number());
        const uint32_t C = uint32_t(m["fl_condition_rows"].get_number());
        const auto vt = floats_of(m["video_timesteps"]), at = floats_of(m["audio_timesteps"]);
        if (mode == "fl2va_latents") {
            Engine::Request r;
            r.prompt = prompt;
            r.width = r.height = size;
            r.frames = frames;
            r.steps = steps;
            const auto nv = read_f32(ref / "fl_noise_video_rows.f32"), na = read_f32(ref / "fl_noise_audio_rows.f32");
            r.noise_video = nv;
            r.noise_audio = na;
            std::vector<std::vector<float>> kn;
            auto hwc = [&](const std::vector<float>& chw) { // [24][h][w] -> [h][w][24]
                std::vector<float> o(chw.size());
                for (uint32_t c = 0; c < 24; ++c)
                    for (uint32_t y = 0; y < lat_h; ++y)
                        for (uint32_t x = 0; x < lat_w; ++x)
                            o[(uint64_t(y) * lat_w + x) * 24 + c] = chw[(uint64_t(c) * lat_h + y) * lat_w + x];
                return o;
            };
            for (int i = 0; i < 2; ++i) {
                const auto rgb = read_file(ref / std::format("fl_keyframe{}.u8", i));
                mako::video::Image img{size, size, std::vector<uint8_t>(rgb.begin(), rgb.end())};
                (i == 0 ? r.first : r.last) = std::move(img);
                kn.push_back(hwc(read_f32(ref / std::format("fl_cond_noise{}.f32", i))));
            }
            for (const auto& k : kn)
                r.keyframe_noise.emplace_back(k);
            const auto L = engine.denoise(r);
            check(L.condition_rows == C, std::format("{} conditioning rows", L.condition_rows));
            const auto want = read_f32(ref / "fl_final_video_rows.f32");
            compare("keyframe rows", std::span(L.video).first(uint64_t(C) * H3::kVideoPatch),
                    std::vector<float>(want.begin(), want.begin() + uint64_t(C) * H3::kVideoPatch), H3::kVideoPatch, 1e-2);
            compare("final video rows", std::span(L.video).subspan(uint64_t(C) * H3::kVideoPatch),
                    std::vector<float>(want.begin() + uint64_t(C) * H3::kVideoPatch, want.end()), H3::kVideoPatch, 0.06);
            compare("final audio rows", L.audio, read_f32(ref / "fl_final_audio_rows.f32"), H3::kAudioChannels, 0.03);
            return failures ? 1 : 0;
        }
        const auto text = read_f32(ref / "fl_text_embeds.f32");
        const uint32_t T = uint32_t(text.size() / H3::kText);
        H3::Layout layout{.text = T, .frames = lat_f, .h = lat_h, .w = lat_w, .audio = n_audio, .cond_frames = 2,
                          .cond_last = {0, 1}};
        for (const auto& v : m["fl_tags"].get_array())
            layout.text_tags.push_back(uint8_t(v.get_number()));
        const uint32_t S = layout.rows();
        check(S == uint32_t(m["fl_seq_len"].get_number()) && layout.cond_rows() == C, std::format("{} rows, {} conditioning", S, C));
        H3& dit = engine.transformer();
        std::map<std::string, std::vector<float>> probes;
        dit.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            probes[layer < 0 ? name : std::format("{}{}", name, layer)] = std::vector<float>(data, data + n);
        });
        dit.begin(text, layout);
        compare("text after the refiner", probes["txt"], read_f32(ref / "fl_s0_txt.f32"), H3::kDim, 5e-3);
        for (uint32_t i = 0; i < vt.size(); ++i) {
            probes.clear();
            const auto vin = read_f32(ref / std::format("fl_step{}_video_rows.f32", i));
            const auto ain = read_f32(ref / std::format("fl_step{}_audio_rows.f32", i));
            const auto vel = dit.step(vin, ain, vt[i], at[i]);
            std::println(" step {} (t video {}, audio {})", i, vt[i], at[i]);
            compare("packed input", probes["x0"], read_f32(ref / std::format("fl_s{}_x0.f32", i)), H3::kDim, 5e-3);
            for (int b : {0, 1, 24, 49})
                compare(std::format("block {}", b), probes[std::format("block{}", b)],
                        read_f32(ref / std::format("fl_s{}_block{}.f32", i, b)), H3::kDim, b < 40 ? 1e-2 : 4e-2);
            compare("final norm", probes["norm_out"], read_f32(ref / std::format("fl_s{}_norm_out.f32", i)), H3::kDim, 6e-2);
            const auto pv = read_f32(ref / std::format("fl_step{}_pred_video.f32", i));
            compare("video velocity (generated rows)", std::span(vel.video).subspan(uint64_t(C) * H3::kVideoPatch),
                    std::vector<float>(pv.begin() + uint64_t(C) * H3::kVideoPatch, pv.end()), H3::kVideoPatch, 2e-2);
            compare("audio velocity", vel.audio, read_f32(ref / std::format("fl_step{}_pred_audio.f32", i)), H3::kAudioChannels, 2e-2);
        }
    } else if (mode == "vision") {
        // The keyframe presentation: the vision tower's merged tokens and DeepStack features for the first keyframe
        // against the fp32 reference, then the text encoder's 50th layer over the whole presentation (labels, vision
        // blocks, prompt) with both keyframes' features in it.
        mako::models::QwenVision vis(model / "text_encoder", ctx);
        std::map<std::string, std::vector<float>> vp;
        vis.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            vp[layer < 0 ? name : std::format("{}{}", name, layer)] = std::vector<float>(data, data + n);
        });
        std::vector<mako::models::QwenVision::Features> feats;
        for (int i = 0; i < 2; ++i) {
            const auto rgb = read_file(ref / std::format("fl_keyframe{}.u8", i));
            feats.push_back(vis.encode(std::span(reinterpret_cast<const uint8_t*>(rgb.data()), rgb.size()), size, size));
            if (i == 0) { // the stages of the first keyframe: the reference holds both keyframes' patches
                const uint32_t np = (size / 16) * (size / 16);
                auto head = [&](const std::string& f) {
                    const auto v = read_f32(ref / f);
                    return std::vector<float>(v.begin(), v.begin() + uint64_t(np) * 1152);
                };
                compare("keyframe 0 patch embedding", vp["patch"], head("fl_vis_patch_embed.f32"), 1152, 1e-2);
                compare("keyframe 0 block input", vp["in"], head("fl_vis_block_in.f32"), 1152, 1e-2);
                compare("keyframe 0 block 0", vp["block0"], head("fl_vis_block0.f32"), 1152, 1e-2);
                compare("keyframe 0 block 26", vp["block26"], head("fl_vis_block_last.f32"), 1152, 2e-2);
            }
        }
        const uint32_t n0 = feats[0].rows * feats[0].cols;
        auto first_n = [&](const std::vector<float>& v) { return std::vector<float>(v.begin(), v.begin() + uint64_t(n0) * 5120); };
        compare("keyframe 0 vision tokens", feats[0].embeds, first_n(read_f32(ref / "fl_vis_merger.f32")), 5120, 2e-2);
        for (int j = 0; j < 3; ++j)
            compare(std::format("keyframe 0 DeepStack {}", j), feats[0].deepstack[j],
                    first_n(read_f32(ref / std::format("fl_vis_deepstack{}.f32", j))), 5120, 2e-2);
        std::vector<int32_t> ids;
        for (const auto& v : m["fl_ids"].get_array())
            ids.push_back(int32_t(v.get_number()));
        std::vector<mako::models::QwenText::Vision> blocks;
        for (size_t i = 0, t = 0; i < feats.size(); ++i) {
            while (t < ids.size() && ids[t] != 151655) // <|image_pad|>
                ++t;
            mako::models::QwenText::Vision b;
            b.first = uint32_t(t);
            b.rows = feats[i].rows;
            b.cols = feats[i].cols;
            b.embeds = feats[i].embeds;
            for (const auto& d : feats[i].deepstack)
                b.deepstack.emplace_back(d);
            blocks.push_back(std::move(b));
            t += uint64_t(b.rows) * b.cols;
        }
        const uint32_t taps[1] = {Engine::kTextLayer};
        const auto got = engine.text_encoder().encode(ids, 0, taps, false, blocks);
        const auto want = read_f32(ref / "fl_text_embeds.f32");
        compare("layer 50 with the keyframes", got, want, H3::kText, 0.03);
        // The rows farthest off, by cosine, with their norms (an outlier row says which part of the presentation).
        std::vector<std::tuple<double, size_t, double, double>> rows;
        for (size_t r = 0; r < ids.size(); ++r) {
            double dot = 0, ng = 0, nw = 0;
            for (uint32_t c = 0; c < H3::kText; ++c) {
                const double g = got[r * H3::kText + c], w = want[r * H3::kText + c];
                dot += g * w;
                ng += g * g;
                nw += w * w;
            }
            rows.emplace_back(dot / std::sqrt(ng * nw + 1e-30), r, std::sqrt(ng), std::sqrt(nw));
        }
        std::ranges::sort(rows);
        for (size_t i = 0; i < 4 && i < rows.size(); ++i)
            std::println("       row {} (id {}): cosine {:.4f}, norm {:.1f} against {:.1f}", std::get<1>(rows[i]),
                         ids[std::get<1>(rows[i])], std::get<0>(rows[i]), std::get<2>(rows[i]), std::get<3>(rows[i]));
    } else if (mode == "ports") {
        // The ports of torchvision's antialiased bicubic on 8-bit frames and torchaudio's resampler, against outputs
        // tools/check_ports.py wrote (in `ports` under the reference directory), and the timestamp labels' rounding.
        const fs::path d = ref / "ports";
        if (!fs::exists(d / "bc_in.u8")) {
            std::println("{} not found: skipped", d.string());
            return 77;
        }
        const auto in = read_file(d / "bc_in.u8");
        const mako::video::Image src{1344, 768, std::vector<uint8_t>(in.begin(), in.end())};
        for (const auto& [name, w, h] : {std::tuple{"down", 1184u, 672u}, std::tuple{"up", 1408u, 800u}, std::tuple{"mixed", 1344u, 704u}}) {
            const auto got = mako::video::resize_bicubic(src, w, h);
            const auto want = read_file(d / std::format("bc_{}.u8", name));
            check(got.rgb.size() == want.size(), std::format("bicubic {}: {}x{}", name, w, h));
            size_t off1 = 0, off2 = 0;
            int worst = 0;
            for (size_t i = 0; i < std::min(got.rgb.size(), want.size()); ++i) {
                const int dlt = std::abs(int(got.rgb[i]) - int(uint8_t(want[i])));
                worst = std::max(worst, dlt);
                off1 += dlt == 1;
                off2 += dlt > 1;
            }
            check(worst <= 1, std::format("bicubic {}: worst {} of 255 off, {:.3f}% of values 1 off, {} more than 1 off",
                                          name, worst, 100.0 * off1 / want.size(), off2));
        }
        for (uint32_t rate : {48000u, 44100u}) {
            const auto wave = read_f32(d / std::format("rs_{}_in.f32", rate)), want = read_f32(d / std::format("rs_{}_out.f32", rate));
            const size_t n = wave.size() / 2, m_ = want.size() / 2;
            std::vector<float> got;
            for (int ch = 0; ch < 2; ++ch) {
                const auto y = mako::video::resample(std::span(wave).subspan(ch * n, n), rate, 32000);
                got.insert(got.end(), y.begin(), y.end());
            }
            compare(std::format("resample {} -> 32000 Hz", rate), got, want, m_, 1e-4);
        }
        const std::pair<double, const char*> stamps[] = {{0.25, "<0.2 seconds>"}, {0.75, "<0.8 seconds>"}, {1.25, "<1.2 seconds>"},
                                                         {2.5, "<2.5 seconds>"}, {0.0, "<0.0 seconds>"}, {10.25, "<10.2 seconds>"}};
        for (const auto& [t, want] : stamps)
            check(std::format("<{:.1f} seconds>", t) == want, std::format("timestamp label of {} is {}", t, want));
    } else if (mode == "ref_layout" || mode == "ref_dit" || mode == "ref_text" || mode == "ref2va" || mode == "ref_vision") {
        // The ref2va layout of [text | picture | soundtrack | video's soundtrack, video | audio | video] against the
        // reference's positions and tags (ref_layout), the vision tower on the picture and the video's frame pair
        // (ref_vision), the presentation and the text encoder (ref_text), the Ref2VA transformer's two steps on the
        // reference's rows (ref_dit), the engine end to end from the raw references (ref2va).
        const uint32_t lat_f = uint32_t(m["latent_frames"].get_number()), lat_h = uint32_t(m["latent_height"].get_number()),
                       lat_w = uint32_t(m["latent_width"].get_number()), n_audio = uint32_t(m["num_audio_latents"].get_number());
        const auto& isz = m["rf_image_norm_size"].get_array();
        const auto& vsh = m["rf_video_norm_shape"].get_array();
        const uint32_t iw = uint32_t(isz[0].get_number()), ih = uint32_t(isz[1].get_number());
        const uint32_t vf = uint32_t(vsh[0].get_number()), vh = uint32_t(vsh[1].get_number()), vw = uint32_t(vsh[2].get_number());
        const auto a1 = read_f32(ref / "rf_audio1_rows.f32"), a2 = read_f32(ref / "rf_audio2_rows.f32");
        const auto cond_video = read_f32(ref / "rf_cond_video_latents.f32"); // [1][24][F'][lh][lw]
        const uint32_t vlf = uint32_t(cond_video.size() / (24ull * (vh / 16) * (vw / 16)));
        const uint32_t C = uint32_t(m["rf_condition_rows"].get_number()), CA = uint32_t(m["rf_condition_audio_rows"].get_number());
        const uint32_t S = uint32_t(m["rf_seq_len"].get_number()), T = uint32_t(m["rf_text_rows"].get_number());
        std::vector<int32_t> ids;
        for (const auto& v : m["rf_ids"].get_array())
            ids.push_back(int32_t(v.get_number()));
        std::vector<uint8_t> text_tags;
        for (const auto& v : m["rf_tags"].get_array())
            text_tags.push_back(uint8_t(v.get_number()));
        using LR = H3::Layout::Reference;
        H3::Layout layout{.text = T, .frames = lat_f, .h = lat_h, .w = lat_w, .audio = n_audio};
        layout.refs = {LR{.kind = LR::Kind::image, .frames = 1, .h = ih / 16, .w = iw / 16},
                       LR{.kind = LR::Kind::audio, .audio = uint32_t(a1.size() / 64)},
                       LR{.kind = LR::Kind::video, .frames = vlf, .h = vh / 16, .w = vw / 16, .audio = uint32_t(a2.size() / 64)}};
        layout.text_tags = text_tags;
        check(layout.rows() == S && layout.cond_rows() == C && layout.cond_audio_rows() == CA,
              std::format("{} rows, {} reference video rows, {} reference audio rows", layout.rows(), layout.cond_rows(), layout.cond_audio_rows()));
        if (mode == "ref_layout") {
            const auto R = H3::rows(layout);
            const auto raw = read_file(ref / "rf_position_ids.f64");
            std::vector<double> want(raw.size() / 8);
            std::memcpy(want.data(), raw.data(), raw.size());
            check(R.pos.size() == want.size(), std::format("{} positions", R.pos.size() / 3));
            double worst = 0;
            size_t at = 0;
            for (size_t i = 0; i < std::min(R.pos.size(), want.size()); ++i)
                if (std::abs(R.pos[i] - want[i]) > worst) {
                    worst = std::abs(R.pos[i] - want[i]);
                    at = i;
                }
            check(worst < 1e-9, std::format("positions: largest difference {:.3e} at row {} axis {}", worst, at / 3, at % 3));
            size_t bad = 0;
            const auto& tg = m["rf_token_tags"].get_array();
            check(tg.size() == R.tags.size(), std::format("{} tags", R.tags.size()));
            for (size_t i = 0; i < std::min(tg.size(), R.tags.size()); ++i)
                bad += uint8_t(tg[i].get_number()) != R.tags[i];
            check(bad == 0, std::format("tags: {} differ", bad));
            return failures ? 1 : 0;
        }
        // The normalized references as the dump wrote them.
        auto frames_u8 = [&](const fs::path& p) {
            const auto s_ = read_file(p);
            return std::vector<uint8_t>(s_.begin(), s_.end());
        };
        mako::video::Reference pic, snd, vid;
        pic.kind = mako::video::Reference::Kind::image;
        pic.image = {iw, ih, frames_u8(ref / "rf_image_norm.u8")};
        snd.kind = mako::video::Reference::Kind::audio;
        snd.audio = read_f32(ref / "rf_audio1.f32");
        snd.channels = 2;
        snd.sample_rate = 32000;
        vid.kind = mako::video::Reference::Kind::video;
        vid.frames = vf;
        vid.width = vw;
        vid.height = vh;
        vid.rgb = frames_u8(ref / "rf_video_norm.u8");
        vid.audio = read_f32(ref / "rf_audio2.f32");
        vid.channels = 2;
        vid.sample_rate = 32000;
        if (mode == "ref_vision") {
            std::vector<uint32_t> idx;
            std::vector<double> stamps;
            Engine::sample_frames(vf, idx, stamps);
            check(idx == std::vector<uint32_t>{0, 12} && stamps == std::vector<double>{0.25}, "the frames sampled from the video reference: 0 and 12, labelled 0.25 s");
            const auto [sh, sw] = Engine::vision_video_size(uint32_t(idx.size()), vh, vw);
            check(sh == vh && sw == vw, std::format("the processor keeps the pair at {}x{}", sw, sh));
            auto& vis = engine.vision();
            const auto fi = vis.encode(pic.image.rgb, ih, iw);
            compare("picture vision tokens", fi.embeds, read_f32(ref / "rf_vis_image_merger.f32"), 5120, 2e-2);
            for (int j = 0; j < 3; ++j)
                compare(std::format("picture DeepStack {}", j), fi.deepstack[j], read_f32(ref / std::format("rf_vis_image_deepstack{}.f32", j)), 5120, 2e-2);
            const uint64_t fb = uint64_t(vh) * vw * 3;
            const auto fv = vis.encode(std::span(vid.rgb).subspan(idx[0] * fb, fb), std::span(vid.rgb).subspan(idx[1] * fb, fb), vh, vw);
            compare("frame pair vision tokens", fv.embeds, read_f32(ref / "rf_vis_video_merger.f32"), 5120, 2e-2);
            for (int j = 0; j < 3; ++j)
                compare(std::format("frame pair DeepStack {}", j), fv.deepstack[j], read_f32(ref / std::format("rf_vis_video_deepstack{}.f32", j)), 5120, 2e-2);
            return failures ? 1 : 0;
        }
        if (mode == "ref_text") {
            std::vector<uint8_t> tags;
            std::vector<int32_t> got_ids;
            const mako::video::Reference refs[3] = {pic, snd, vid};
            const auto text = engine.reference_embeddings(prompt, refs, &tags, &got_ids);
            check(got_ids == ids, std::format("the presentation's {} token ids (the reference's {})", got_ids.size(), ids.size()));
            check(tags == text_tags, "the presentation's tags");
            const auto want = read_f32(ref / "rf_text_embeds.f32");
            compare("layer 50 over the presentation", text, want, H3::kText, 0.03);
            // The rows furthest off, with their tokens, to see what a drift sits on.
            std::vector<std::pair<double, uint32_t>> rows;
            for (uint32_t t = 0; t < T && text.size() == want.size(); ++t) {
                double dot = 0, a = 0, b = 0;
                for (uint32_t i = 0; i < H3::kText; ++i) {
                    const double x = text[uint64_t(t) * H3::kText + i], y = want[uint64_t(t) * H3::kText + i];
                    dot += x * y;
                    a += x * x;
                    b += y * y;
                }
                rows.emplace_back(dot / std::sqrt(a * b), t);
            }
            std::ranges::sort(rows);
            for (size_t i = 0; i < std::min<size_t>(8, rows.size()); ++i)
                std::println("       row {} (token {}, tag {}): cosine {:.4f}", rows[i].second, ids[rows[i].second],
                             text_tags[rows[i].second], rows[i].first);
            return failures ? 1 : 0;
        }
        const auto vt = floats_of(m["video_timesteps"]), at = floats_of(m["audio_timesteps"]);
        if (mode == "ref2va") {
            Engine::Request r;
            r.prompt = prompt;
            r.width = r.height = size;
            r.frames = frames;
            r.steps = steps;
            const auto nv = read_f32(ref / "rf_noise_video_rows.f32"), na = read_f32(ref / "rf_noise_audio_rows.f32");
            r.noise_video = nv;
            r.noise_audio = na;
            // The raw references: the picture at its own size, the video's frames at theirs (24 fps), the soundtracks.
            const auto& rsz = m["rf_image_size"].get_array();
            const auto& rsh = m["rf_video_shape"].get_array();
            pic.image = {uint32_t(rsz[0].get_number()), uint32_t(rsz[1].get_number()), frames_u8(ref / "rf_image.u8")};
            vid.frames = uint32_t(rsh[0].get_number());
            vid.height = uint32_t(rsh[1].get_number());
            vid.width = uint32_t(rsh[2].get_number());
            vid.rgb = frames_u8(ref / "rf_video.u8");
            r.references = {pic, snd, vid};
            // The noise mixed into the latents, [24][F][h][w] in the dump, [F][h][w][24] here.
            auto fhwc = [&](const std::vector<float>& cfhw, uint32_t F, uint32_t h, uint32_t w) {
                std::vector<float> o(cfhw.size());
                for (uint32_t c = 0; c < 24; ++c)
                    for (uint32_t f = 0; f < F; ++f)
                        for (uint32_t y = 0; y < h; ++y)
                            for (uint32_t x = 0; x < w; ++x)
                                o[((uint64_t(f) * h + y) * w + x) * 24 + c] = cfhw[((uint64_t(c) * F + f) * h + y) * w + x];
                return o;
            };
            const auto n0 = fhwc(read_f32(ref / "rf_cond_noise0.f32"), 1, ih / 16, iw / 16);
            const auto n1 = fhwc(read_f32(ref / "rf_cond_noise1.f32"), vlf, vh / 16, vw / 16);
            r.reference_noise = {n0, n1};
            const auto L = engine.denoise(r);
            check(L.condition_rows == C && L.condition_audio_rows == CA, std::format("{} conditioning rows, {} reference audio rows", L.condition_rows, L.condition_audio_rows));
            const auto want = read_f32(ref / "rf_final_video_rows.f32");
            compare("reference rows", std::span(L.video).first(uint64_t(C) * H3::kVideoPatch),
                    std::vector<float>(want.begin(), want.begin() + uint64_t(C) * H3::kVideoPatch), H3::kVideoPatch, 1e-2);
            compare("final video rows", std::span(L.video).subspan(uint64_t(C) * H3::kVideoPatch),
                    std::vector<float>(want.begin() + uint64_t(C) * H3::kVideoPatch, want.end()), H3::kVideoPatch, 0.06);
            const auto wa = read_f32(ref / "rf_final_audio_rows.f32");
            compare("reference audio rows", std::span(L.audio).first(uint64_t(CA) * H3::kAudioChannels),
                    std::vector<float>(wa.begin(), wa.begin() + uint64_t(CA) * H3::kAudioChannels), H3::kAudioChannels, 2e-2);
            compare("final audio rows", std::span(L.audio).subspan(uint64_t(CA) * H3::kAudioChannels),
                    std::vector<float>(wa.begin() + uint64_t(CA) * H3::kAudioChannels, wa.end()), H3::kAudioChannels, 0.03);
            return failures ? 1 : 0;
        }
        // ref_dit: the transformer on the reference's rows.
        const auto text = read_f32(ref / "rf_text_embeds.f32");
        check(text.size() == uint64_t(T) * H3::kText, std::format("{} text rows", T));
        H3& dit = engine.transformer();
        std::map<std::string, std::vector<float>> probes;
        dit.set_probe([&](const std::string& name, int layer, const float* data, size_t n) {
            probes[layer < 0 ? name : std::format("{}{}", name, layer)] = std::vector<float>(data, data + n);
        });
        dit.begin(text, layout);
        for (uint32_t i = 0; i < vt.size(); ++i) {
            probes.clear();
            const auto vin = read_f32(ref / std::format("rf_step{}_video_rows.f32", i));
            const auto ain = read_f32(ref / std::format("rf_step{}_audio_rows.f32", i));
            const auto vel = dit.step(vin, ain, vt[i], at[i]);
            std::println(" step {} (t video {}, audio {})", i, vt[i], at[i]);
            compare("packed input", probes["x0"], read_f32(ref / std::format("rf_s{}_x0.f32", i)), H3::kDim, 5e-3);
            for (int b : {0, 1, 24})
                compare(std::format("block {}", b), probes[std::format("block{}", b)],
                        read_f32(ref / std::format("rf_s{}_block{}.f32", i, b)), H3::kDim, 1e-2);
            // The last block and the final norm over the rows the heads read (the generated audio and video rows, the
            // sequence's last): the text and reference video rows, which no head reads, drift there in BF16 (hundreds
            // of them to cosines near 0 against the fp32 reference) while the generated rows hold; the drift is
            // reported, not judged.
            const uint32_t gen0 = S - (2 * n_audio + lat_f * (lat_h / 2) * (lat_w / 2));
            const auto R = H3::rows(layout);
            for (const auto& [name, file, tol] : {std::tuple{"block 49", "block49", 4e-2}, std::tuple{"final norm", "norm_out", 6e-2}}) {
                const auto& got = probes[file];
                const auto want = read_f32(ref / std::format("rf_s{}_{}.f32", i, file));
                if (got.size() != want.size()) {
                    check(false, std::format("{}: {} values, the reference {}", name, got.size(), want.size()));
                    continue;
                }
                compare(std::format("{} (generated rows)", name), std::span(got).subspan(uint64_t(gen0) * H3::kDim),
                        std::vector<float>(want.begin() + uint64_t(gen0) * H3::kDim, want.end()), H3::kDim, tol);
                size_t below[5] = {}; // by role: text, reference video, reference audio, audio, video
                double worst = 1;
                for (uint32_t r = 0; r < S; ++r) {
                    double dot = 0, a = 0, b = 0;
                    for (uint32_t k = 0; k < H3::kDim; ++k) {
                        const double x = got[uint64_t(r) * H3::kDim + k], y = want[uint64_t(r) * H3::kDim + k];
                        dot += x * y;
                        a += x * x;
                        b += y * y;
                    }
                    const double cs = dot / std::sqrt(a * b);
                    worst = std::min(worst, cs);
                    if (cs < 0.99)
                        ++below[R.roles[r]];
                }
                std::println("       {} rows under cosine 0.99 (lowest {:.3f}): {} text, {} reference video, {} reference audio, {} audio, {} video",
                             name, worst, below[0], below[1], below[2], below[3], below[4]);
            }
            const auto pv = read_f32(ref / std::format("rf_step{}_pred_video.f32", i));
            compare("video velocity (generated rows)", std::span(vel.video).subspan(uint64_t(C) * H3::kVideoPatch),
                    std::vector<float>(pv.begin() + uint64_t(C) * H3::kVideoPatch, pv.end()), H3::kVideoPatch, 2e-2);
            const auto pa = read_f32(ref / std::format("rf_step{}_pred_audio.f32", i));
            compare("audio velocity (generated rows)", std::span(vel.audio).subspan(uint64_t(CA) * H3::kAudioChannels),
                    std::vector<float>(pa.begin() + uint64_t(CA) * H3::kAudioChannels, pa.end()), H3::kAudioChannels, 2e-2);
        }
    } else if (mode == "ref_image") {
        // The picture reference (1024 x 1024: four by four tiles blended) through the VAE encoder against the
        // reference's posterior and its seed-42 sample.
        auto& vae = engine.video_vae();
        const auto& sz = m["rf_image_norm_size"].get_array();
        const uint32_t w = uint32_t(sz[0].get_number()), h = uint32_t(sz[1].get_number()), lh = h / 16, lw = w / 16;
        const auto rgb = read_file(ref / "rf_image_norm.u8");
        check(rgb.size() == uint64_t(h) * w * 3, std::format("picture reference {}x{}", w, h));
        auto hwc = [&](const std::vector<float>& chw) { // [24][h][w] -> [h][w][24]
            std::vector<float> o(chw.size());
            for (uint32_t c = 0; c < 24; ++c)
                for (uint32_t y = 0; y < lh; ++y)
                    for (uint32_t x = 0; x < lw; ++x)
                        o[(uint64_t(y) * lw + x) * 24 + c] = chw[(uint64_t(c) * lh + y) * lw + x];
            return o;
        };
        std::vector<float> mean, sd;
        const std::span<const uint8_t> px(reinterpret_cast<const uint8_t*>(rgb.data()), rgb.size());
        vae.encode_moments(px, h, w, mean, sd);
        compare("picture posterior mean", mean, hwc(read_f32(ref / "rf_cond_image_mean.f32")), 24, 2e-2);
        compare("picture posterior sd", sd, hwc(read_f32(ref / "rf_cond_image_std.f32")), 24, 2e-2);
        compare("picture latents (generator reproduced)", vae.encode(px, h, w), hwc(read_f32(ref / "rf_cond_image_latents.f32")), 24, 2e-2);
    } else if (mode == "ref_video") {
        // The video reference (22 frames of 768 x 768: two 17-frame clips in three by three tiles, the last three
        // latents dropped) through the 3D encoder against the reference's posterior and its seed-42 sample.
        auto& vae = engine.video_vae();
        const auto& sh = m["rf_video_norm_shape"].get_array();
        const uint32_t F = uint32_t(sh[0].get_number()), h = uint32_t(sh[1].get_number()), w = uint32_t(sh[2].get_number());
        const uint32_t Fe = uint32_t(m["rf_video_frames_encoded"].get_number()), lh = h / 16, lw = w / 16;
        const auto rgb = read_file(ref / "rf_video_norm.u8");
        check(rgb.size() == uint64_t(F) * h * w * 3, std::format("video reference {} frames of {}x{}, {} encoded", F, w, h, Fe));
        const auto want_mean = read_f32(ref / "rf_cond_video_mean.f32"); // [1][24][F'][lh][lw]
        const uint32_t LF = uint32_t(want_mean.size() / (24ull * lh * lw));
        auto hwc = [&](const std::vector<float>& cfhw) {
            std::vector<float> o(cfhw.size());
            for (uint32_t c = 0; c < 24; ++c)
                for (uint32_t f = 0; f < LF; ++f)
                    for (uint32_t y = 0; y < lh; ++y)
                        for (uint32_t x = 0; x < lw; ++x)
                            o[((uint64_t(f) * lh + y) * lw + x) * 24 + c] = cfhw[((uint64_t(c) * LF + f) * lh + y) * lw + x];
            return o;
        };
        std::vector<float> mean, sd;
        const std::span<const uint8_t> px(reinterpret_cast<const uint8_t*>(rgb.data()), uint64_t(Fe) * h * w * 3);
        vae.encode_video_moments(px, Fe, h, w, mean, sd);
        check(mean.size() == want_mean.size(), std::format("{} latent frames", mean.size() / (24ull * lh * lw)));
        compare("video posterior mean", mean, hwc(want_mean), 24, 2e-2);
        compare("video posterior sd", sd, hwc(read_f32(ref / "rf_cond_video_std.f32")), 24, 2e-2);
        compare("video latents (generator reproduced)", vae.encode_video(px, Fe, h, w), hwc(read_f32(ref / "rf_cond_video_latents.f32")), 24, 2e-2);
    } else if (mode == "ref_audio") {
        // The reference soundtracks through the audio VAE's encoder: the normalized channel-major rows against the
        // fp32 reference's posterior means.
        auto& avae = engine.audio_vae();
        for (int i = 1; i <= 2; ++i) {
            const auto wave = read_f32(ref / std::format("rf_audio{}.f32", i)); // [2][samples]
            const size_t half = wave.size() / 2;
            const auto got = avae.encode(std::span(wave).first(half), std::span(wave).subspan(half));
            compare(std::format("soundtrack {} rows", i), got, read_f32(ref / std::format("rf_audio{}_rows.f32", i)), 32, 2e-2);
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
