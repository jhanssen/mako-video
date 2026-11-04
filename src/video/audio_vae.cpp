#include "video/audio_vae.hpp"

#include "log/log.hpp"

#include <glaze/glaze.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace mako::video {

namespace {

struct Conv1dArgs {
    uint32_t cin, cout, tin, tout, k, stride, pad, dil;
};
struct ResampleArgs {
    uint32_t tin, tout, taps;
};

constexpr uint32_t kRates[7] = {5, 5, 2, 2, 2, 2, 2}, kUpKernels[7] = {9, 9, 4, 4, 4, 4, 4};
constexpr uint32_t kResKernels[3] = {3, 7, 11}, kDilations[3] = {1, 3, 5};
constexpr uint32_t kDecoderDim = 1024, kLatentDim = 2048, kTaps = 12;

safetensors::Files open_files(const std::string& path)
{
    auto f = safetensors::Files::open(path);
    if (!f)
        throw std::runtime_error(f.error());
    return std::move(*f);
}

} // namespace

AudioVae::AudioVae(const std::string& dir, mtl::Context& ctx, gpu::Telemetry telemetry)
    : TransformerRuntime(ctx, telemetry)
{
    {
        std::ifstream in(dir + "/config.json", std::ios::binary);
        const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        glz::generic j;
        if (!in || glz::read_json(j, text) || !j.contains("latents_mean") || !j.contains("latents_std"))
            throw std::runtime_error(dir + "/config.json: no latents_mean / latents_std");
        for (const auto& v : j["latents_mean"].get_array())
            mean_.push_back(float(v.get_number()));
        for (const auto& v : j["latents_std"].get_array())
            std_.push_back(float(v.get_number()));
        if (mean_.size() != kChannels || std_.size() != kChannels)
            throw std::runtime_error("audio VAE: latent statistics are not over 32 channels");
    }
    const auto f = open_files(dir + "/model.safetensors");
    in_proj_ = load_conv(f, "dec_in_proj", 1, 1, 0, 1, false, true);
    conv_pre_ = load_conv(f, "decoder.conv_pre", 7, 1, 3, 1, false, true);
    uint32_t ch = kDecoderDim;
    for (uint32_t i = 0; i < 7; ++i) {
        ups_.push_back(load_conv(f, std::format("decoder.ups.{}.0", i), kUpKernels[i], kRates[i],
                                 (kUpKernels[i] - kRates[i]) / 2, 1, true, true));
        ch /= 2;
        for (uint32_t j = 0; j < 3; ++j) {
            const std::string p = std::format("decoder.resblocks.{}.", i * 3 + j);
            const uint32_t k = kResKernels[j];
            Res r;
            for (uint32_t d = 0; d < 3; ++d) {
                r.c1[d] = load_conv(f, std::format("{}convs1.{}", p, d), k, 1, (k * kDilations[d] - kDilations[d]) / 2,
                                    kDilations[d], false, true);
                r.c2[d] = load_conv(f, std::format("{}convs2.{}", p, d), k, 1, (k - 1) / 2, 1, false, true);
                r.a1[d] = load_act(f, std::format("{}activations.{}", p, 2 * d));
                r.a2[d] = load_act(f, std::format("{}activations.{}", p, 2 * d + 1));
                if (r.c1[d].cin != ch || r.c2[d].cout != ch)
                    throw std::runtime_error(p + ": unexpected channels");
            }
            res_.push_back(std::move(r));
        }
    }
    act_post_ = load_act(f, "decoder.activation_post");
    conv_post_ = load_conv(f, "decoder.conv_post", 7, 1, 3, 1, false, false);
    if (in_proj_.cin != kChannels || in_proj_.cout != kLatentDim || conv_pre_.cin != kLatentDim ||
        conv_pre_.cout != kDecoderDim || conv_post_.cin != ch || conv_post_.cout != 1)
        throw std::runtime_error("audio VAE: unexpected shapes");
    log::info("audio VAE decoder: {:.0f} MB", double(dense_bytes_) / 1e6);
}

AudioVae::Conv AudioVae::load_conv(const safetensors::Files& f, const std::string& name, uint32_t k, uint32_t stride,
                                   uint32_t pad, uint32_t dil, bool transposed, bool bias)
{
    Conv c;
    c.k = k;
    c.stride = stride;
    c.pad = pad;
    c.dil = dil;
    c.transposed = transposed;
    std::vector<float> w;
    std::vector<uint64_t> dims;
    if (const auto r = f.find(name + ".weight_v")) { // weight norm: g v / |v| over each first-axis slice
        dims = r.tensor->dims;
        w = models::read_floats(*r.file, *r.tensor);
        const auto g = floats(f, name + ".weight_g");
        const uint64_t rows = dims[2], per = dims[0] * dims[1];
        if (g.size() != rows)
            throw std::runtime_error(name + ": weight_g does not match weight_v");
        for (uint64_t i = 0; i < rows; ++i) {
            double n = 0;
            for (uint64_t j = 0; j < per; ++j)
                n += double(w[i * per + j]) * w[i * per + j];
            const float s = float(double(g[i]) / std::sqrt(n));
            for (uint64_t j = 0; j < per; ++j)
                w[i * per + j] *= s;
        }
    } else if (const auto r = f.find(name + ".weight")) {
        dims = r.tensor->dims;
        w = models::read_floats(*r.file, *r.tensor);
    } else {
        throw std::runtime_error("missing convolution " + name);
    }
    if (dims.size() != 3 || dims[0] != k)
        throw std::runtime_error(name + ": not a 1-D convolution of kernel " + std::to_string(k));
    // dims are contiguous first: [k][in][out] for a convolution's [out][in][k], [k][out][in] for a transposed one's.
    c.cin = uint32_t(transposed ? dims[2] : dims[1]);
    c.cout = uint32_t(transposed ? dims[1] : dims[2]);
    c.w = reserve(w.size() * 4, gguf::Type::F32, dims);
    std::memcpy(static_cast<char*>(c.w.buf->contents()) + c.w.off, w.data(), w.size() * 4);
    if (bias)
        c.b = upload_f32(f, name + ".bias");
    return c;
}

AudioVae::Act AudioVae::load_act(const safetensors::Files& f, const std::string& name)
{
    Act a;
    // SnakeBeta's parameters are stored in log space.
    for (const char* which : {"alpha", "beta"}) {
        auto v = floats(f, std::format("{}.act.{}", name, which));
        for (float& x : v)
            x = std::exp(x);
        Tensor t = reserve(v.size() * 4, gguf::Type::F32, {v.size()});
        std::memcpy(static_cast<char*>(t.buf->contents()) + t.off, v.data(), v.size() * 4);
        (which[0] == 'a' ? a.alpha : a.beta) = t;
    }
    a.up = upload_f32(f, name + ".upsample.filter");
    a.down = upload_f32(f, name + ".downsample.lowpass.filter");
    if (a.up.dims[0] != kTaps || a.down.dims[0] != kTaps)
        throw std::runtime_error(name + ": resampling filters are not of 12 taps");
    return a;
}

void AudioVae::conv(MTL::ComputeCommandEncoder*& enc, const Conv& c, MTL::Buffer* x, uint32_t tin, MTL::Buffer* y,
                    uint32_t tout)
{
    const Conv1dArgs a{c.cin, c.cout, tin, tout, c.k, c.stride, c.pad, c.dil};
    enc->setComputePipelineState(pso(c.transposed ? "conv1d_transposed" : "conv1d"));
    enc->setBuffer(x, 0, 0);
    enc->setBuffer(c.w.buf, c.w.off, 1);
    enc->setBuffer(c.b.buf ? c.b.buf : c.w.buf, c.b.buf ? c.b.off : c.w.off, 2);
    enc->setBuffer(y, 0, 3);
    const uint32_t has_bias = c.b.buf ? 1 : 0;
    enc->setBytes(&a, sizeof(a), 4);
    enc->setBytes(&has_bias, sizeof(has_bias), 5);
    enc->dispatchThreads(MTL::Size(tout, c.cout, 1), MTL::Size(64, 1, 1));
    after(enc, c.transposed ? "conv1d_transposed" : "conv1d");
}

void AudioVae::activation(MTL::ComputeCommandEncoder*& enc, const Act& a, MTL::Buffer* x, uint32_t channels,
                          uint32_t t, MTL::Buffer* up, MTL::Buffer* y)
{
    const ResampleArgs ua{t, 2 * t, kTaps};
    dispatch_at(enc, "upsample2", {x, a.up, up}, &ua, sizeof(ua), MTL::Size(2 * t, channels, 1), MTL::Size(64, 1, 1));
    const uint32_t n = 2 * t;
    dispatch_at(enc, "snake_beta", {up, a.alpha, a.beta}, &n, sizeof(n), MTL::Size(2 * t, channels, 1),
                MTL::Size(64, 1, 1));
    const ResampleArgs da{2 * t, t, kTaps};
    dispatch_at(enc, "downsample2", {up, a.down, y}, &da, sizeof(da), MTL::Size(t, channels, 1), MTL::Size(64, 1, 1));
}

void AudioVae::decode_channel(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t n, MTL::Buffer* out)
{
    // Channels x time, the time axis contiguous; the longest buffers are the final stage's at 8 x 2 x 800 n.
    const uint64_t cap = std::max<uint64_t>(uint64_t(kLatentDim) * n, uint64_t(2) * 8 * kHop * n) * 4;
    MTL::Buffer* a = scratch("a_a", cap);
    MTL::Buffer* b = scratch("a_b", cap);
    MTL::Buffer* r = scratch("a_r", cap);
    MTL::Buffer* s = scratch("a_s", cap);
    MTL::Buffer* up = scratch("a_up", cap);
    MTL::Buffer* acc = scratch("a_acc", cap);
    conv(enc, in_proj_, x, n, a, n);
    conv(enc, conv_pre_, a, n, b, n);
    MTL::Buffer* cur = b;
    MTL::Buffer* nxt = a;
    uint32_t t = n;
    for (uint32_t i = 0; i < 7; ++i) {
        const Conv& u = ups_[i];
        const uint32_t tout = (t - 1) * u.stride + u.k - 2 * u.pad;
        conv(enc, u, cur, t, nxt, tout);
        std::swap(cur, nxt);
        t = tout;
        const uint32_t ch = u.cout, count = ch * t;
        for (uint32_t j = 0; j < 3; ++j) {
            // x_j = the block on cur; r holds the block's running value.
            const Res& R = res_[i * 3 + j];
            const uint32_t copy[1] = {count};
            dispatch(enc, "copy_f32", {cur, r}, copy, sizeof(copy), MTL::Size(count, 1, 1), MTL::Size(256, 1, 1));
            for (uint32_t d = 0; d < 3; ++d) {
                activation(enc, R.a1[d], r, ch, t, up, s);
                conv(enc, R.c1[d], s, t, nxt, t);
                activation(enc, R.a2[d], nxt, ch, t, up, s);
                conv(enc, R.c2[d], s, t, nxt, t);
                add(enc, r, nxt, r, count);
            }
            if (j == 0)
                dispatch(enc, "copy_f32", {r, acc}, copy, sizeof(copy), MTL::Size(count, 1, 1), MTL::Size(256, 1, 1));
            else
                add(enc, acc, r, acc, count);
        }
        const float third = 1.0f / 3.0f;
        const struct {
            uint32_t n;
            float s;
        } sa{count, third};
        dispatch(enc, "scale_f32", {acc, cur}, &sa, sizeof(sa), MTL::Size(count, 1, 1), MTL::Size(256, 1, 1));
    }
    activation(enc, act_post_, cur, conv_post_.cin, t, up, s);
    conv(enc, conv_post_, s, t, out, t);
}

std::vector<float> AudioVae::decode(std::span<const float> rows, uint32_t n)
{
    if (!n || rows.size() != uint64_t(2) * n * kChannels)
        throw std::runtime_error("audio rows must be [2 n][32]");
    MTL::Buffer* x = scratch("a_x", uint64_t(kChannels) * n * 4);
    MTL::Buffer* out = scratch("a_out", uint64_t(n) * kHop * 4);
    std::vector<float> wav(uint64_t(2) * n * kHop);
    for (uint32_t ch = 0; ch < 2; ++ch) {
        // Denormalized, as channels x time.
        auto* xf = static_cast<float*>(x->contents());
        for (uint32_t t = 0; t < n; ++t)
            for (uint32_t c = 0; c < kChannels; ++c)
                xf[uint64_t(c) * n + t] = rows[(uint64_t(ch) * n + t) * kChannels + c] * std_[c] + mean_[c];
        MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
        cur_cb_ = cb;
        MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
        decode_channel(enc, x, n, out);
        enc->endEncoding();
        finish(cb);
        const auto* o = static_cast<const float*>(out->contents());
        for (uint64_t i = 0; i < uint64_t(n) * kHop; ++i)
            wav[uint64_t(ch) * n * kHop + i] = std::clamp(o[i], -1.0f, 1.0f);
    }
    return wav;
}

} // namespace mako::video
