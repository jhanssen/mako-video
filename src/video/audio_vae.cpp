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
    : TransformerRuntime(ctx, telemetry), dir_(dir)
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

AudioVae::Tensor AudioVae::load_alpha(const safetensors::Files& f, const std::string& name)
{
    const auto v = floats(f, name); // Snake1d's alpha, [1][channels][1], used as it is
    Tensor t = reserve(v.size() * 4, gguf::Type::F32, {v.size()});
    std::memcpy(static_cast<char*>(t.buf->contents()) + t.off, v.data(), v.size() * 4);
    return t;
}

void AudioVae::load_encoder()
{
    if (enc_)
        return;
    const auto f = open_files(dir_ + "/model.safetensors");
    static constexpr uint32_t kStrides[5] = {2, 4, 4, 5, 5}, kDil[3] = {1, 3, 9};
    auto e = std::make_unique<Encoder>();
    e->in = load_conv(f, "encoder.block.0", 7, 1, 3, 1, false, true);
    uint32_t ch = 64;
    for (uint32_t i = 0; i < 5; ++i) {
        EncBlock& B = e->blocks[i];
        const std::string p = std::format("encoder.block.{}.block.", i + 1);
        for (uint32_t j = 0; j < 3; ++j) {
            EncUnit& u = B.units[j];
            const std::string q = std::format("{}{}.block.", p, j);
            u.a1 = load_alpha(f, q + "0.alpha");
            u.c1 = load_conv(f, q + "1", 7, 1, 3 * kDil[j], kDil[j], false, true);
            u.a2 = load_alpha(f, q + "2.alpha");
            u.c2 = load_conv(f, q + "3", 1, 1, 0, 1, false, true);
            if (u.c1.cin != ch || u.c2.cout != ch)
                throw std::runtime_error(q + ": unexpected channels");
        }
        B.act = load_alpha(f, p + "3.alpha");
        const uint32_t s_ = kStrides[i];
        B.down = load_conv(f, p + "4", 2 * s_, s_, (s_ + 1) / 2, 1, false, true);
        if (B.down.cin != ch || B.down.cout != 2 * ch)
            throw std::runtime_error(p + "4: unexpected channels");
        ch *= 2;
    }
    e->act_out = load_alpha(f, "encoder.block.6.alpha");
    e->out = load_conv(f, "encoder.block.7", 3, 1, 1, 1, false, true);
    if (e->out.cin != kLatentDim || e->out.cout != kLatentDim)
        throw std::runtime_error("audio VAE: the encoder does not end in 2048 channels");
    auto vec = [&](const std::string& n) { return floats(f, n); };
    e->ln1_w = vec("pre_block.norm1.weight");
    e->ln1_b = vec("pre_block.norm1.bias");
    e->ln3_w = vec("pre_block.norm3.weight");
    e->ln3_b = vec("pre_block.norm3.bias");
    e->ln2_w = vec("pre_block.norm2.weight");
    e->ln2_b = vec("pre_block.norm2.bias");
    e->lnm_w = vec("pre_block.mlp.norm.weight");
    e->lnm_b = vec("pre_block.mlp.norm.bias");
    e->qkv = vec("pre_block.attn.qkv.weight");
    e->q_bias = vec("pre_block.attn.q_bias");
    e->v_bias = vec("pre_block.attn.v_bias");
    e->attn_proj_w = vec("pre_block.attn.proj.weight");
    e->attn_proj_b = vec("pre_block.attn.proj.bias");
    e->proj_w = vec("pre_block.proj.weight");
    e->proj_b = vec("pre_block.proj.bias");
    e->w0 = vec("pre_block.mlp.w0.weight");
    e->w0_b = vec("pre_block.mlp.w0.bias");
    e->w1 = vec("pre_block.mlp.w1.weight");
    e->w1_b = vec("pre_block.mlp.w1.bias");
    e->w2 = vec("pre_block.mlp.w2.weight");
    e->w2_b = vec("pre_block.mlp.w2.bias");
    e->mean_w = vec("mean_proj.weight");
    e->mean_b = vec("mean_proj.bias");
    if (e->qkv.size() != uint64_t(3) * kLatentDim * kLatentDim || e->proj_w.size() != uint64_t(kChannels) * kLatentDim ||
        e->mean_w.size() != kChannels * kChannels || e->w0.size() != 64 * kChannels)
        throw std::runtime_error("audio VAE: the projection block's shapes are not the expected ones");
    enc_ = std::move(e);
    log::info("audio VAE encoder loaded ({:.0f} MB resident in all)", double(dense_bytes_) / 1e6);
}

void AudioVae::encode_channel(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t samples, MTL::Buffer* out)
{
    // Channels x time throughout; three scratch buffers of the widest layer (64 channels at full rate).
    const uint64_t cap = uint64_t(64) * samples * 4;
    MTL::Buffer* h = scratch("e_h", cap);
    MTL::Buffer* t1 = scratch("e_t1", cap);
    MTL::Buffer* t2 = scratch("e_t2", cap);
    auto snake = [&](MTL::Buffer* b, const Tensor& alpha, uint32_t channels, uint32_t t) {
        dispatch_at(enc, "snake_beta", {b, alpha, alpha}, &t, sizeof(t), MTL::Size(t, channels, 1), MTL::Size(64, 1, 1));
    };
    uint32_t t = samples, ch = 64;
    conv(enc, enc_->in, x, samples, h, t);
    for (const EncBlock& B : enc_->blocks) {
        for (const EncUnit& u : B.units) {
            // y = c2(snake(c1(snake(h)))); h += y
            const uint32_t count = ch * t;
            dispatch_at(enc, "copy_f32", {h, t1}, &count, sizeof(count), MTL::Size(count, 1, 1), MTL::Size(256, 1, 1));
            snake(t1, u.a1, ch, t);
            conv(enc, u.c1, t1, t, t2, t);
            snake(t2, u.a2, ch, t);
            conv(enc, u.c2, t2, t, t1, t);
            add(enc, h, t1, h, ch * t);
        }
        snake(h, B.act, ch, t);
        const uint32_t tout = (t + 2 * B.down.pad - B.down.k) / B.down.stride + 1;
        conv(enc, B.down, h, t, t1, tout);
        std::swap(h, t1);
        t = tout;
        ch *= 2;
    }
    snake(h, enc_->act_out, ch, t);
    conv(enc, enc_->out, h, t, out, t);
}

std::vector<float> AudioVae::encode(std::span<const float> left, std::span<const float> right)
{
    if (left.empty() || left.size() != right.size())
        throw std::runtime_error("audio VAE: a stereo waveform, both channels of the same length");
    load_encoder();
    const uint32_t samples = uint32_t((left.size() + kHop - 1) / kHop * kHop), n = samples / kHop, D = kLatentDim;
    MTL::Buffer* x = scratch("e_x", uint64_t(samples) * 4);
    MTL::Buffer* out = scratch("e_out", uint64_t(D) * n * 4);
    std::vector<float> feats(uint64_t(2) * n * D); // [channel][n][2048]
    for (uint32_t ch = 0; ch < 2; ++ch) {
        auto* xf = static_cast<float*>(x->contents());
        std::fill_n(xf, samples, 0.0f);
        const auto& w = ch ? right : left;
        std::copy(w.begin(), w.end(), xf);
        MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
        cur_cb_ = cb;
        MTL::ComputeCommandEncoder* enc = encoder(cb, 0, -1);
        encode_channel(enc, x, samples, out);
        enc->endEncoding();
        finish(cb);
        const auto* o = static_cast<const float*>(out->contents()); // [2048][n]
        for (uint32_t c = 0; c < D; ++c)
            for (uint32_t t = 0; t < n; ++t)
                feats[(uint64_t(ch) * n + t) * D + c] = o[uint64_t(c) * n + t];
    }
    // The projection block on the CPU: h = proj(LN3(x)) + attn(LN1(x)); h += mlp(LN2(h)); then mean_proj.
    const Encoder& E = *enc_;
    auto layer_norm = [](std::span<const float> x, const std::vector<float>& w, const std::vector<float>& b,
                         std::vector<float>& y, uint32_t n_) {
        for (uint32_t i = 0; i < n_; ++i) {
            double mean = 0, var = 0;
            for (uint32_t c = 0; c < w.size(); ++c)
                mean += x[uint64_t(i) * w.size() + c];
            mean /= double(w.size());
            for (uint32_t c = 0; c < w.size(); ++c) {
                const double d = x[uint64_t(i) * w.size() + c] - mean;
                var += d * d;
            }
            const double inv = 1.0 / std::sqrt(var / double(w.size()) + 1e-5);
            for (uint32_t c = 0; c < w.size(); ++c)
                y[uint64_t(i) * w.size() + c] = float((x[uint64_t(i) * w.size() + c] - mean) * inv) * w[c] + b[c];
        }
    };
    auto linear = [](std::span<const float> x, uint32_t n_, uint32_t in, const std::vector<float>& w,
                     const std::vector<float>* b, uint32_t outd, std::vector<float>& y) { // w [outd][in]
        y.assign(uint64_t(n_) * outd, 0.0f);
        for (uint32_t i = 0; i < n_; ++i)
            for (uint32_t o = 0; o < outd; ++o) {
                double s = b ? (*b)[o] : 0.0;
                const float* xr = &x[uint64_t(i) * in];
                const float* wr = &w[uint64_t(o) * in];
                for (uint32_t c = 0; c < in; ++c)
                    s += double(xr[c]) * wr[c];
                y[uint64_t(i) * outd + o] = float(s);
            }
    };
    constexpr uint32_t kHeads = 8, kHd = kLatentDim / kHeads; // 256
    std::vector<float> rows(uint64_t(2) * n * kChannels);
    std::vector<float> ln(uint64_t(n) * D), qkv, direct, attn_out(uint64_t(n) * kHd), pooled(uint64_t(n) * kChannels),
        h, ln2, mlp0, mlp1, mlp2, mean;
    for (uint32_t ch = 0; ch < 2; ++ch) {
        const std::span<const float> xch(&feats[uint64_t(ch) * n * D], uint64_t(n) * D);
        // The direct path.
        layer_norm(xch, E.ln3_w, E.ln3_b, ln, n);
        linear(ln, n, D, E.proj_w, &E.proj_b, kChannels, direct);
        // The causal attention: 8 heads of 256, the outputs averaged over the heads and then pooled 8 to 1 onto 32.
        layer_norm(xch, E.ln1_w, E.ln1_b, ln, n);
        linear(ln, n, D, E.qkv, nullptr, 3 * D, qkv);
        std::fill(attn_out.begin(), attn_out.end(), 0.0f);
        const float scale = 1.0f / std::sqrt(float(kHd));
        std::vector<double> sc(n);
        for (uint32_t hh = 0; hh < kHeads; ++hh)
            for (uint32_t i = 0; i < n; ++i) {
                const float* q = &qkv[uint64_t(i) * 3 * D + hh * kHd];
                double mx = -1e300;
                for (uint32_t j = 0; j <= i; ++j) {
                    const float* k = &qkv[uint64_t(j) * 3 * D + D + hh * kHd];
                    double s = 0;
                    for (uint32_t d = 0; d < kHd; ++d)
                        s += double(q[d] + E.q_bias[hh * kHd + d]) * k[d]; // k's bias is zero
                    sc[j] = s * scale;
                    mx = std::max(mx, sc[j]);
                }
                double sum = 0;
                for (uint32_t j = 0; j <= i; ++j) {
                    sc[j] = std::exp(sc[j] - mx);
                    sum += sc[j];
                }
                float* o = &attn_out[uint64_t(i) * kHd];
                for (uint32_t j = 0; j <= i; ++j) {
                    const float* v = &qkv[uint64_t(j) * 3 * D + 2 * D + hh * kHd];
                    const float wgt = float(sc[j] / sum);
                    for (uint32_t d = 0; d < kHd; ++d)
                        o[d] += wgt * (v[d] + E.v_bias[hh * kHd + d]) / float(kHeads);
                }
            }
        for (uint32_t i = 0; i < n; ++i)
            for (uint32_t c = 0; c < kChannels; ++c) {
                float s = 0.0f;
                for (uint32_t d = 0; d < kHd / kChannels; ++d)
                    s += attn_out[uint64_t(i) * kHd + c * (kHd / kChannels) + d];
                pooled[uint64_t(i) * kChannels + c] = s / float(kHd / kChannels);
            }
        linear(pooled, n, kChannels, E.attn_proj_w, &E.attn_proj_b, kChannels, h);
        for (uint64_t i = 0; i < h.size(); ++i)
            h[i] += direct[i];
        // The GeGLU MLP on LN2(h): w2(gelu_tanh(w0(LN(x))) * w1(LN(x))).
        ln2.resize(h.size());
        layer_norm(h, E.ln2_w, E.ln2_b, ln2, n);
        std::vector<float> lnm(ln2.size());
        layer_norm(ln2, E.lnm_w, E.lnm_b, lnm, n);
        linear(lnm, n, kChannels, E.w0, &E.w0_b, 64, mlp0);
        linear(lnm, n, kChannels, E.w1, &E.w1_b, 64, mlp1);
        for (uint64_t i = 0; i < mlp0.size(); ++i) {
            const float v = mlp0[i];
            const float g = 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
            mlp0[i] = g * mlp1[i];
        }
        linear(mlp0, n, 64, E.w2, &E.w2_b, kChannels, mlp2);
        for (uint64_t i = 0; i < h.size(); ++i)
            h[i] += mlp2[i];
        linear(h, n, kChannels, E.mean_w, &E.mean_b, kChannels, mean);
        for (uint32_t t = 0; t < n; ++t)
            for (uint32_t c = 0; c < kChannels; ++c)
                rows[(uint64_t(ch) * n + t) * kChannels + c] = (mean[uint64_t(t) * kChannels + c] - mean_[c]) / std_[c];
    }
    return rows;
}

} // namespace mako::video
