#pragma once

#include "models/runtime.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace mako::video {

// MiniMax-H3's audio VAE decoder (DAC's latent space, BigVGAN's decoder): 32 latent channels at 40 a second to a
// mono 32 kHz waveform, 800 samples a latent. The decoder runs once per stereo channel: a 1x1 projection to 2048
// channels, a 7-tap convolution, seven transposed-convolution upsamplings (5, 5, 2, 2, 2, 2, 2) each followed by the
// mean of three anti-aliased multi-periodicity residual blocks (kernels 3, 7, 11; dilations 1, 3, 5; SnakeBeta
// activations between a 2x Kaiser-sinc upsampling and downsampling), then one more activation and a 7-tap convolution
// to one channel. F32 throughout, as the checkpoint (weight norms folded in at load).
class AudioVae final : public models::TransformerRuntime {
public:
    // The audio_vae directory: config.json (latents_mean / latents_std) and model.safetensors.
    AudioVae(const std::string& dir, mtl::Context& ctx, gpu::Telemetry telemetry = gpu::Telemetry::off);

    const char* name() const override { return "audio VAE"; }

    static constexpr uint32_t kChannels = 32, kRate = 32000, kHop = 800;

    // The transformer's normalized audio rows ([2 * n][32], the left channel's n latents then the right's) to
    // stereo samples [2][n * 800] in [-1, 1].
    std::vector<float> decode(std::span<const float> rows, uint32_t n);

private:
    struct Conv {
        Tensor w, b; // w [cout][cin][k] (transposed: [cin][cout][k]); b empty: none
        uint32_t cin = 0, cout = 0, k = 0, stride = 1, pad = 0, dil = 1;
        bool transposed = false;
    };
    // An alias-free SnakeBeta activation: exp(alpha), exp(beta) per channel, the resampling filters.
    struct Act {
        Tensor alpha, beta, up, down;
    };
    struct Res {
        Conv c1[3], c2[3];
        Act a1[3], a2[3];
    };
    Conv load_conv(const safetensors::Files& f, const std::string& name, uint32_t k, uint32_t stride, uint32_t pad,
                   uint32_t dil, bool transposed, bool bias);
    Act load_act(const safetensors::Files& f, const std::string& name);
    // One channel's latents [32][n] (denormalized) in x, the waveform [n * 800] out.
    void decode_channel(MTL::ComputeCommandEncoder*& enc, MTL::Buffer* x, uint32_t n, MTL::Buffer* out);
    void conv(MTL::ComputeCommandEncoder*& enc, const Conv& c, MTL::Buffer* x, uint32_t tin, MTL::Buffer* y,
              uint32_t tout);
    // y = act(x) over [channels][t]: upsampled 2x, SnakeBeta, downsampled 2x; up: scratch of [channels][2 t].
    void activation(MTL::ComputeCommandEncoder*& enc, const Act& a, MTL::Buffer* x, uint32_t channels, uint32_t t,
                    MTL::Buffer* up, MTL::Buffer* y);

    Conv in_proj_, conv_pre_, conv_post_;
    std::vector<Conv> ups_;
    std::vector<Res> res_;
    Act act_post_;
    std::vector<float> mean_, std_;
};

} // namespace mako::video
