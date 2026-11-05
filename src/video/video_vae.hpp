#pragma once

#include "models/conv.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mako::video {

// MiniMax-H3's video VAE decoder: a ViT over the latent voxels of a clip (one token each, plus four register tokens
// and a zero token), 36 blocks of 32 heads of 64 with three-axis rotary positions, each token then a 16x16 patch of
// four pixel frames. Latents are 16x smaller than the video and 4x fewer in time (17 frames encode to 5 latents),
// decoded 7 latent frames (28 pixel frames) at a time in 256-pixel tiles whose overlaps are blended, consecutive clips
// cross-faded over 5 frames, as the model was released. The weights are F16 here (F32 in the file).
class VideoVae final : public models::ConvRuntime {
public:
    // The video_vae directory: config.json (latents_mean / latents_std) and source/model.safetensors.
    VideoVae(const std::string& dir, mtl::Context& ctx, gpu::Telemetry telemetry = gpu::Telemetry::off);

    const char* name() const override { return "video VAE"; }

    static constexpr uint32_t kLatent = 24, kPatch = 16, kPatchT = 4, kDim = 2048, kHeads = 32, kHeadDim = 64;
    static constexpr uint32_t kRot = 48, kFfn = 8192, kRegisters = 4, kOut = 3 * kPatchT * kPatch * kPatch;
    static constexpr uint32_t kClipLatents = 5, kClipOverlap = 2, kPrePad = 3, kFrameOverlap = 5;
    static constexpr uint32_t kTile = 256, kTileOverlap = 64;
    static constexpr uint32_t kEncCh = 128, kEncodeSeed = 42; // the encoder's base width; the posterior's seed

    // A decoded frame: its index and its pixels [16 h][16 w][3] in [0, 1].
    using FrameSink = std::function<void(uint32_t index, std::span<const float> rgb)>;
    // The transformer's normalized latents as voxels [frames][h][w][24] (frames = 5 n + 2) to the video's 17 n + 5
    // frames, delivered in order.
    void decode(std::span<const float> latents, uint32_t frames, uint32_t h, uint32_t w, const FrameSink& sink);
    static uint32_t pixel_frames(uint32_t latent_frames) { return (latent_frames - 2) / kClipLatents * 17 + 5; }

    // A keyframe [h][w][3] of 8-bit RGB (h and w multiples of 16) to its normalized latents [h / 16][w / 16][24]: the
    // causal 3D encoder on the single frame (its last time taps), in 256-pixel tiles blended as the decoder's, the
    // posterior sampled under seed 42 as the released pipeline does (its torch generator reproduced; `noise`
    // [h / 16][w / 16][24] replaces the draw), the sample rounded to fp16. The encoder is loaded on first use.
    std::vector<float> encode(std::span<const uint8_t> rgb, uint32_t h, uint32_t w, std::span<const float> noise = {});
    // The posterior's mean and standard deviation, [h / 16][w / 16][24] each.
    void encode_moments(std::span<const uint8_t> rgb, uint32_t h, uint32_t w, std::vector<float>& mean,
                        std::vector<float>& sd);
    // A reference video's frames [F][h][w][3] (F = 17 n + 5 at 24 fps) to its normalized latents [5 n + 2][h / 16]
    // [w / 16][24]: the causal 3D encoder over 17-frame clips (the frames padded by repeating the last one, the last
    // three latents dropped), tiled and sampled as encode(); the 3D weights are loaded on first use.
    std::vector<float> encode_video(std::span<const uint8_t> rgb, uint32_t frames, uint32_t h, uint32_t w,
                                    std::span<const float> noise = {});
    void encode_video_moments(std::span<const uint8_t> rgb, uint32_t frames, uint32_t h, uint32_t w,
                              std::vector<float>& mean, std::vector<float>& sd);
    // torch's CPU randn(n, generator=manual_seed(seed)): mt19937's 24-bit uniforms through Box-Muller over blocks of
    // 16 (matches to about 1e-6; n at least 16).
    static std::vector<float> torch_randn(size_t n, uint32_t seed);

private:
    struct Block {
        Tensor norm1, norm2, qkv, qkv_b, o, o_b, w1, w1_b, w2, w2_b, scale1, scale2;
    };
    // The ViT over a clip's voxels [f][th][tw][24] (denormalized, after the post-quant projection): pixels
    // [4 f][16 th][16 tw][3] in the decoder's own (ImageNet-normalized) space.
    void vit(std::span<const float> voxels, uint32_t f, uint32_t th, uint32_t tw, std::vector<float>& out);
    // Tiles over `length` pixels: each tile's start and the overlap with the one before it (0 for the first).
    struct Tile {
        uint32_t start, size, overlap;
    };
    static std::vector<Tile> tiles(uint32_t length);

    Tensor embed_, embed_b_, registers_, ones_, out_norm_w_, out_norm_b_, proj_out_, proj_out_b_;
    std::vector<Block> blocks_;
    std::vector<float> mean_, std_, pq_w_, pq_b_; // latent statistics; the post-quant 1x1 convolution
    std::string dir_;

    // The CNN encoder: conv_in, 6 levels of 2 residual blocks (group norm, SiLU, 3x3, reflect padded; a 1x1 shortcut
    // where the width changes) with a stride-2 downsample after the first 4, norm_out, conv_out to 48 moments, the
    // 1x1 quant convolution.
    struct EncRes {
        Tensor g1, b1, g2, b2;
        Conv c1, c2, sc;
        bool shortcut = false;
    };
    struct EncLevel {
        std::vector<EncRes> res;
        bool down = false;
        Conv down_conv;
    };
    struct Encoder {
        Conv in, out, quant;
        Tensor out_g, out_b;
        std::vector<EncLevel> levels;
    };
    std::unique_ptr<Encoder> enc_, enc3d_; // single-frame weights (the last time taps) and all three taps
    void load_encoder();
    void load_encoder3d();
    void load_encoder_impl(bool temporal);
    // The 3D encoder over one clip tile [T][th][tw][3] (ImageNet-normalized) to moments [T'][th / 16][tw / 16][48].
    void encode_clip_tile(std::span<const float> pixels, uint32_t T, uint32_t th, uint32_t tw, std::vector<float>& out,
                          uint32_t& T_out);
    // A whole frame stack's moments [F'][h / 16][w / 16][48], tiles blended, clips concatenated, the tail dropped.
    void video_moments(std::span<const uint8_t> rgb, uint32_t frames, uint32_t h, uint32_t w, std::vector<float>& out,
                       uint32_t& latent_frames);
    // One tile of ImageNet-normalized pixels [th][tw][3] to its moments [th / 16][tw / 16][48] (mean, log variance).
    void encode_tile(std::span<const float> pixels, uint32_t th, uint32_t tw, std::vector<float>& moments);
    // The whole frame's moments, tiles blended.
    void moments(std::span<const uint8_t> rgb, uint32_t h, uint32_t w, std::vector<float>& out);
};

} // namespace mako::video
