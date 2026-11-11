#pragma once

#include "models/h3_vae.hpp"

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace mako::video {

// MiniMax-H3's video VAE (models::H3Vae, the decoder with tokens four frames deep) and its decoding of a whole
// video as the model was released: latents are 16x smaller than the video and 4x fewer in time (17 frames encode to 5
// latents), decoded 7 latent frames (28 pixel frames) at a time, consecutive clips cross-faded over 5 frames.
class VideoVae final : public models::H3Vae {
public:
    // The video_vae directory: config.json (latents_mean / latents_std) and source/model.safetensors.
    VideoVae(const std::string& dir, mtl::Context& ctx, gpu::Telemetry telemetry = gpu::Telemetry::off);
    static constexpr uint32_t kPatchT = 4, kClipLatents = 5, kClipOverlap = 2, kPrePad = 3, kFrameOverlap = 5;
    // A decoded frame: its index and its pixels [16 h][16 w][3] in [0, 1].
    using FrameSink = std::function<void(uint32_t index, std::span<const float> rgb)>;
    using models::H3Vae::decode;
    // The transformer's normalized latents as voxels [frames][h][w][24] (frames = 5 n + 2) to the video's 17 n + 5
    // frames, delivered in order.
    void decode(std::span<const float> latents, uint32_t frames, uint32_t h, uint32_t w, const FrameSink& sink);
    static uint32_t pixel_frames(uint32_t latent_frames) { return (latent_frames - 2) / kClipLatents * 17 + 5; }
};

} // namespace mako::video
