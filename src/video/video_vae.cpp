#include "video/video_vae.hpp"

#include <algorithm>
#include <format>
#include <stdexcept>

namespace mako::video {

VideoVae::VideoVae(const std::string& dir, mtl::Context& ctx, gpu::Telemetry telemetry)
    : models::H3Vae(dir + "/source/model.safetensors", dir + "/config.json", ctx, telemetry, "video VAE")
{
    if (patch_t() != kPatchT)
        throw std::runtime_error(std::format("{}: a decoder with tokens {} frame(s) deep, not {} (the image model's VAE?)",
                                             dir, patch_t(), kPatchT));
}

void VideoVae::decode(std::span<const float> latents, uint32_t frames, uint32_t h, uint32_t w, const FrameSink& sink)
{
    if (frames < 7 || (frames - 2) % kClipLatents || !h || !w || latents.size() != uint64_t(frames) * h * w * kLatent)
        throw std::runtime_error("video VAE: voxels [5 n + 2][h][w][24]");
    const uint32_t H = h * kPatch, W = w * kPatch, chunks = (frames - 2) / kClipLatents;
    const uint32_t cf = kClipLatents + kClipOverlap, pf = cf * kPatchT; // latent and pixel frames of a clip
    const uint64_t frame = uint64_t(H) * W * 3, voxels = uint64_t(h) * w * kLatent;
    std::vector<float> canvas, overlap, rgb(frame);
    uint32_t emitted = 0;
    // A finished frame of the canvas (ImageNet-normalized) to the sink, in [0, 1].
    auto emit = [&](const float* src) {
        for (uint64_t i = 0; i < frame; ++i)
            rgb[i] = std::clamp(src[i] * kPixelStd[i % 3] + kPixelMean[i % 3], 0.0f, 1.0f);
        sink(emitted++, rgb);
    };
    for (uint32_t ci = 0; ci < chunks; ++ci) {
        decode_normalized(latents.subspan(uint64_t(ci) * kClipLatents * voxels, uint64_t(cf) * voxels), cf, h, w, canvas);
        // The clip's 28 frames: 3 of padding then 17 frames, 3 of padding then the 5 the next clip fades in from.
        for (uint32_t fr = kPrePad; fr < kClipLatents * kPatchT; ++fr) {
            float* cur = &canvas[uint64_t(fr) * frame];
            if (ci > 0 && fr - kPrePad < kFrameOverlap) {
                const float wb = float(fr - kPrePad) / float(kFrameOverlap), wa = 1.0f - wb;
                const float* prev = &overlap[uint64_t(fr - kPrePad) * frame];
                for (uint64_t k = 0; k < frame; ++k)
                    cur[k] = prev[k] * wa + cur[k] * wb;
            }
            emit(cur);
        }
        overlap.assign(canvas.begin() + uint64_t(kClipLatents * kPatchT + kPrePad) * frame,
                       canvas.begin() + uint64_t(pf) * frame);
    }
    for (uint32_t fr = 0; fr < kFrameOverlap; ++fr)
        emit(&overlap[uint64_t(fr) * frame]);
}

} // namespace mako::video
