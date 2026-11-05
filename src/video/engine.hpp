#pragma once

#include "models/qwen_text.hpp"
#include "text/tokenizer.hpp"
#include "video/audio_vae.hpp"
#include "models/qwen_vision.hpp"
#include "video/h3.hpp"
#include "video/image.hpp"
#include "video/video_vae.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace mako::video {

struct Options {
    std::string text_encoder; // another directory of the text encoder's safetensors (default: the model's)
    std::string transformer;  // another directory or file of the transformer's weights (default: the model's)
    // Keep the text encoder loaded between generations (49 GB); else it is loaded for each prompt and freed.
    bool keep_text_encoder = false;
    gpu::Telemetry telemetry = gpu::Telemetry::off;
    uint32_t int8 = 0; // H3::Int8: the transformer blocks' matrices kept as int8
    float sparse = 0;  // H3::set_sparse: the share of video cubes each cube attends (0: dense attention)
    uint32_t sparse_local = 1; // the cubes kept regardless of score: 0 none, 1 the cube itself, 3 with its neighbours
};

// A LoRA file and its scale, merged into the transformer's weights.
struct Lora {
    std::string path;
    float scale = 1.0f;
};

// MiniMax-H3 from its original files (a directory with model_index.json, text_encoder/, tokenizer/, transformer/,
// video_vae/, audio_vae/): the prompt through the Qwen3-VL conditioner (its 50th layer's output), the packed sequence
// of text, audio and video rows denoised by the transformer on two rectified-flow schedules (shift 12 for the video,
// 3 for the audio), one forward a step (the model is guidance-distilled), the video and audio VAEs decoding the
// result. Text to video with audio (t2va).
class Engine {
public:
    Engine(const std::string& dir, mtl::Context& ctx, const Options& opts = {});

    static constexpr uint32_t kFps = 24, kAudioRate = 32000, kAudioLatentsPerSecond = 40;
    static constexpr uint32_t kClipFrames = 17, kClipLatents = 5; // the video VAE's chunking: 17 n + 5 frames
    static constexpr uint32_t kMinFrames = 124, kMaxFrames = 362; // 5 to 15 seconds
    static constexpr uint32_t kTextLayer = 50;
    static constexpr float kVideoShift = 12.0f, kAudioShift = 3.0f;

    // Frames snapped up to the next 17 n + 5 the video VAE decodes, and the latent frames of that many.
    static uint32_t align_frames(uint32_t frames);
    static uint32_t latent_frames(uint32_t frames) { return (frames - kClipLatents) / kClipFrames * kClipLatents + 2; }
    static uint32_t audio_latents(uint32_t frames);
    // The canvas for an aspect ratio: the short edge 768, the area at most 768 x 1344, both sides multiples of 32.
    static std::pair<uint32_t, uint32_t> canvas(double aspect_w, double aspect_h);
    // The noise levels of a schedule: `steps` grid points from 1 to 0 pushed through the shift, in float as the
    // reference computes them (steps - 1 model evaluations).
    static std::vector<float> sigmas(uint32_t steps, float shift);

    struct Request {
        std::string prompt;
        uint32_t width = 0, height = 0; // multiples of 32 (0: the 16:9 canvas, 1344x768)
        uint32_t frames = kMinFrames;   // snapped up to 17 n + 5, 124 to 362
        uint32_t steps = 0;             // schedule grid points (steps - 1 forwards); 0: the model's 50
        uint64_t seed = 0;
        // Instead of the seed's: the video rows' noise [latent frames * (h / 32) * (w / 32)][96] and the audio rows'
        // [2 * audio latents][32].
        std::span<const float> noise_video, noise_audio;
        // fl2va: the first frame and / or the last, any size (put onto the canvas: the first stretched, the last
        // cover-cropped). Without width and height the canvas takes the first keyframe's aspect ratio.
        std::optional<Image> first, last;
        // Instead of the seed's: the noise mixed into each keyframe's latents ([latent_h][latent_w][24], first then
        // last).
        std::vector<std::span<const float>> keyframe_noise;
        float video_shift = kVideoShift, audio_shift = kAudioShift;
        std::function<void(uint32_t step, uint32_t steps)> progress;
    };
    // The denoised, normalized latents of a request: the video rows (frame-major, then 2x2 patches of the latent
    // frame, each row a patch's [24 channels][2][2] values) and the channel-major audio rows.
    struct Latents {
        uint32_t width = 0, height = 0, frames = 0; // pixels and frames of the video
        uint32_t latent_frames = 0, latent_h = 0, latent_w = 0, audio_latents = 0;
        uint32_t condition_rows = 0; // leading video rows that are keyframe conditioning, not decoded
        std::vector<float> video, audio;
    };
    // A keyframe onto the canvas as the released pipeline puts it: the first stretched (Lanczos), another scaled to
    // cover the canvas and cropped at the centre.
    static Image prepare_keyframe(const Image& img, uint32_t width, uint32_t height, bool first);
    Latents denoise(const Request& r);
    // The video rows as voxels [latent frames][latent_h][latent_w][24], the video VAE's input.
    static std::vector<float> voxels(const Latents& l);
    // The latents decoded: each video frame ([height][width][3] in [0, 1]) to `sink` in order, and the stereo
    // soundtrack [2][frames / 24 * 32000] in [-1, 1].
    void decode_video(const Latents& l, const VideoVae::FrameSink& sink);
    std::vector<float> decode_audio(const Latents& l);

    // A generation: the frames as 8-bit RGB and the soundtrack.
    struct Result {
        uint32_t width = 0, height = 0, frames = 0, fps = kFps, sample_rate = kAudioRate;
        std::vector<uint8_t> rgb;  // [frames][height][width][3]
        std::vector<float> audio;  // [2][samples]
    };
    Result generate(const Request& r);

    // The transformer's text input for a prompt: the conditioner's 50th layer's output, [tokens][5120]. With
    // keyframes (on the canvas), the presentation is a "<Picture i>: " label and a vision block per keyframe, then
    // the prompt; `tags` gets each token's modality (text, or video for a vision block's).
    std::vector<float> prompt_embeddings(const std::string& prompt, std::span<const Image> keyframes = {},
                                         std::vector<uint8_t>* tags = nullptr);
    const text::Tokenizer& tokenizer() const { return *tok_; }
    // The parts, each loaded on first use (the text encoder kept with Options::keep_text_encoder).
    H3& transformer();
    models::QwenText& text_encoder();
    models::QwenVision& vision();
    VideoVae& video_vae();
    AudioVae& audio_vae();
    void free_text_encoder()
    {
        text_.reset();
        vision_.reset();
    }
    // A LoRA merged into the transformer's weights, when it loads (the text encoder runs first: both do not fit).
    void add_lora(const Lora& l);
    uint32_t default_steps() const { return 50; }

private:
    mtl::Context& ctx_;
    Options opts_;
    std::string dir_;
    std::optional<text::Tokenizer> tok_;
    std::unique_ptr<models::QwenText> text_;
    std::unique_ptr<models::QwenVision> vision_;
    std::unique_ptr<H3> dit_;
    std::vector<Lora> loras_; // to merge when the transformer loads
    std::unique_ptr<VideoVae> vae_;
    std::unique_ptr<AudioVae> avae_;
};

} // namespace mako::video
