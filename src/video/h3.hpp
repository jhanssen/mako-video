#pragma once

#include "models/runtime.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mako::video {

// MiniMax-H3's diffusion transformer: 50 blocks of full self-attention over one packed sequence [text | keyframe
// conditions | audio | video], each row modulated by the AdaLN parameters of its (timestep, modality); a two-block
// token refiner on the projected text; one velocity head for the video rows and one for the audio rows. From the
// original BF16 shards (rows of the fused q / k / v matrices interleaved per head) or a single-file checkpoint of
// the same names (rows stacked q, k, v), including the pruned ones whose time embedder and modulation projections
// are folded into an 8-coordinate curve table (adaln_t_table), as the ComfyUI and Civitai files are.
class H3 final : public models::TransformerRuntime {
public:
    // Which of the 50 blocks' matrices are kept as int8 rows with a scale each (the GEMM at 1.7x BF16's rate, half
    // the bytes; the activations quantized per row before each): a mask of these.
    enum Int8 : uint32_t { kInt8Qkv = 1, kInt8Out = 2, kInt8Fc1 = 4, kInt8Fc2 = 8, kInt8All = 15 };
    // "none", "all", or a comma list of qkv, out, fc1 and fc2; nullopt for anything else.
    static std::optional<uint32_t> parse_int8(std::string_view s);

    // The transformer's directory (its shards and config.json) or a single file; int8: the Int8 mask.
    H3(const std::string& path, mtl::Context& ctx, gpu::Telemetry telemetry = gpu::Telemetry::off, uint32_t int8 = 0);

    const char* name() const override { return "MiniMax-H3"; }

    static constexpr uint32_t kDim = 5376, kHeads = 56, kText = 5120, kVideoPatch = 96, kAudioChannels = 32;
    static constexpr uint32_t kTimeFreq = 256, kTimeDim = 2688, kRot = 96, kRopeFreqs = 16;
    static constexpr uint32_t kCurve = 8, kCurvePoints = 1025; // a pruned checkpoint's curve table
    static constexpr float kRopeTheta = 10000.0f;
    // Modality tags, which index the AdaLN table with the timestep.
    static constexpr uint32_t kVideoTag = 0, kTextTag = 1, kAudioTag = 2;

    // The packed sequence of a generation: text rows, then the keyframe conditioning frames (fl2va), the audio rows
    // (channel-major: every latent of the left channel, then the right's) and the generated video rows (frame-major,
    // then row-major over 2x2 patches of the latent frame).
    struct Layout {
        uint32_t text = 0;
        uint32_t frames = 0;      // generated latent frames
        uint32_t h = 0, w = 0;    // latent height and width (even)
        uint32_t audio = 0;       // audio latents a channel
        uint32_t cond_frames = 0; // keyframe conditioning frames before the generated ones
        std::vector<uint8_t> cond_last; // per conditioning frame: anchored at the last generated frame (else the first)
        std::vector<uint8_t> text_tags; // a tag per text row (kTextTag, or kVideoTag for a vision block's); empty:
                                        // all text
        uint32_t rows_per_frame() const { return (h / 2) * (w / 2); }
        uint32_t cond_rows() const { return cond_frames * rows_per_frame(); }
        uint32_t video_rows() const { return (cond_frames + frames) * rows_per_frame(); }
        uint32_t audio_rows() const { return 2 * audio; }
        uint32_t rows() const { return text + video_rows() + audio_rows(); }
    };
    // Begins a generation: the text embeddings [text][5120] through the condition projection and the token refiner
    // (kept for every step), the layout's rotary positions and tags.
    void begin(std::span<const float> text, const Layout& layout);
    // One forward: the video rows [video_rows()][96] (conditioning rows first) and audio rows [audio_rows()][32] at
    // noise levels t = 1 - sigma (t_video for the generated video rows and the text, t_audio for the audio, t_cond
    // for the conditioning rows), their velocities out (conditioning rows' zero).
    struct Velocity {
        std::vector<float> video, audio;
    };
    Velocity step(std::span<const float> video, std::span<const float> audio, float t_video, float t_audio,
                  float t_cond = 0.999f);
    const Layout& layout() const { return layout_; }

protected:
    LoraTarget lora_target(const std::string& module) override;
    Tensor* lora_bias(const std::string& module) override;

private:
    struct Block {
        Tensor norm1, norm2, qkv, o, norm_q, norm_k, fc1, fc2;
        Tensor adaln, adaln_b; // the transformer blocks' modulation projection (none in the refiner's)
    };
    void load(const safetensors::Files& f);
    Block load_block(const safetensors::Files& f, const std::string& prefix, bool modulated);
    // The fused q / k / v matrix with its rows stacked [q; k; v] (the shards interleave them per head).
    Tensor upload_qkv(const safetensors::Files& f, const std::string& name, bool int8);
    // A small F32 matrix with its rows padded to a multiple of 256, so it runs on the GEMM.
    Tensor upload_padded(const safetensors::Files& f, const std::string& name, uint32_t rows);
    // Attention and the FFN of a block over `rows` rows of x (its first `rows` of Tg), modulated per row from mods
    // (null: the refiner's plain pre-norm block).
    void block(MTL::ComputeCommandEncoder*& enc, const Block& B, MTL::Buffer* x, uint32_t rows, uint32_t Tg,
               MTL::Buffer* mods, uint64_t mods_off, MTL::Buffer* off, MTL::Buffer* cs);

    std::vector<Block> blocks_, refiner_;
    Tensor refiner_norm_, cond_, cond_b_, video_in_, video_in_b_, audio_in_, audio_in_b_;
    Tensor t1_, t1_b_, t2_, t2_b_, final_adaln_, final_adaln_b_, final_norm_, video_out_, video_out_b_, audio_out_,
        audio_out_b_;
    bool interleaved_ = false; // the file's q / k / v rows interleaved per head
    uint32_t int8_ = 0;        // the Int8 mask
    // A pruned checkpoint: the time embedder and the 2688-wide modulation inputs folded into an 8-coordinate curve
    // c(t), tabulated at 1025 points of t in [0, 1] (adaln_t_table) and read with linear interpolation; the blocks'
    // and the final layer's modulation projections then take the 8 coordinates (no SiLU).
    bool curve_ = false;
    std::vector<float> table_; // [1025][8]
    Layout layout_;
    std::vector<uint32_t> tags_; // a modality tag per row
    uint32_t cond_rows_ = 0, S_ = 0;
};

} // namespace mako::video
