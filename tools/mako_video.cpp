// Text to video with audio with MiniMax-H3 from its original files.
//   mako_video <model dir> <prompt> [-o out.mp4] [--width N --height N] [--seconds F | --frames N] [--steps N]
//              [--seed N] [--lora FILE[:SCALE] ...] [--still FILE] [--keep-text-encoder] [--text-encoder DIR]
//              [--transformer PATH] [--llm URL] [--llm-api-key KEY]
// --width / --height: the canvas in multiples of 32 (default the model's 16:9 canvas, 1344x768). --seconds / --frames:
// the length, 5 to 15 s at 24 fps, snapped up to 17 n + 5 frames (default 124 frames, 5.2 s). --steps: schedule
// points, one forward each but the last (default 50; a turbo LoRA wants 5 or 9). --lora adds a LoRA merged into the
// transformer (repeatable; SCALE its strength, default 1). --still: also a PNG of the middle frame. --keep-text-encoder
// keeps the text encoder loaded (49 GB) instead of freeing it before the transformer runs.
// --llm: a mako_serve on this machine whose memory to borrow while the model is loaded (it unloads its LLM meanwhile).
#include "app/cli.hpp"
#include "log/log.hpp"
#include "memory/borrower.hpp"
#include "mtl/context.hpp"
#include "video/engine.hpp"
#include "video/mp4.hpp"

#include <Foundation/Foundation.hpp>

#include <chrono>
#include <cmath>
#include <format>
#include <optional>
#include <print>
#include <stdexcept>
#include <vector>

namespace {

std::string thermal_state()
{
    switch (NS::ProcessInfo::processInfo()->thermalState()) {
    case NS::ProcessInfoThermalStateNominal:
        return "nominal";
    case NS::ProcessInfoThermalStateFair:
        return "fair";
    case NS::ProcessInfoThermalStateSerious:
        return "serious";
    case NS::ProcessInfoThermalStateCritical:
        return "critical";
    }
    return "unknown";
}

// "FILE" or "FILE:SCALE".
mako::video::Lora parse_lora(const std::string& spec)
{
    mako::video::Lora l{spec, 1.0f};
    const size_t colon = spec.rfind(':');
    if (colon != std::string::npos && colon > 0 && spec.find('/', colon) == std::string::npos) {
        l.path = spec.substr(0, colon);
        l.scale = mako::app::Cli::parse<float>(spec.substr(colon + 1));
    }
    return l;
}

} // namespace

int main(int argc, char** argv)
{
    std::string llm, llm_key, out = "video.mp4", still, text_encoder, transformer;
    mako::gpu::Telemetry telemetry = mako::gpu::Telemetry::off;
    uint32_t width = 0, height = 0, frames = 0, steps = 0;
    double seconds = 0;
    uint64_t seed = 0;
    bool keep_text = false;
    std::vector<mako::video::Lora> loras;
    mako::app::Cli cli("mako_video", "<model dir> <prompt>", 2, 2);
    cli.option("-o", "FILE", "the MP4 to write (default video.mp4)", out)
        .option("--width", "N", "pixels, a multiple of 32 (default: the 16:9 canvas, 1344)", width)
        .option("--height", "N", "pixels, a multiple of 32 (default 768)", height)
        .option("--seconds", "F", "length in seconds, 5 to 15 (default 5.2)", seconds)
        .option("--frames", "N", "length in frames at 24 fps, snapped up to 17 n + 5 (default 124)", frames)
        .option("--steps", "N", "schedule points, one forward each but the last (default 50)", steps)
        .option("--seed", "N", "the noise's seed (default 0)", seed)
        .option("--lora", "FILE[:SCALE]", "add a LoRA merged into the transformer (repeatable)",
                [&](const std::string& v) { loras.push_back(parse_lora(v)); })
        .option("--still", "FILE", "also write the middle frame as a PNG", still)
        .flag("--keep-text-encoder", "keep the text encoder loaded beside the transformer", keep_text)
        .option("--text-encoder", "DIR", "another directory of the text encoder's weights", text_encoder)
        .option("--transformer", "PATH", "another directory or file of the transformer's weights", transformer)
        .option("--telemetry", "LEVEL", "off, summary, steps, layers or kernels: the transformer's GPU time per kernel",
                [&](const std::string& v) {
                    const auto t = mako::gpu::parse_telemetry(v);
                    if (!t)
                        throw std::invalid_argument("--telemetry: off, summary, steps, layers or kernels");
                    telemetry = *t;
                })
        .option("--llm", "URL", "a mako_serve on this machine to borrow memory from", llm)
        .option("--llm-api-key", "KEY", "the key that mako_serve wants", llm_key)
        .secret();
    cli.env_prefix("MAKO_VIDEO_");
    const auto pos = cli.parse(argc, argv);
    using Clock = std::chrono::steady_clock;
    auto since = [](Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); };
    try {
        if (seconds > 0 && frames)
            throw std::runtime_error("--seconds or --frames, not both");
        if (seconds > 0)
            frames = uint32_t(std::lround(seconds * mako::video::Engine::kFps));
        const auto t_load = Clock::now();
        std::optional<mako::memory::Borrower> borrower; // the LLM's memory, given back on the way out
        if (!llm.empty()) {
            borrower.emplace(llm, llm_key, "mako_video");
            borrower->acquire();
            mako::log::info("the LLM's memory is free ({:.1f} s)", since(t_load));
        }
        mako::mtl::Context ctx;
        mako::video::Engine engine(pos[0], ctx, {.text_encoder = text_encoder, .transformer = transformer,
                                                 .keep_text_encoder = keep_text, .telemetry = telemetry});
        for (const auto& l : loras)
            engine.add_lora(l);
        mako::video::Engine::Request r;
        r.prompt = pos[1];
        r.width = width;
        r.height = height;
        if (frames)
            r.frames = frames;
        r.steps = steps;
        r.seed = seed;
        auto t_step = Clock::now();
        r.progress = [&](uint32_t i, uint32_t total) {
            mako::log::info("step {} of {}: {:.1f} s, thermal {}", i, total, since(t_step), thermal_state());
            t_step = Clock::now();
        };
        const auto t0 = Clock::now();
        const auto res = engine.generate(r);
        mako::log::info("{}x{}, {} frames ({:.2f} s) in {:.1f} s", res.width, res.height, res.frames,
                        double(res.frames) / res.fps, since(t0));
        mako::video::write_mp4(out, res.width, res.height, res.fps, res.frames, res.rgb, res.audio, res.sample_rate);
        mako::log::info("wrote {}", out);
        if (!still.empty()) {
            const uint64_t frame = uint64_t(res.width) * res.height * 3;
            mako::video::write_png(still, res.width, res.height,
                                   std::span(res.rgb).subspan(frame * (res.frames / 2), frame));
            mako::log::info("wrote {}", still);
        }
    } catch (const std::exception& e) {
        mako::log::error("{}", e.what());
        return 1;
    }
    return 0;
}
