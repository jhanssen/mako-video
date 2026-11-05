// A media file's first seconds (its frames and soundtrack as AVFoundation decodes them) written as an MP4 at the
// source frame rate: a way to cut a reference clip, and to see what the reader makes of a file. With --frame, one
// frame of it as a PNG instead.
//   mako_clip <in> <out.mp4> [seconds]
//   mako_clip <in> --frame N <out.png>
#include "video/media.hpp"
#include "video/mp4.hpp"

#include <span>

#include <cmath>
#include <print>
#include <stdexcept>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::println(stderr, "mako_clip <in> <out.mp4> [seconds]");
        return 2;
    }
    try {
        if (std::string(argv[2]) == "--frame") {
            if (argc < 5) {
                std::println(stderr, "mako_clip <in> --frame N <out.png>");
                return 2;
            }
            const uint32_t n = uint32_t(std::stoul(argv[3]));
            auto m = mako::video::read_media(argv[1], true, false, 0);
            if (n >= m.frames)
                throw std::runtime_error(std::format("frame {} of {}", n, m.frames));
            const uint64_t frame = uint64_t(m.width) * m.height * 3;
            mako::video::write_png(argv[4], m.width, m.height, std::span(m.rgb).subspan(frame * n, frame));
            std::println("{}: frame {} of {} ({}x{}) to {}", argv[1], n, m.frames, m.width, m.height, argv[4]);
            return 0;
        }
        const double seconds = argc > 3 ? std::stod(argv[3]) : 0.0;
        auto m = mako::video::read_media(argv[1], true, true, seconds);
        if (!m.frames)
            throw std::runtime_error("no video track");
        const uint32_t fps = uint32_t(std::lround(m.fps));
        uint32_t frames = m.frames;
        if (seconds > 0)
            frames = std::min<uint32_t>(frames, uint32_t(std::lround(seconds * m.fps)));
        const uint64_t samples = m.channels ? m.audio.size() / m.channels : 0;
        const uint64_t keep = std::min<uint64_t>(samples, uint64_t(double(frames) / m.fps * m.sample_rate));
        // The soundtrack as stereo [2][keep].
        std::vector<float> audio;
        if (m.channels) {
            audio.resize(2 * keep);
            for (uint32_t ch = 0; ch < 2; ++ch)
                std::copy_n(&m.audio[uint64_t(std::min(ch, m.channels - 1)) * samples], keep, &audio[uint64_t(ch) * keep]);
        }
        std::println("{}: {} frames of {}x{} at {:.3g} fps{}; writing {} frames{}", argv[1], m.frames, m.width, m.height, m.fps,
                     m.channels ? std::format(", {} channel{} at {} Hz ({:.2f} s)", m.channels, m.channels == 1 ? "" : "s",
                                              m.sample_rate, double(samples) / m.sample_rate)
                                : std::string(", no soundtrack"),
                     frames, m.channels ? std::format(" and {:.2f} s of sound", double(keep) / m.sample_rate) : std::string());
        mako::video::write_mp4(argv[2], m.width, m.height, fps, frames,
                               std::span(m.rgb).first(uint64_t(frames) * m.width * m.height * 3), audio, m.sample_rate);
        return 0;
    } catch (const std::exception& e) {
        std::println(stderr, "{}", e.what());
        return 1;
    }
}
