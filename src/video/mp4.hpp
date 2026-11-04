#pragma once

#include <cstdint>
#include <span>
#include <string>

namespace mako::video {

// An MP4 of H.264 video with an AAC stereo soundtrack: `count` frames of 8-bit RGB [height][width][3] at `fps`, the
// samples [2][n] in [-1, 1] at `sample_rate` (none: no audio track). An existing file is replaced. Throws
// std::runtime_error.
void write_mp4(const std::string& path, uint32_t width, uint32_t height, uint32_t fps, uint32_t count,
               std::span<const uint8_t> rgb, std::span<const float> audio, uint32_t sample_rate);

// A PNG of one 8-bit RGB frame [height][width][3]. Throws std::runtime_error.
void write_png(const std::string& path, uint32_t width, uint32_t height, std::span<const uint8_t> rgb);

} // namespace mako::video
