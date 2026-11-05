#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mako::video {

// A media file's tracks as AVFoundation decodes them: the video's frames as 8-bit RGB (the track's rotation
// applied) at its nominal frame rate, and the soundtrack's samples as float at its own rate and channel count.
struct Media {
    uint32_t width = 0, height = 0, frames = 0;
    double fps = 0;
    std::vector<uint8_t> rgb;   // [frames][height][width][3]
    std::vector<float> audio;   // [channels][samples] in [-1, 1]
    uint32_t channels = 0, sample_rate = 0;
};
// The tracks asked for, at most `max_seconds` of each (0: all). A file without a video track gives no frames, one
// without a soundtrack no samples. Throws std::runtime_error.
Media read_media(const std::string& path, bool video, bool audio, double max_seconds = 0);

} // namespace mako::video
