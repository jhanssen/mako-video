#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mako::video {

// An 8-bit RGB image [height][width][3].
struct Image {
    uint32_t width = 0, height = 0;
    std::vector<uint8_t> rgb;
};

// An image file (PNG, JPEG, HEIC, ... what ImageIO reads) at its own size, transparency over white; the file's
// orientation tag is not applied. Throws std::runtime_error.
Image read_image(const std::string& path);
// `img` resampled to width x height with a 3-lobe Lanczos filter as PIL's resize does it (the filter widened by the
// ratio when shrinking, each pass rounded to 8 bits).
Image resize_lanczos(const Image& img, uint32_t width, uint32_t height);

} // namespace mako::video
