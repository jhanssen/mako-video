#include "video/image.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace mako::video {

namespace {

float lanczos3(float x)
{
    x = std::fabs(x);
    if (x >= 3.0f)
        return 0.0f;
    if (x < 1e-6f)
        return 1.0f;
    const float px = std::numbers::pi_v<float> * x;
    return 3.0f * std::sin(px) * std::sin(px / 3.0f) / (px * px);
}

// One axis of the resampling: for each output index its first input index and normalized weights.
struct Taps {
    std::vector<uint32_t> first;
    std::vector<std::vector<float>> weights;
};

Taps taps(uint32_t in, uint32_t out)
{
    const double scale = double(in) / double(out), fscale = std::max(scale, 1.0), support = 3.0 * fscale;
    Taps t;
    t.first.resize(out);
    t.weights.resize(out);
    for (uint32_t o = 0; o < out; ++o) {
        const double center = (o + 0.5) * scale;
        const int64_t xmin = std::max<int64_t>(0, int64_t(center - support + 0.5));
        const int64_t xmax = std::min<int64_t>(in, int64_t(center + support + 0.5));
        auto& w = t.weights[o];
        double sum = 0;
        for (int64_t x = xmin; x < xmax; ++x) {
            w.push_back(lanczos3(float((double(x) - center + 0.5) / fscale)));
            sum += w.back();
        }
        for (float& v : w)
            v = float(v / sum);
        t.first[o] = uint32_t(xmin);
    }
    return t;
}

} // namespace

Image resize_lanczos(const Image& img, uint32_t width, uint32_t height)
{
    if (!img.width || !img.height || img.rgb.size() != uint64_t(img.width) * img.height * 3 || !width || !height)
        throw std::runtime_error("resize: an image [height][width][3] to a positive size");
    if (img.width == width && img.height == height)
        return img;
    // Horizontal pass to 8 bits, then vertical, as PIL does.
    const Taps tx = taps(img.width, width), ty = taps(img.height, height);
    std::vector<uint8_t> mid(uint64_t(img.height) * width * 3);
    for (uint32_t y = 0; y < img.height; ++y)
        for (uint32_t x = 0; x < width; ++x)
            for (uint32_t c = 0; c < 3; ++c) {
                float s = 0.0f;
                const auto& w = tx.weights[x];
                for (size_t k = 0; k < w.size(); ++k)
                    s += w[k] * float(img.rgb[(uint64_t(y) * img.width + tx.first[x] + k) * 3 + c]);
                mid[(uint64_t(y) * width + x) * 3 + c] = uint8_t(std::clamp(std::lround(s), 0l, 255l));
            }
    Image out{width, height, std::vector<uint8_t>(uint64_t(width) * height * 3)};
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x)
            for (uint32_t c = 0; c < 3; ++c) {
                float s = 0.0f;
                const auto& w = ty.weights[y];
                for (size_t k = 0; k < w.size(); ++k)
                    s += w[k] * float(mid[(uint64_t(ty.first[y] + k) * width + x) * 3 + c]);
                out.rgb[(uint64_t(y) * width + x) * 3 + c] = uint8_t(std::clamp(std::lround(s), 0l, 255l));
            }
    return out;
}

} // namespace mako::video
