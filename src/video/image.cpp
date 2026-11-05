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

// Keys' cubic with a = -0.5, over [-2, 2].
double bicubic(double x)
{
    constexpr double a = -0.5;
    x = std::fabs(x);
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0)
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

// The filter's taps for every output coordinate: PIL's antialiased scheme (torchvision's as well), the support
// scaled by the downscale factor, the weights normalized.
Taps taps(uint32_t in, uint32_t out, bool cubic = false)
{
    const double scale = double(in) / double(out), fscale = std::max(scale, 1.0), support = (cubic ? 2.0 : 3.0) * fscale;
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
            const double u = (double(x) - center + 0.5) / fscale;
            w.push_back(cubic ? float(bicubic(u)) : lanczos3(float(u)));
            sum += w.back();
        }
        for (float& v : w)
            v = float(v / sum);
        t.first[o] = uint32_t(xmin);
    }
    return t;
}

Image resize(const Image& img, uint32_t width, uint32_t height, bool cubic)
{
    if (!img.width || !img.height || img.rgb.size() != uint64_t(img.width) * img.height * 3 || !width || !height)
        throw std::runtime_error("resize: an image [height][width][3] to a positive size");
    if (img.width == width && img.height == height)
        return img;
    // Horizontal pass to 8 bits, then vertical, as PIL does.
    const Taps tx = taps(img.width, width, cubic), ty = taps(img.height, height, cubic);
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

} // namespace

Image resize_lanczos(const Image& img, uint32_t width, uint32_t height) { return resize(img, width, height, false); }

Image resize_bicubic(const Image& img, uint32_t width, uint32_t height)
{
    if (!img.width || !img.height || img.rgb.size() != uint64_t(img.width) * img.height * 3 || !width || !height)
        throw std::runtime_error("resize: an image [height][width][3] to a positive size");
    if (img.width == width && img.height == height)
        return img;
    // torch's antialiased resampling of 8-bit tensors: the taps in double as PIL places them, each axis's weights
    // rounded to 16-bit integers at the finest precision that keeps the largest one under 2^15, the sums rounded
    // and shifted back; horizontal pass first, 8 bits between the passes.
    struct Axis {
        std::vector<uint32_t> first;
        std::vector<std::vector<int32_t>> weights;
        int precision = 0;
    };
    auto axis = [](uint32_t in, uint32_t out) {
        const double scale = double(in) / double(out), fscale = std::max(scale, 1.0), support = 2.0 * fscale;
        std::vector<std::vector<double>> w(out);
        Axis a;
        a.first.resize(out);
        double wmax = 0;
        for (uint32_t i = 0; i < out; ++i) {
            const double center = scale * (i + 0.5);
            const int64_t xmin = std::max<int64_t>(int64_t(center - support + 0.5), 0);
            const int64_t xmax = std::min<int64_t>(int64_t(center + support + 0.5), in);
            a.first[i] = uint32_t(xmin);
            double total = 0;
            for (int64_t x = xmin; x < xmax; ++x) {
                const double v = bicubic((double(x) - center + 0.5) / fscale);
                w[i].push_back(v);
                total += v;
            }
            for (double& v : w[i]) {
                if (total != 0)
                    v /= total;
                wmax = std::max(wmax, v);
            }
        }
        for (a.precision = 0; a.precision < 22; ++a.precision)
            if (int(0.5 + wmax * double(1 << (a.precision + 1))) >= (1 << 15))
                break;
        a.weights.resize(out);
        for (uint32_t i = 0; i < out; ++i)
            for (double v : w[i]) {
                const double q = v * double(1 << a.precision);
                a.weights[i].push_back(int32_t(q >= 0 ? 0.5 + q : -0.5 + q));
            }
        return a;
    };
    const Axis ax = axis(img.width, width), ay = axis(img.height, height);
    auto pass = [](const Axis& a, const std::vector<uint8_t>& src, uint32_t rows, uint32_t in, uint32_t out, bool vertical,
                   std::vector<uint8_t>& dst) {
        // Horizontal: every row of `in` pixels to `out`; vertical: every column of `in` rows to `out` (rows = width).
        for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t o = 0; o < out; ++o)
                for (uint32_t c = 0; c < 3; ++c) {
                    int32_t sum = 1 << (a.precision - 1);
                    const auto& w = a.weights[o];
                    for (size_t k = 0; k < w.size(); ++k) {
                        const uint64_t at = vertical ? (uint64_t(a.first[o] + k) * rows + r) : (uint64_t(r) * in + a.first[o] + k);
                        sum += int32_t(src[at * 3 + c]) * w[k];
                    }
                    const uint64_t to = vertical ? (uint64_t(o) * rows + r) : (uint64_t(r) * out + o);
                    dst[to * 3 + c] = uint8_t(std::clamp(sum >> a.precision, 0, 255));
                }
    };
    std::vector<uint8_t> mid(uint64_t(img.height) * width * 3);
    pass(ax, img.rgb, img.height, img.width, width, false, mid);
    Image out{width, height, std::vector<uint8_t>(uint64_t(width) * height * 3)};
    pass(ay, mid, width, img.height, height, true, out.rgb);
    return out;
}

} // namespace mako::video
