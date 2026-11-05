#include "video/audio.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>

namespace mako::video {

std::vector<float> resample(std::span<const float> x, uint32_t from, uint32_t to)
{
    if (!from || !to)
        throw std::runtime_error("resample: positive sample rates");
    if (from == to)
        return std::vector<float>(x.begin(), x.end());
    const uint32_t g = std::gcd(from, to), of = from / g, nf = to / g;
    constexpr int kWidth = 6;
    constexpr double kRolloff = 0.99;
    // Output phase j of every nf takes the kernel [j][k] over input samples m * of + k - width.
    const double base = double(std::min(of, nf)) * kRolloff, scale = base / double(of);
    const int width = int(std::ceil(kWidth * double(of) / base));
    const uint32_t K = uint32_t(2 * width) + of;
    std::vector<float> kernel(uint64_t(nf) * K);
    for (uint32_t j = 0; j < nf; ++j)
        for (uint32_t k = 0; k < K; ++k) {
            double t = (-double(j) / double(nf) + (double(k) - double(width)) / double(of)) * base;
            t = std::clamp(t, -double(kWidth), double(kWidth));
            const double window = std::pow(std::cos(t * std::numbers::pi / kWidth / 2.0), 2.0);
            t *= std::numbers::pi;
            kernel[uint64_t(j) * K + k] = float((t == 0.0 ? 1.0 : std::sin(t) / t) * window * scale);
        }
    const uint64_t n = x.size(), padded = n + uint64_t(2 * width) + of;
    std::vector<float> xp(padded, 0.0f);
    std::copy(x.begin(), x.end(), xp.begin() + width);
    const uint64_t frames = padded >= K ? (padded - K) / of + 1 : 0, want = (n * nf + of - 1) / of;
    std::vector<float> y(std::min(frames * nf, want));
    for (uint64_t m = 0; m < frames; ++m)
        for (uint32_t j = 0; j < nf; ++j) {
            const uint64_t o = m * nf + j;
            if (o >= y.size())
                break;
            const float* src = &xp[m * of];
            const float* kr = &kernel[uint64_t(j) * K];
            float s = 0.0f;
            for (uint32_t k = 0; k < K; ++k)
                s += src[k] * kr[k];
            y[o] = s;
        }
    return y;
}

} // namespace mako::video
