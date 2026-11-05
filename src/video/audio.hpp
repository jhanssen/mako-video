#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace mako::video {

// One channel of a waveform from one sample rate to another: torchaudio's sinc interpolation with a Hann window
// (lowpass width 6, rolloff 0.99, the kernel in float), ceil(n * to / from) samples out.
std::vector<float> resample(std::span<const float> x, uint32_t from, uint32_t to);

} // namespace mako::video
